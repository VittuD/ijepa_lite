"""Convert an official original I-JEPA checkpoint into ijepa_lite format.

Example:
    python -m ijepa_lite.scripts.convert_original_ijepa_checkpoint \
      --input $WORK/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.pth.tar \
      --output $WORK/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.ijepa_lite.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from ijepa_lite.build import (
    _adapt_original_ijepa_encoder_state_dict,
    _adapt_original_ijepa_predictor_state_dict,
)
from ijepa_lite.losses.vanilla import VanillaTokenLoss
from ijepa_lite.models.ijepa import IJEPAModel
from ijepa_lite.models.predictor import Predictor
from ijepa_lite.models.vit_tokens import build_torchvision_vit_tokens


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert an official original I-JEPA checkpoint into an ijepa_lite "
            "checkpoint with matching encoder/predictor geometry."
        )
    )
    p.add_argument("--input", required=True, help="Path to the official I-JEPA checkpoint.")
    p.add_argument("--output", required=True, help="Path for the converted ijepa_lite checkpoint.")
    p.add_argument(
        "--encoder-num-heads",
        type=int,
        default=16,
        help="Encoder attention heads for the target ijepa_lite model.",
    )
    p.add_argument(
        "--predictor-num-heads",
        type=int,
        default=12,
        help="Predictor attention heads for the target ijepa_lite model.",
    )
    return p.parse_args()


def _infer_depth(sd: dict[str, torch.Tensor], pattern: str) -> int:
    max_idx = -1
    for key in sd:
        import re

        match = re.match(pattern, key)
        if match is not None:
            max_idx = max(max_idx, int(match.group(1)))
    if max_idx < 0:
        raise ValueError(f"Could not infer depth from pattern={pattern!r}.")
    return max_idx + 1


def _require_tensor_dict(payload: dict, key: str) -> dict[str, torch.Tensor]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint is missing dict payload[{key!r}].")
    out: dict[str, torch.Tensor] = {}
    for raw_key, tensor in value.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        norm_key = raw_key[len("module."):] if raw_key.startswith("module.") else raw_key
        out[norm_key] = tensor
    if not out:
        raise ValueError(f"Checkpoint payload[{key!r}] has no tensors.")
    return out


def _infer_model_spec(payload: dict, encoder_sd: dict[str, torch.Tensor], predictor_sd: dict[str, torch.Tensor]):
    patch_w = encoder_sd["patch_embed.proj.weight"]
    pos = encoder_sd["pos_embed"]
    pred_in = predictor_sd["predictor_embed.weight"]
    pred_pos = predictor_sd["predictor_pos_embed"]
    pred_fc1 = predictor_sd["predictor_blocks.0.mlp.fc1.weight"]

    embed_dim = int(patch_w.shape[0])
    patch_size = int(patch_w.shape[-1])
    grid_size = int(math.isqrt(int(pos.shape[1])))
    image_size = grid_size * patch_size
    encoder_depth = _infer_depth(encoder_sd, r"blocks\.(\d+)\.")
    predictor_depth = _infer_depth(predictor_sd, r"predictor_blocks\.(\d+)\.")
    predictor_dim = int(pred_in.shape[0])
    predictor_mlp_ratio = float(pred_fc1.shape[0]) / float(predictor_dim)

    if pred_pos.shape[1] != grid_size * grid_size:
        raise ValueError(
            "Encoder/predictor grid mismatch in original checkpoint: "
            f"encoder={grid_size * grid_size} predictor={pred_pos.shape[1]}"
        )

    return {
        "arch": "vit_huge",
        "image_size": image_size,
        "patch_size": patch_size,
        "embed_dim": embed_dim,
        "depth": encoder_depth,
        "predictor_dim": predictor_dim,
        "predictor_depth": predictor_depth,
        "predictor_mlp_ratio": predictor_mlp_ratio,
        "num_patches": grid_size * grid_size,
    }


def _build_model(spec: dict[str, float | int | str], encoder_num_heads: int, predictor_num_heads: int) -> IJEPAModel:
    model_cfg = SimpleNamespace(
        arch=str(spec["arch"]),
        image_size=int(spec["image_size"]),
        patch_size=int(spec["patch_size"]),
        embed_dim=int(spec["embed_dim"]),
        num_heads=int(encoder_num_heads),
        depth=int(spec["depth"]),
        remove_head=True,
        pos_embed_kind="learned",
        use_cls_token=False,
    )

    context = build_torchvision_vit_tokens(model_cfg)
    target = build_torchvision_vit_tokens(model_cfg)

    predictor = Predictor(
        dim=int(spec["embed_dim"]),
        predictor_dim=int(spec["predictor_dim"]),
        depth=int(spec["predictor_depth"]),
        num_heads=int(predictor_num_heads),
        mlp_ratio=float(spec["predictor_mlp_ratio"]),
        dropout=0.0,
        num_patches=int(spec["num_patches"]),
        pos_embed_kind="learned",
    )

    return IJEPAModel(
        context_encoder=context,
        target_encoder=target,
        predictor=predictor,
        loss_fn=VanillaTokenLoss(normalize=False, kind="mse"),
        ema_momentum=0.996,
        target_mode="ema",
        predict_blocks_jointly=True,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    payload = torch.load(str(input_path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected the original checkpoint payload to be a dict.")

    encoder_sd = _require_tensor_dict(payload, "encoder")
    target_encoder_sd = _require_tensor_dict(payload, "target_encoder")
    predictor_sd = _require_tensor_dict(payload, "predictor")

    spec = _infer_model_spec(payload, encoder_sd, predictor_sd)
    model = _build_model(spec, args.encoder_num_heads, args.predictor_num_heads)

    converted_model_sd = model.state_dict()

    context_adapted = _adapt_original_ijepa_encoder_state_dict(
        encoder_sd, model.context_encoder
    )
    target_adapted = _adapt_original_ijepa_encoder_state_dict(
        target_encoder_sd, model.target_encoder
    )
    predictor_adapted = _adapt_original_ijepa_predictor_state_dict(
        predictor_sd, model.predictor
    )

    for key, value in context_adapted.items():
        converted_model_sd[f"context_encoder.{key}"] = value
    for key, value in target_adapted.items():
        converted_model_sd[f"target_encoder.{key}"] = value
    for key, value in predictor_adapted.items():
        converted_model_sd[f"predictor.{key}"] = value

    incompatible = model.load_state_dict(converted_model_sd, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Converted checkpoint failed strict model reload: "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )

    converted_payload = {
        "model": model.state_dict(),
        "state": {},
        "conversion": {
            "source_checkpoint": str(input_path),
            "source_format": "original_ijepa",
            "required_model_overrides": {
                "model.arch": str(spec["arch"]),
                "model.image_size": int(spec["image_size"]),
                "model.patch_size": int(spec["patch_size"]),
                "model.embed_dim": int(spec["embed_dim"]),
                "model.num_heads": int(args.encoder_num_heads),
                "model.depth": int(spec["depth"]),
                "model.use_cls_token": False,
                "predictor.predictor_dim": int(spec["predictor_dim"]),
                "predictor.depth": int(spec["predictor_depth"]),
                "predictor.num_heads": int(args.predictor_num_heads),
                "predictor.mlp_ratio": float(spec["predictor_mlp_ratio"]),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted_payload, str(output_path))

    print(f"Saved converted checkpoint: {output_path}")
    print("Required overrides:")
    for key, value in converted_payload["conversion"]["required_model_overrides"].items():
        print(f"  {key}={value}")


if __name__ == "__main__":
    main()
