"""Inspect checkpoint tensor structure for downstream encoder comparison.

Example:
    python -m ijepa_lite.scripts.inspect_checkpoint_shapes \
      --checkpoint ours=$WORK/vitturini/outputs/2026-05-04/19-42-25_vanilla_baseline_multiblock_stl10/rank0/checkpoints/epoch_00999.pt \
      --checkpoint orig=$WORK/vitturini/original_jepa_weights/IN1K-vit.h.14-300e.pth.tar
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import torch


KNOWN_ENCODER_PREFIXES = (
    "target_encoder.vit.",
    "target_encoder.",
    "ema_encoder.vit.",
    "ema_encoder.",
    "context_encoder.vit.",
    "context_encoder.",
    "encoder.vit.",
    "encoder.",
    "vit.",
    "backbone.",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inspect checkpoint tensor structure and encoder-shaped subtrees. "
            "Useful for comparing our JEPA checkpoints against original I-JEPA ones."
        )
    )
    p.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Checkpoint spec in the form label=/abs/or/rel/path/to/checkpoint.",
    )
    p.add_argument(
        "--max-keys",
        type=int,
        default=20,
        help="How many example tensor keys to print per selected encoder subtree.",
    )
    return p.parse_args()


def _parse_checkpoint_specs(specs: Iterable[str]) -> list[tuple[str, Path]]:
    out: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --checkpoint '{spec}'. Expected label=/path/to/checkpoint."
            )
        label, raw_path = spec.split("=", 1)
        out.append((label, Path(raw_path).expanduser().resolve()))
    return out


def _is_tensor_dict(obj) -> bool:
    return isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values())


def _load_payload(path: Path):
    return torch.load(str(path), map_location="cpu", weights_only=True)


def _print_top_level_summary(label: str, payload) -> None:
    print(f"## {label}")
    print(f"payload_type: {type(payload).__name__}")
    if isinstance(payload, dict):
        print(f"top_level_keys ({len(payload)}): {list(payload.keys())[:20]}")
        for key in ("model", "target_encoder", "context_encoder", "encoder", "ema_encoder", "state"):
            if key in payload:
                value = payload[key]
                kind = type(value).__name__
                extra = ""
                if isinstance(value, dict):
                    extra = f" len={len(value)}"
                print(f"  - payload[{key!r}] -> {kind}{extra}")
    else:
        print("top_level_keys: <not a dict>")


def _normalize_state_dict(sd: dict) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        out[key] = value
    return out


def _extract_candidate_state_dicts(payload) -> list[tuple[str, dict[str, torch.Tensor]]]:
    candidates: list[tuple[str, dict[str, torch.Tensor]]] = []

    if _is_tensor_dict(payload):
        candidates.append(("payload", _normalize_state_dict(payload)))

    if isinstance(payload, dict):
        for key in ("model", "target_encoder", "context_encoder", "encoder", "ema_encoder"):
            value = payload.get(key)
            if _is_tensor_dict(value):
                candidates.append((f"payload.{key}", _normalize_state_dict(value)))

    dedup: dict[tuple[str, int], tuple[str, dict[str, torch.Tensor]]] = {}
    for name, sd in candidates:
        dedup[(name, len(sd))] = (name, sd)
    return list(dedup.values())


def _collect_prefix_views(sd: dict[str, torch.Tensor]) -> list[tuple[str, dict[str, torch.Tensor]]]:
    out: list[tuple[str, dict[str, torch.Tensor]]] = []
    for prefix in KNOWN_ENCODER_PREFIXES:
        view = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if view:
            out.append((prefix, view))
    return out


def _first_matching_key(sd: dict[str, torch.Tensor], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key in sd:
            return key
    return None


def _infer_depth(sd: dict[str, torch.Tensor]) -> int | None:
    max_idx = -1
    for key in sd:
        parts = key.split(".")
        for idx, part in enumerate(parts[:-1]):
            if part in ("layers", "blocks") and idx + 1 < len(parts):
                try:
                    max_idx = max(max_idx, int(parts[idx + 1]))
                except ValueError:
                    pass
    return max_idx + 1 if max_idx >= 0 else None


def _format_shape(t: torch.Tensor) -> str:
    return str(tuple(t.shape))


def _infer_encoder_shape(sd: dict[str, torch.Tensor]) -> dict[str, object]:
    patch_key = _first_matching_key(
        sd,
        (
            "vit.conv_proj.weight",
            "conv_proj.weight",
            "patch_embed.proj.weight",
            "patch_embed.weight",
        ),
    )
    pos_key = _first_matching_key(
        sd,
        (
            "vit.encoder.pos_embedding",
            "encoder.pos_embedding",
            "pos_embed",
            "vit.pos_embed",
        ),
    )
    cls_key = _first_matching_key(sd, ("vit.class_token", "class_token", "cls_token"))

    info: dict[str, object] = {
        "num_tensors": len(sd),
        "patch_key": patch_key,
        "pos_key": pos_key,
        "cls_key": cls_key,
        "depth": _infer_depth(sd),
    }

    if patch_key is not None:
        patch_w = sd[patch_key]
        if patch_w.ndim == 4:
            info["patch_proj_shape"] = tuple(patch_w.shape)
            info["embed_dim"] = int(patch_w.shape[0])
            info["in_chans"] = int(patch_w.shape[1])
            info["patch_hw"] = tuple(int(x) for x in patch_w.shape[-2:])

    if pos_key is not None:
        pos = sd[pos_key]
        if pos.ndim == 3:
            info["pos_shape"] = tuple(pos.shape)
            tokens = int(pos.shape[1])
            info["pos_tokens"] = tokens
            if cls_key is not None:
                patch_tokens = tokens - 1
                info["pos_has_cls"] = True
            else:
                patch_tokens = tokens
                info["pos_has_cls"] = False
            grid = int(math.isqrt(max(patch_tokens, 0)))
            if grid * grid == patch_tokens:
                info["grid_hw"] = (grid, grid)

    return info


def _print_encoder_summary(name: str, sd: dict[str, torch.Tensor], max_keys: int) -> None:
    info = _infer_encoder_shape(sd)
    print(f"### encoder view: {name}")
    print(f"num_tensors: {info['num_tensors']}")
    if info.get("patch_key") is not None:
        print(f"patch_proj_key: {info['patch_key']}")
        print(f"patch_proj_shape: {info.get('patch_proj_shape')}")
        print(f"embed_dim: {info.get('embed_dim')}")
        print(f"in_chans: {info.get('in_chans')}")
        print(f"patch_hw: {info.get('patch_hw')}")
    if info.get("pos_key") is not None:
        print(f"pos_key: {info['pos_key']}")
        print(f"pos_shape: {info.get('pos_shape')}")
        print(f"pos_tokens: {info.get('pos_tokens')}")
        print(f"pos_has_cls: {info.get('pos_has_cls')}")
        print(f"grid_hw: {info.get('grid_hw')}")
    print(f"cls_key: {info.get('cls_key')}")
    print(f"depth_hint: {info.get('depth')}")
    print("example_keys:")
    for key in sorted(sd.keys())[:max_keys]:
        print(f"  {key:<60} {_format_shape(sd[key])}")


def _pick_best_encoder_view(
    candidates: list[tuple[str, dict[str, torch.Tensor]]],
) -> tuple[str, dict[str, torch.Tensor]] | None:
    best_name = None
    best_sd = None
    best_score = -1
    for candidate_name, sd in candidates:
        for prefix_name, pref_sd in _collect_prefix_views(sd):
            score = len(pref_sd)
            if score > best_score:
                best_name = f"{candidate_name}:{prefix_name}"
                best_sd = pref_sd
                best_score = score
        if len(sd) > best_score:
            best_name = candidate_name
            best_sd = sd
            best_score = len(sd)
    if best_name is None or best_sd is None:
        return None
    return best_name, best_sd


def _print_comparison(best_views: list[tuple[str, dict[str, torch.Tensor]]]) -> None:
    if len(best_views) < 2:
        return
    print("## comparison")
    print("label | tensors | patch_proj | pos_embed | cls | depth")
    print("----- | ------- | ---------- | --------- | --- | -----")
    for label, sd in best_views:
        info = _infer_encoder_shape(sd)
        print(
            f"{label} | "
            f"{info['num_tensors']} | "
            f"{info.get('patch_proj_shape', '-')} | "
            f"{info.get('pos_shape', '-')} | "
            f"{'yes' if info.get('cls_key') else 'no'} | "
            f"{info.get('depth', '-')}"
        )


def main() -> None:
    args = parse_args()
    specs = _parse_checkpoint_specs(args.checkpoint)
    best_views: list[tuple[str, dict[str, torch.Tensor]]] = []

    for label, path in specs:
        print()
        print(f"{'=' * 24} {label} {'=' * 24}")
        print(f"path: {path}")
        payload = _load_payload(path)
        _print_top_level_summary(label, payload)

        candidates = _extract_candidate_state_dicts(payload)
        if not candidates:
            print("no tensor dictionaries found in payload")
            continue

        print("candidate_state_dicts:")
        for candidate_name, sd in candidates:
            prefix_views = _collect_prefix_views(sd)
            prefix_counts = ", ".join(f"{prefix}:{len(view)}" for prefix, view in prefix_views[:6])
            print(f"  - {candidate_name}: {len(sd)} tensors")
            if prefix_counts:
                print(f"    prefix_matches: {prefix_counts}")

        best = _pick_best_encoder_view(candidates)
        if best is None:
            print("could not identify an encoder-shaped subtree")
            continue

        best_name, best_sd = best
        _print_encoder_summary(best_name, best_sd, max_keys=args.max_keys)
        best_views.append((label, best_sd))

    if best_views:
        print()
        _print_comparison(best_views)


if __name__ == "__main__":
    main()
