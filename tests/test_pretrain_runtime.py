from __future__ import annotations

import torch
import torch.nn as nn

from ijepa_lite.losses.vanilla import VanillaTokenLoss
from ijepa_lite.masking.base import (
    LatentMasker,
    MaskOutput,
    MaskPartition,
    TwoWayAssignment,
)
from ijepa_lite.masking.compressor import TokenCompressor
from ijepa_lite.models.ijepa import IJEPAModel


class _TinyEncoder(nn.Module):
    def __init__(self, num_patches: int = 4, dim: int = 3) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.dim = dim
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(
        self,
        images: torch.Tensor,
        keep_idx: torch.Tensor | None = None,
        return_cls: bool = False,
    ):
        self.forward_calls += 1
        batch_size = images.shape[0]
        image_value = images.mean(dim=(1, 2, 3)).reshape(batch_size, 1, 1)
        positions = torch.arange(
            self.num_patches,
            device=images.device,
            dtype=images.dtype,
        ).reshape(1, self.num_patches, 1)
        channels = torch.arange(
            self.dim,
            device=images.device,
            dtype=images.dtype,
        ).reshape(1, 1, self.dim)
        tokens = self.scale * image_value + positions + channels
        if keep_idx is not None:
            tokens = tokens.gather(
                1,
                keep_idx.unsqueeze(-1).expand(-1, -1, self.dim),
            )
        if return_cls:
            return tokens.mean(dim=1), tokens
        return tokens


class _TinyPredictor(nn.Module):
    def __init__(self, dim: int = 3) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        context: torch.Tensor,
        *,
        ctx_idx: torch.Tensor,
        tgt_idx: torch.Tensor,
        return_ctx_pred: bool = False,
    ):
        del ctx_idx
        pooled = context.mean(dim=1, keepdim=True)
        pred = pooled.expand(-1, tgt_idx.shape[1], -1) + self.bias
        if return_ctx_pred:
            return pred, context + self.bias
        return pred


class _FixedLatentMasker(LatentMasker):
    needs_full_tokens = True

    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        tokens: torch.Tensor,
        ema_full: torch.Tensor | None = None,
    ) -> MaskOutput:
        del ema_full
        batch_size, num_patches, _ = tokens.shape
        target = torch.sigmoid(self.logit).expand(batch_size, num_patches)
        context = 1.0 - target
        device = tokens.device
        return MaskOutput(
            partition=MaskPartition(
                context_idx=torch.tensor([[0, 1]], device=device).expand(
                    batch_size, -1
                ),
                target_idx=torch.tensor([[2, 3]], device=device).expand(
                    batch_size, -1
                ),
            ),
            assignment=TwoWayAssignment(context=context, target=target),
        )

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask_output, reconstruction_loss, patch_loss
        return self.logit.square() + self.logit


def _build_model(latent_masker: LatentMasker | None = None) -> IJEPAModel:
    return IJEPAModel(
        context_encoder=_TinyEncoder(),
        target_encoder=_TinyEncoder(),
        predictor=_TinyPredictor(),
        loss_fn=VanillaTokenLoss(),
        ema_momentum=0.99,
        latent_masker=latent_masker,
        token_compressor=(
            TokenCompressor(mode="full", dim=3)
            if latent_masker is not None
            else None
        ),
        grid_size=2,
    )


def test_deterministic_forward_preserves_public_result_contract() -> None:
    model = _build_model()
    images = torch.ones(2, 1, 2, 2)
    masks = {
        "context_idx": torch.tensor([[0, 1], [0, 1]]),
        "target_idx": torch.tensor([[2, 3], [2, 3]]),
    }

    output = model(images, masks=masks, compute_agreement=True)

    assert set(output) == {
        "loss",
        "reconstruction_loss",
        "pred",
        "target",
        "patch_loss",
        "mask_stats",
        "model_stats",
        "ctx_loss",
        "ctx_tokens_all",
        "tgt_tokens_all",
    }
    assert output["pred"].shape == (2, 2, 3)
    assert output["target"].shape == (2, 2, 3)
    assert output["patch_loss"].shape == (2, 2)
    assert output["mask_stats"]["mask/nctx"] == 2.0
    assert output["mask_stats"]["mask/ntgt"] == 2.0


def test_latent_forward_reuses_target_tokens_and_routes_masker_gradient() -> None:
    masker = _FixedLatentMasker()
    model = _build_model(masker)
    images = torch.ones(2, 1, 2, 2)

    output = model(images)
    output["loss"].backward()

    assert model.target_encoder.forward_calls == 1
    assert masker.logit.grad is not None
    assert model.context_encoder.scale.grad is not None
    assert model.target_encoder.scale.grad is None


def test_multiblock_forward_respects_unpadded_target_counts() -> None:
    model = _build_model()
    images = torch.ones(2, 1, 2, 2)
    masks = {
        "context_idx": torch.tensor([[0, 1], [0, 1]]),
        "target_idx": torch.tensor(
            [
                [[2, 2, 2], [3, 2, 3]],
                [[2, 2, 2], [3, 2, 3]],
            ]
        ),
        "target_block_counts": torch.tensor([1, 3]),
    }

    output = model(images, masks=masks)

    assert output["pred"].shape == (2, 2, 3, 3)
    assert output["patch_loss"].shape == (2, 4)
    assert output["mask_stats"]["mask/ntgt"] == 4.0


def test_runtime_does_not_own_checkpoint_parameters() -> None:
    model = _build_model(_FixedLatentMasker())

    assert set(model.state_dict()) == {
        "context_encoder.scale",
        "target_encoder.scale",
        "predictor.bias",
        "latent_masker.logit",
    }
