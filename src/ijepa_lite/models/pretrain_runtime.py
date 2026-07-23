from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F

from ijepa_lite.losses.context_loss import context_loss
from ijepa_lite.masking.base import MaskOutput
from ijepa_lite.masking.metrics import mask_diagnostics


@dataclass(frozen=True)
class PretrainStepRequest:
    images: torch.Tensor
    masks: Optional[dict[str, torch.Tensor]] = None
    compute_agreement: bool = False
    compute_mask_metrics: bool = False
    epoch: int = 0


@dataclass
class PretrainStepResult:
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    patch_loss: torch.Tensor
    mask_stats: dict[str, float]
    model_stats: dict[str, float]
    ctx_loss: Optional[float]
    sigreg_loss_weighted: Optional[torch.Tensor] = None
    include_agreement: bool = False
    ctx_tokens_all: Optional[torch.Tensor] = None
    tgt_tokens_all: Optional[torch.Tensor] = None

    def as_dict(self) -> dict[str, Any]:
        output: dict[str, Any] = {
            "loss": self.loss,
            "reconstruction_loss": self.reconstruction_loss,
            "pred": self.pred,
            "target": self.target,
            "patch_loss": self.patch_loss,
            "mask_stats": self.mask_stats,
            "model_stats": self.model_stats,
            "ctx_loss": self.ctx_loss,
        }
        if self.sigreg_loss_weighted is not None:
            output["sigreg_loss_weighted"] = self.sigreg_loss_weighted
        if self.include_agreement:
            output["ctx_tokens_all"] = self.ctx_tokens_all
            output["tgt_tokens_all"] = self.tgt_tokens_all
        return output


class PretrainingModel(Protocol):
    latent_masker: Any
    context_encoder: Any
    sigreg_loss: Any
    sigreg_weight_start: float
    sigreg_weight_end: float
    sigreg_weight_schedule: str
    ctx_loss_weight: float
    ctx_loss_gamma: float
    grid_size: int
    loss_fn: Any

    def _resolve_latent_masks(
        self,
        images: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> tuple[MaskOutput, torch.Tensor]: ...

    def _resolve_collate_masks(
        self,
        masks: Optional[dict[str, torch.Tensor]],
        images: torch.Tensor,
    ) -> MaskOutput: ...

    def _encode_full_view_tokens(self, images: torch.Tensor) -> torch.Tensor: ...

    def _uses_projected_prediction_space(self) -> bool: ...

    def _forward_single_block(self, *args: Any, **kwargs: Any) -> tuple: ...

    def _forward_multi_block(self, *args: Any, **kwargs: Any) -> tuple: ...

    def _flatten_multi_block_target_idx(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def _project_prediction_tokens(self, tokens: torch.Tensor) -> torch.Tensor: ...

    def _get_sigreg_weight(self, epoch: int) -> float: ...

    def _ctx_alpha(self, epoch: int) -> float: ...


def run_pretraining_step(
    model: PretrainingModel,
    request: PretrainStepRequest,
) -> PretrainStepResult:
    """Run one JEPA pretraining forward pass without owning model state."""

    images = request.images
    cached_full_tokens: Optional[torch.Tensor] = None
    if model.latent_masker is not None:
        mask_output, cached_full_tokens = model._resolve_latent_masks(
            images,
            epoch=request.epoch,
        )
    else:
        mask_output = model._resolve_collate_masks(request.masks, images)

    ctx_idx = mask_output.context_idx
    tgt_idx = mask_output.target_idx
    target_block_counts = mask_output.partition.target_block_counts

    ctx_tokens = model.context_encoder(images, keep_idx=ctx_idx)
    batch_size = images.shape[0]
    embed_dim = ctx_tokens.shape[-1]

    full_tokens = (
        cached_full_tokens
        if cached_full_tokens is not None
        else model._encode_full_view_tokens(images)
    )
    raw_tgt_tokens_all = F.layer_norm(full_tokens, (full_tokens.shape[-1],))
    if model._uses_projected_prediction_space():
        tgt_tokens_all = model.sigreg_loss.project_tokens(full_tokens)
    else:
        tgt_tokens_all = raw_tgt_tokens_all

    ctx_tokens_all: Optional[torch.Tensor] = None
    tgt_at_ctx: Optional[torch.Tensor] = None
    if request.compute_agreement:
        ctx_tokens_all = ctx_tokens.detach()
        tgt_at_ctx = raw_tgt_tokens_all.gather(
            1,
            ctx_idx.unsqueeze(-1).expand(
                -1, -1, raw_tgt_tokens_all.shape[-1]
            ),
        ).detach()

    if tgt_idx.dim() == 2:
        reconstruction_loss, pred, tgt_tokens, patch_loss, pred_ctx = (
            model._forward_single_block(
                ctx_tokens,
                ctx_idx,
                tgt_idx,
                tgt_tokens_all,
                batch_size,
                embed_dim,
            )
        )
    elif tgt_idx.dim() == 3:
        reconstruction_loss, pred, tgt_tokens, patch_loss, pred_ctx = (
            model._forward_multi_block(
                ctx_tokens,
                ctx_idx,
                tgt_idx,
                tgt_tokens_all,
                batch_size,
                embed_dim,
                target_block_counts=target_block_counts,
            )
        )
    else:
        raise ValueError(
            f"Unsupported target_idx.dim()={tgt_idx.dim()}, expected 2 or 3."
        )

    if model.latent_masker is not None:
        masker_aux = model.latent_masker.aux_loss(
            mask_output,
            reconstruction_loss,
            patch_loss=patch_loss,
        )
        total_loss = (
            masker_aux
            if model.latent_masker.owns_loss
            else reconstruction_loss + masker_aux
        )
    else:
        masker_aux = reconstruction_loss.new_zeros(())
        total_loss = reconstruction_loss

    model_stats: dict[str, float] = {}
    sigreg_loss_weighted: Optional[torch.Tensor] = None
    sigreg_weight = model._get_sigreg_weight(request.epoch)
    if model.sigreg_loss is not None and sigreg_weight > 0.0:
        sigreg_val, sigreg_logs = model.sigreg_loss.forward_projected(
            tgt_tokens_all,
            full_tokens,
        )
        sigreg_loss_weighted = sigreg_weight * sigreg_val
        total_loss = total_loss + sigreg_loss_weighted
        model_stats.update(sigreg_logs)
        model_stats["sigreg/weight"] = float(sigreg_weight)
        model_stats["sigreg/weight_start"] = float(model.sigreg_weight_start)
        model_stats["sigreg/weight_end"] = float(model.sigreg_weight_end)
        model_stats["sigreg/weight_schedule_id"] = {
            "constant": 0.0,
            "linear": 1.0,
            "cosine": 2.0,
        }[model.sigreg_weight_schedule]
        model_stats["sigreg/loss_weighted"] = float(
            sigreg_loss_weighted.detach().item()
        )

    ctx_loss_value: Optional[float] = None
    if pred_ctx is not None:
        tgt_idx_flat = (
            model._flatten_multi_block_target_idx(
                tgt_idx,
                batch_size,
                target_block_counts,
            )
            if tgt_idx.dim() == 3
            else tgt_idx
        )
        tgt_at_ctx_pos = tgt_tokens_all.gather(
            1,
            ctx_idx.unsqueeze(-1).expand(-1, -1, tgt_tokens_all.shape[-1]),
        )
        pred_ctx_loss = model._project_prediction_tokens(pred_ctx)
        ctx_value = context_loss(
            pred_ctx_loss,
            tgt_at_ctx_pos,
            ctx_idx,
            tgt_idx_flat,
            model.grid_size,
            model.loss_fn,
            gamma=model.ctx_loss_gamma,
            alpha=model._ctx_alpha(request.epoch),
        )
        ctx_loss_value = float(ctx_value.detach().item())
        total_loss = total_loss + model.ctx_loss_weight * ctx_value

    mask_stats = mask_diagnostics(
        mask_output,
        num_patches=tgt_tokens_all.shape[1],
        masker_loss=masker_aux,
        full=request.compute_mask_metrics,
        patch_loss=patch_loss,
    )

    return PretrainStepResult(
        loss=total_loss,
        reconstruction_loss=reconstruction_loss,
        pred=pred.detach(),
        target=tgt_tokens.detach(),
        patch_loss=patch_loss.detach(),
        mask_stats=mask_stats,
        model_stats=model_stats,
        ctx_loss=ctx_loss_value,
        sigreg_loss_weighted=sigreg_loss_weighted,
        include_agreement=request.compute_agreement,
        ctx_tokens_all=ctx_tokens_all,
        tgt_tokens_all=tgt_at_ctx,
    )
