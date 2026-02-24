from __future__ import annotations

import math
from typing import Optional

import torch

from ijepa_lite.masking.base import MaskOutput


@torch.no_grad()
def mask_diagnostics(
    mask_output: MaskOutput,
    num_patches: int,
    masker_loss: Optional[torch.Tensor] = None,
    full: bool = False,
) -> dict[str, float]:
    """
    Compute diagnostic metrics from a MaskOutput.

    All returned keys are prefixed with "mask/".

    Args
    ----
    mask_output   : MaskOutput produced by any masker.
    num_patches   : Total patch positions N.
    masker_loss   : Scalar aux loss from the latent masker, or None.
    full          : When False, only compute cheap always-on metrics.
                    When True, also compute richer spatial and distributional
                    diagnostics.  Pass full=True only at log steps.

    Metric catalogue
    ----------------
    Always computed
    ~~~~~~~~~~~~~~~
    mask/nctx              Number of context patches per sample.
    mask/ntgt              Total target patches per sample.
    mask/nblocks           Number of target blocks.
    mask/context_ratio     nctx / N.
    mask/target_ratio      ntgt / N.
    mask/masker_loss       Aux loss scalar (0.0 for deterministic maskers).

    RD masker — always (keys present in aux after aux_loss):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mask/lambda            λ value used this step.
    mask/D_soft            Soft-weighted distortion D_soft.
    mask/rate              R — expected context fraction ∈ [0, 1].
    mask/expected_nctx     Expected context count = R · N.
    mask/expected_ntgt     Expected target count  = sum(p_tgt) / N · N.
    mask/expected_nign     Expected ignored count = N - expected_nctx - expected_ntgt.

    Full diagnostics — both masker types (full=True):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mask/spatial_coverage  Fraction of N positions hit by ≥1 batch sample.
    mask/batch_iou         Mean pairwise Jaccard similarity across batch.

    Full diagnostics — 2-way learned maskers (target_soft present, no p_ign):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mask/selection_entropy
    mask/selection_entropy_norm
    mask/effective_patches
    mask/effective_patches_norm
    mask/score_max_mean
    mask/topk_mass_ratio

    Full diagnostics — 3-way RD masker (p_ign present in aux):
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mask/entropy_ctx       Shannon entropy of p_ctx distribution.
    mask/entropy_tgt       Shannon entropy of p_tgt distribution.
    mask/entropy_ign       Shannon entropy of p_ign distribution.
    mask/entropy_3way      Mean per-patch entropy of the 3-way categorical.
                           High = uncertain/uniform assignments.
                           Low  = confident/decisive assignments.
    """
    stats: dict[str, float] = {}

    ctx_idx = mask_output.context_idx   # (B, Nctx)
    tgt_idx = mask_output.target_idx    # (B, Ntgt) or (B, M, K)

    nctx = ctx_idx.shape[1]
    tgt_flat = tgt_idx.reshape(tgt_idx.shape[0], -1)
    ntgt_total = tgt_flat.shape[1]
    B = tgt_flat.shape[0]
    nblocks = float(tgt_idx.shape[1]) if tgt_idx.dim() == 3 else 1.0

    # ------------------------------------------------------------------
    # Always-on metrics
    # ------------------------------------------------------------------
    stats["mask/nctx"] = float(nctx)
    stats["mask/ntgt"] = float(ntgt_total)
    stats["mask/nblocks"] = nblocks
    stats["mask/context_ratio"] = float(nctx) / num_patches
    stats["mask/target_ratio"] = float(ntgt_total) / num_patches
    stats["mask/masker_loss"] = float(masker_loss.item()) if masker_loss is not None else 0.0

    # ------------------------------------------------------------------
    # RD masker — read from aux (written by aux_loss in-place)
    # ------------------------------------------------------------------
    aux = mask_output.aux

    if "lambda" in aux:
        lam = aux["lambda"]
        stats["mask/lambda"] = float(lam.mean().item()) if torch.is_tensor(lam) else float(lam)

    if "D_soft" in aux:
        stats["mask/D_soft"] = float(aux["D_soft"])

    if "R" in aux:
        R = float(aux["R"])
        stats["mask/rate"] = R
        stats["mask/expected_nctx"] = R * num_patches

    # Expected target and ignore counts from soft probabilities
    p_tgt = mask_output.target_soft   # (B, N) or None
    p_ctx = mask_output.context_soft  # (B, N) or None
    p_ign = aux.get("p_ign", None)    # (B, N) or None — 3-way masker only

    if p_tgt is not None:
        stats["mask/expected_ntgt"] = float(p_tgt.sum(dim=-1).mean().item())

    if p_ign is not None:
        stats["mask/expected_nign"] = float(p_ign.sum(dim=-1).mean().item())

    # Surprise metrics — always log when present
    if "surprise_mean" in aux:
        stats["mask/surprise_mean"] = float(aux["surprise_mean"])

    if not full:
        return stats

    # ------------------------------------------------------------------
    # Full diagnostics — both masker types
    # ------------------------------------------------------------------
    device = tgt_flat.device
    binary = torch.zeros(B, num_patches, device=device, dtype=torch.float32)
    binary.scatter_(1, tgt_flat, 1.0)

    stats["mask/spatial_coverage"] = float(
        binary.sum(dim=0).gt(0).float().mean().item()
    )

    if B > 1:
        inter = binary @ binary.T
        ntgt_vec = binary.sum(dim=1, keepdim=True)
        union = ntgt_vec + ntgt_vec.T - inter
        iou = inter / union.clamp(min=1.0)
        mask_upper = torch.triu(
            torch.ones(B, B, device=device, dtype=torch.bool), diagonal=1
        )
        stats["mask/batch_iou"] = float(iou[mask_upper].mean().item())
    else:
        stats["mask/batch_iou"] = 1.0

    # ------------------------------------------------------------------
    # Full diagnostics — 3-way masker (p_ign present)
    # ------------------------------------------------------------------
    if p_ign is not None and p_ctx is not None and p_tgt is not None:
        p_ctx_f = p_ctx.float()
        p_tgt_f = p_tgt.float()
        p_ign_f = p_ign.float()

        def _entropy(p: torch.Tensor) -> float:
            return float(-(p * (p + 1e-10).log()).sum(dim=-1).mean().item())

        stats["mask/entropy_ctx"] = _entropy(p_ctx_f)
        stats["mask/entropy_tgt"] = _entropy(p_tgt_f)
        stats["mask/entropy_ign"] = _entropy(p_ign_f)

        # Per-patch 3-way categorical entropy, averaged over patches and batch.
        # Uses the 3-way soft from the stacked distribution (B, N, 3).
        # Note: logits are in aux but we reconstruct from the soft values to
        # avoid storing the full (B, N, 3) tensor after detach.
        soft_3way = torch.stack([p_ctx_f, p_tgt_f, p_ign_f], dim=-1)  # (B, N, 3)
        per_patch_H = -(soft_3way * (soft_3way + 1e-10).log()).sum(dim=-1)  # (B, N)
        stats["mask/entropy_3way"] = float(per_patch_H.mean().item())
        return stats

    # ------------------------------------------------------------------
    # Full diagnostics — 2-way learned masker (target_soft, no p_ign)
    # ------------------------------------------------------------------
    if p_tgt is None:
        return stats

    soft = p_tgt.float()

    entropy = -(soft * (soft + 1e-10).log()).sum(dim=-1).mean()
    stats["mask/selection_entropy"] = float(entropy.item())
    stats["mask/selection_entropy_norm"] = float(entropy.item()) / math.log(num_patches)

    effective = (1.0 / (soft.pow(2).sum(dim=-1) + 1e-12)).mean()
    stats["mask/effective_patches"] = float(effective.item())
    stats["mask/effective_patches_norm"] = float(effective.item()) / num_patches

    stats["mask/score_max_mean"] = float(soft.max(dim=-1).values.mean().item())

    topk_mass = soft.topk(ntgt_total, dim=-1).values.sum(dim=-1).mean()
    uniform_baseline = ntgt_total / num_patches
    stats["mask/topk_mass_ratio"] = float(topk_mass.item()) / uniform_baseline

    return stats