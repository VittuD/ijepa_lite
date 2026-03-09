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
    patch_loss: Optional[torch.Tensor] = None,
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

    if "alpha" in aux:
        alpha = aux["alpha"]
        stats["mask/alpha"] = float(alpha.mean().item()) if torch.is_tensor(alpha) else float(alpha)

    if "beta" in aux:
        beta = aux["beta"]
        stats["mask/beta"] = float(beta.mean().item()) if torch.is_tensor(beta) else float(beta)

    if "lambda_tgt" in aux:
        lam_tgt = aux["lambda_tgt"]
        stats["mask/lambda_tgt"] = float(lam_tgt.mean().item()) if torch.is_tensor(lam_tgt) else float(lam_tgt)

    if "D_soft" in aux:
        stats["mask/D_soft"] = float(aux["D_soft"])

    if "R" in aux:
        R = float(aux["R"])
        stats["mask/rate"] = R
        stats["mask/expected_nctx"] = R * num_patches

    # Hard ignore count (complements the always-on nctx / ntgt)
    stats["mask/nign"] = float(num_patches - nctx - ntgt_total)

    # Expected counts from soft probabilities
    p_tgt = mask_output.target_soft   # (B, N) or None
    p_ctx = mask_output.context_soft  # (B, N) or None
    p_ign = aux.get("p_ign", None)    # (B, N) or None — 3-way masker only

    if p_ctx is not None:
        stats["mask/expected_nctx"] = float(p_ctx.sum(dim=-1).mean().item())

    if p_tgt is not None:
        stats["mask/expected_ntgt"] = float(p_tgt.sum(dim=-1).mean().item())

    if p_ign is not None:
        stats["mask/expected_nign"] = float(p_ign.sum(dim=-1).mean().item())

    # Surprise and ignore-tax metrics — always log when present
    if "surprise_mean" in aux:
        stats["mask/surprise_mean"] = float(aux["surprise_mean"])

    if "ign_rate" in aux:
        stats["mask/ign_rate"] = float(aux["ign_rate"])

    if "mi_rate" in aux:
        stats["mask/mi_rate"] = float(aux["mi_rate"])

    if "entropy_conditional" in aux:
        stats["mask/entropy_conditional"] = float(aux["entropy_conditional"])

    if "entropy_marginal" in aux:
        stats["mask/entropy_marginal"] = float(aux["entropy_marginal"])

    if "floor_penalty" in aux:
        stats["mask/floor_penalty"] = float(aux["floor_penalty"])

    # Goldilocks masker — sampled budgets and content-adaptivity diagnostics
    for key in ("k_tgt", "k_ctx",
                "tgt_pos_std", "tgt_pos_std_norm",
                "p_tgt_score_std", "marginal_score_std",
                "tgt_assignment_entropy",
                "batch_iou", "batch_iou_random"):
        if key in aux:
            stats[f"mask/{key}"] = float(aux[key])

    # ------------------------------------------------------------------
    # Goldilocks error distribution diagnostics
    #
    # patch_loss is (B, K) per-patch reconstruction error at target positions.
    # These stats characterise the error landscape the masker trains on:
    #   - mean, std: overall difficulty level and spread
    #   - skewness: positive = long tail of hard patches; near 0 = symmetric
    #   - kurtosis: heavy tails (> 0) vs. light tails (< 0) relative to Gaussian
    #   - percentile bins: histogram of difficulty distribution
    #   - score-error correlation: whether high-scoring patches are actually harder
    #   - z-score stats: properties of the normalised signal the loss sees
    # ------------------------------------------------------------------
    if patch_loss is not None and "k_tgt" in aux:
        pl = patch_loss.float()
        if pl.dim() == 3:
            pl = pl.mean(-1)  # (B, K)

        flat = pl.reshape(-1)  # pool across batch for robust stats
        n = flat.numel()

        pl_mean = flat.mean()
        pl_std = flat.std()
        stats["goldilocks/error_mean"] = float(pl_mean.item())
        stats["goldilocks/error_std"] = float(pl_std.item())

        # Skewness and kurtosis (excess, Fisher definition)
        if n > 2 and pl_std > 1e-12:
            centered = flat - pl_mean
            m3 = (centered.pow(3)).mean()
            m4 = (centered.pow(4)).mean()
            stats["goldilocks/error_skewness"] = float((m3 / pl_std.pow(3)).item())
            stats["goldilocks/error_kurtosis"] = float((m4 / pl_std.pow(4) - 3.0).item())

        # Raw error values for histogram (wandb.Histogram in WandbCallback).
        # Cap at 4096 samples to keep serialization light.
        if n > 4096:
            idx = torch.randperm(n, device=flat.device)[:4096]
            stats["_hist/goldilocks/error"] = flat[idx].cpu().numpy()
        else:
            stats["_hist/goldilocks/error"] = flat.cpu().numpy()

        # Log-transformed error histogram (always computed for comparison)
        log_err = torch.log(flat.clamp(min=1e-8))
        if n > 4096:
            idx_log = torch.randperm(n, device=flat.device)[:4096]
            stats["_hist/goldilocks/log_error"] = log_err[idx_log].cpu().numpy()
        else:
            stats["_hist/goldilocks/log_error"] = log_err.cpu().numpy()

        # Z-scored error stats (what the Goldilocks loss actually sees)
        # Per-sample z-score to match the local z-score branch
        mu_s = pl.mean(dim=1, keepdim=True)
        sigma_s = pl.std(dim=1, keepdim=True).clamp(min=1e-6)
        z = (pl - mu_s) / sigma_s  # (B, K)
        stats["goldilocks/z_std_mean"] = float(sigma_s.mean().item())
        stats["goldilocks/z_range"] = float((z.max() - z.min()).item())
        # Fraction of patches near zero z-score (|z| < 0.5) — the "Goldilocks zone"
        stats["goldilocks/z_goldilocks_frac"] = float((z.abs() < 0.5).float().mean().item())

        # Z-score histogram
        z_flat = z.reshape(-1)
        if z_flat.numel() > 4096:
            idx = torch.randperm(z_flat.numel(), device=z_flat.device)[:4096]
            stats["_hist/goldilocks/z_score"] = z_flat[idx].cpu().numpy()
        else:
            stats["_hist/goldilocks/z_score"] = z_flat.cpu().numpy()

        # Score-error correlation (do high-scoring patches have higher error?)
        if p_tgt is not None:
            scores_at_tgt = p_tgt.gather(1, mask_output.target_idx)  # (B, K)
            sf = scores_at_tgt.float().reshape(-1)
            if sf.std() > 1e-12 and flat.std() > 1e-12:
                cov = ((sf - sf.mean()) * (flat - flat.mean())).mean()
                corr = cov / (sf.std() * flat.std())
                stats["goldilocks/score_error_corr"] = float(corr.item())

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