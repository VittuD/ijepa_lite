from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


@torch.no_grad()
def token_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Predictor output vs target encoder output.

    pred/target shapes:
      - single block: (B, K, D)
      - multiblock:   (B, M, K, D)
    Flattens all but the last dim.

    NOTE: this measures how well the predictor matches the target encoder.
    It does NOT measure whether encoder representations are semantically
    meaningful. Use encoder_agreement() for that.
    """
    p = pred.detach().float().reshape(-1, pred.shape[-1])
    t = target.detach().float().reshape(-1, target.shape[-1])

    cos = F.cosine_similarity(
        F.normalize(p, dim=-1), F.normalize(t, dim=-1), dim=-1
    ).mean()
    mse_raw = F.mse_loss(p, t)

    p_norm = p.norm(dim=-1).mean()
    t_norm = t.norm(dim=-1).mean()
    p_var_d = p.var(dim=0, unbiased=False)  # (D,)
    t_var_d = t.var(dim=0, unbiased=False)
    p_std = p_var_d.sqrt().mean()
    t_std = t_var_d.sqrt().mean()
    p_var = p_var_d.mean()
    t_var = t_var_d.mean()

    return {
        "train/pred_tgt_cos_sim": float(cos.item()),
        "train/mse_raw": float(mse_raw.item()),
        "train/pred_norm": float(p_norm.item()),
        "train/tgt_norm": float(t_norm.item()),
        "train/pred_std": float(p_std.item()),
        "train/tgt_std": float(t_std.item()),
        "train/pred_var": float(p_var.item()),
        "train/tgt_var": float(t_var.item()),
    }


@torch.no_grad()
def encoder_agreement(
    ctx_tokens_all: torch.Tensor,  # (B, Nctx, D) context encoder output at ctx positions
    tgt_tokens_all: torch.Tensor,  # (B, Nctx, D) target encoder output at same positions
) -> Dict[str, float]:
    """
    Cosine similarity between context and target encoder at the SAME patch
    positions, averaged over all patches and images.

    This is the primary diagnostic for whether the encoder is learning:
      - Should start low and rise slowly over training as the context
        encoder improves and the EMA pulls the target encoder along
      - If it stays low while pred_tgt_cos_sim is high: the predictor is
        doing all the work; encoder is not learning useful representations
      - If it rockets to >0.95 early: task is too easy or representations
        are collapsing

    Also tracks spatial_std: std across the patch dimension (dim=1) per
    image, averaged over batch and feature dims. Measures whether different
    patch positions produce different representations:
      - Near zero: all patches look the same = collapsed spatial structure
      - Healthy: non-trivial and stable or slowly growing over training

    correction=0 (population std, not sample std) is used for the spatial
    diversity metrics.  This is correct here — we are describing a property
    of the current batch, not estimating a population parameter.  It also
    avoids the UserWarning when Nctx=1 (e.g. extreme-lambda RD masker steps
    where the masker collapses context to a single token), where
    Bessel-corrected std is undefined.
    """
    c = F.normalize(ctx_tokens_all.detach().float(), dim=-1)  # (B, Nctx, D)
    t = F.normalize(tgt_tokens_all.detach().float(), dim=-1)  # (B, Nctx, D)

    # Per-patch cosine similarity averaged over B and Nctx
    cos = (c * t).sum(dim=-1).mean()

    # Spatial diversity: std over patch positions — correction=0 avoids
    # undefined behaviour when Nctx=1 (no change in behaviour when Nctx>1)
    ctx_spatial_std = ctx_tokens_all.detach().float().std(dim=1, correction=0).mean()
    tgt_spatial_std = tgt_tokens_all.detach().float().std(dim=1, correction=0).mean()

    return {
        "train/encoder_agreement": float(cos.item()),
        "train/ctx_spatial_std": float(ctx_spatial_std.item()),
        "train/tgt_spatial_std": float(tgt_spatial_std.item()),
    }


def grad_norm(parameters, norm_type: float = 2.0) -> float:
    """Global grad norm over parameters that have grads."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    device = grads[0].device
    total = torch.zeros((), device=device)
    for g in grads:
        total += g.detach().pow(norm_type).sum()
    return float(total.pow(1.0 / norm_type).item())


@torch.no_grad()
def ema_param_metrics(
    ema_model: torch.nn.Module,
    online_model: torch.nn.Module,
) -> Dict[str, float]:
    """
    Compare EMA (target) parameters to online (context) parameters.
    Intended to be called only at log steps.
    """
    ema_p    = [p.detach().float() for p in ema_model.parameters()    if p is not None]
    online_p = [p.detach().float() for p in online_model.parameters() if p is not None]
    if not ema_p:
        return {"ema/param_l2": 0.0, "ema/param_rel_l2": 0.0, "ema/param_mean_abs": 0.0}

    # Flatten all parameter tensors into 1D for bulk ops — ~6 kernels vs ~300 in a loop
    ema_flat    = torch.cat([p.reshape(-1) for p in ema_p])
    online_flat = torch.cat([p.reshape(-1) for p in online_p])
    d = ema_flat - online_flat

    diff_l2   = d.norm()
    online_l2 = online_flat.norm()
    rel       = diff_l2 / (online_l2 + 1e-12)
    mean_abs  = d.abs().mean()

    return {
        "ema/param_l2":       float(diff_l2.item()),
        "ema/param_rel_l2":   float(rel.item()),
        "ema/param_mean_abs": float(mean_abs.item()),
    }