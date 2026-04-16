from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from ijepa_lite.utils.dist import all_reduce_sum


_TOKEN_METRIC_KEYS = (
    "train/pred_tgt_cos_sim",
    "train/mse_raw",
    "train/pred_norm",
    "train/tgt_norm",
    "train/pred_std",
    "train/tgt_std",
    "train/pred_var",
    "train/tgt_var",
)


def zero_token_metrics() -> Dict[str, float]:
    return {k: 0.0 for k in _TOKEN_METRIC_KEYS}


def _flatten_valid_tokens(
    x: torch.Tensor,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    x_flat = x.detach().float().reshape(-1, x.shape[-1])
    if valid is None:
        return x_flat

    valid = valid.to(dtype=torch.bool)
    if tuple(valid.shape) != tuple(x.shape[:-1]):
        raise ValueError(
            f"valid shape {tuple(valid.shape)} does not match token tensor shape "
            f"{tuple(x.shape[:-1])}."
        )
    valid_flat = valid.reshape(-1)
    if not bool(valid_flat.any().item()):
        return x_flat.new_zeros((0, x.shape[-1]))
    return x_flat[valid_flat]


def _masked_spatial_std(
    x: torch.Tensor,
    valid: torch.Tensor | None,
) -> torch.Tensor:
    x = x.detach().float()
    if valid is None:
        return x.std(dim=1, correction=0).mean()

    valid = valid.to(device=x.device, dtype=torch.bool)
    if tuple(valid.shape) != tuple(x.shape[:-1]):
        raise ValueError(
            f"valid shape {tuple(valid.shape)} does not match token tensor shape "
            f"{tuple(x.shape[:-1])}."
        )

    mask = valid.unsqueeze(-1).to(dtype=x.dtype)
    counts = mask.sum(dim=1)  # (B, 1)
    nonzero = counts.squeeze(-1) > 0
    if not bool(nonzero.any().item()):
        return x.new_zeros(())

    mean = (x * mask).sum(dim=1) / counts.clamp(min=1.0)
    centered = (x - mean.unsqueeze(1)) * mask
    var = centered.pow(2).sum(dim=1) / counts.clamp(min=1.0)
    std = var.sqrt().mean(dim=-1)
    return std[nonzero].mean()


@torch.no_grad()
def token_metric_sums(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Streaming-friendly sufficient statistics for token_metrics().

    This avoids materializing a large padded (B, M, K, D) diagnostic tensor in
    separate winners mode.  Callers can accumulate the returned tensors across
    blocks, then pass the result to finalize_token_metric_sums().
    """
    if tuple(valid.shape) != tuple(pred.shape[:-1]):
        raise ValueError(
            f"valid shape {tuple(valid.shape)} does not match token tensor shape "
            f"{tuple(pred.shape[:-1])}."
        )
    if pred.shape != target.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} does not match "
            f"target shape {tuple(target.shape)}."
        )

    valid_f = valid.to(device=pred.device, dtype=torch.float32)
    mask = valid_f.to(dtype=torch.bool).unsqueeze(-1)
    mask_f = valid_f.unsqueeze(-1)
    pred_f = torch.where(
        mask,
        pred.detach().float(),
        torch.zeros_like(pred, dtype=torch.float32),
    )
    target_f = torch.where(
        mask,
        target.detach().float(),
        torch.zeros_like(target, dtype=torch.float32),
    )
    count = mask_f.sum()

    cos = F.cosine_similarity(
        F.normalize(pred_f, dim=-1), F.normalize(target_f, dim=-1), dim=-1
    )
    diff = pred_f - target_f
    return {
        "count": count,
        "cos_sum": (cos * valid_f).sum(),
        "mse_sum": diff.pow(2).sum(),
        "pred_norm_sum": pred_f.norm(dim=-1).sum(),
        "tgt_norm_sum": target_f.norm(dim=-1).sum(),
        "pred_sum": pred_f.sum(dim=tuple(range(pred_f.dim() - 1))),
        "pred_sumsq": pred_f.pow(2).sum(dim=tuple(range(pred_f.dim() - 1))),
        "tgt_sum": target_f.sum(dim=tuple(range(target_f.dim() - 1))),
        "tgt_sumsq": target_f.pow(2).sum(dim=tuple(range(target_f.dim() - 1))),
    }


def merge_token_metric_sums(
    acc: dict[str, torch.Tensor] | None,
    update: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if acc is None:
        return {k: v.clone() for k, v in update.items()}
    for k, v in update.items():
        acc[k] = acc[k] + v
    return acc


@torch.no_grad()
def finalize_token_metric_sums(
    sums: dict[str, torch.Tensor] | None,
    *,
    distributed: bool = True,
) -> Dict[str, float]:
    if not sums:
        return zero_token_metrics()

    scalar_keys = (
        "count",
        "cos_sum",
        "mse_sum",
        "pred_norm_sum",
        "tgt_norm_sum",
    )
    vector_keys = ("pred_sum", "pred_sumsq", "tgt_sum", "tgt_sumsq")
    packed = torch.cat(
        [sums[k].reshape(-1) for k in scalar_keys + vector_keys],
        dim=0,
    )
    if distributed:
        packed = all_reduce_sum(packed)

    reduced = {}
    offset = 0
    for k in scalar_keys:
        reduced[k] = packed[offset]
        offset += 1
    d = sums["pred_sum"].numel()
    for k in vector_keys:
        reduced[k] = packed[offset: offset + d]
        offset += d

    count = reduced["count"].clamp(min=1.0)
    if float(reduced["count"].item()) <= 0.0:
        return zero_token_metrics()

    pred_mean = reduced["pred_sum"] / count
    tgt_mean = reduced["tgt_sum"] / count
    pred_var_d = (reduced["pred_sumsq"] / count - pred_mean.pow(2)).clamp(min=0.0)
    tgt_var_d = (reduced["tgt_sumsq"] / count - tgt_mean.pow(2)).clamp(min=0.0)

    return {
        "train/pred_tgt_cos_sim": float((reduced["cos_sum"] / count).item()),
        "train/mse_raw": float(
            (reduced["mse_sum"] / (count * pred_mean.numel())).item()
        ),
        "train/pred_norm": float((reduced["pred_norm_sum"] / count).item()),
        "train/tgt_norm": float((reduced["tgt_norm_sum"] / count).item()),
        "train/pred_std": float(pred_var_d.sqrt().mean().item()),
        "train/tgt_std": float(tgt_var_d.sqrt().mean().item()),
        "train/pred_var": float(pred_var_d.mean().item()),
        "train/tgt_var": float(tgt_var_d.mean().item()),
    }


@torch.no_grad()
def token_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> Dict[str, float]:
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
    p = _flatten_valid_tokens(pred, valid)
    t = _flatten_valid_tokens(target, valid)
    if p.numel() == 0 or t.numel() == 0:
        return zero_token_metrics()

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
    valid: torch.Tensor | None = None,
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

    cos_map = (c * t).sum(dim=-1)  # (B, Nctx)
    if valid is not None:
        valid = valid.to(device=cos_map.device, dtype=torch.bool)
        if tuple(valid.shape) != tuple(cos_map.shape):
            raise ValueError(
                f"valid shape {tuple(valid.shape)} does not match agreement shape "
                f"{tuple(cos_map.shape)}."
            )
        if bool(valid.any().item()):
            cos = cos_map[valid].mean()
        else:
            cos = cos_map.new_zeros(())
    else:
        cos = cos_map.mean()

    # Spatial diversity: std over patch positions — correction=0 avoids
    # undefined behaviour when Nctx=1 (no change in behaviour when Nctx>1)
    ctx_spatial_std = _masked_spatial_std(ctx_tokens_all, valid)
    tgt_spatial_std = _masked_spatial_std(tgt_tokens_all, valid)

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
