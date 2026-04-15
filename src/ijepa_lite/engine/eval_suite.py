"""
eval_suite.py — orchestrator for multi-mode evaluation in a single Hydra job.

Three modes are supported (enable via task.eval_modes.*):
  linear_probe : SGD-trained linear head on frozen mean-pool encoder features
  masker_probe : learnable (λ, α) through a frozen pretrained masker; soft-pool repr
  logreg       : sklearn LogisticRegression on mean-pool encoder features (no grad)

Entry point: eval_suite(cfg, encoder, masker, train_loader, val_loader,
                         num_classes, callbacks, device)
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader as _TDL, TensorDataset

from ijepa_lite.engine.eval_linear import LinearProbeModel, _acc_top1, _build_scheduler, _extract_features
from ijepa_lite.utils.dist import (
    all_reduce_sum,
    is_distributed,
    is_rank0,
    unwrap_model,
)
from ijepa_lite.utils.meters import AverageMeter


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def eval_suite(
    cfg,
    encoder: nn.Module,
    masker: Optional[nn.Module],
    train_loader,
    val_loader,
    num_classes: int,
    callbacks,
    device: torch.device,
) -> None:
    modes = getattr(cfg.task, "eval_modes", {})

    state: Dict = {"epoch": 0, "global_step": 0}
    callbacks.on_run_start(cfg=cfg, state=state, model=encoder)

    results: Dict = {}

    if getattr(modes, "linear_probe", True):
        results["linear_probe"] = _run_linear_probe(
            cfg, encoder, train_loader, val_loader, num_classes,
            callbacks, state, device,
        )

    if getattr(modes, "masker_probe", False):
        if masker is None:
            print("[eval_suite] masker_probe skipped: no masker loaded")
        else:
            results["masker_probe"] = _run_masker_probe(
                cfg, encoder, masker, train_loader, val_loader, num_classes,
                callbacks, state, device,
            )

    if getattr(modes, "logreg", False):
        results["logreg"] = _run_logreg(
            cfg, encoder, train_loader, val_loader,
            callbacks, state, device,
        )

    if getattr(modes, "masker_hard_probe", False):
        if masker is None:
            print("[eval_suite] masker_hard_probe skipped: no masker loaded")
        else:
            results["masker_hard_probe"] = _run_masker_hard_probe(
                cfg, encoder, masker, train_loader, val_loader, num_classes,
                callbacks, state, device,
            )

    if getattr(modes, "spatial_probe", False):
        results["spatial_probe"] = _run_spatial_probe(
            cfg, encoder, train_loader, val_loader, num_classes,
            callbacks, state, device,
        )

    if is_rank0():
        print("\n=== Eval Suite Results ===")
        for mode, res in results.items():
            if mode == "linear_probe":
                print(f"  linear_probe      : val_acc1={res['val_acc1']:.4f}")
            elif mode == "masker_probe":
                print(
                    f"  masker_probe      : val_acc1={res['val_acc1']:.4f}"
                    f"  \u03bb={res['lam']:.4f}  \u03b1={res['alpha']:.4f}"
                )
            elif mode == "logreg":
                print(f"  logreg            : val_acc1={res['val_acc1']:.4f}")
            elif mode == "masker_hard_probe":
                print(
                    f"  masker_hard_probe : val_acc1={res['val_acc1']:.4f}"
                    f"  \u03bb={res['lam']:.4f}  \u03b1={res['alpha']:.4f}"
                    f"  pool={res['pool']}"
                    f"  nctx={res['nctx']}  ntgt={res['ntgt']}  nign={res['nign']}"
                )
            elif mode == "spatial_probe":
                print(
                    f"  spatial_probe     : val_acc1={res['val_acc1']:.4f}"
                    f"  pool={res['pool']}  n_patches={res['n_patches']}"
                )

        callbacks.on_run_end(cfg=cfg, state=state)


# ---------------------------------------------------------------------------
# Mode: linear_probe
# ---------------------------------------------------------------------------

def _run_linear_probe(
    cfg,
    encoder: nn.Module,
    train_loader,
    val_loader,
    num_classes: int,
    callbacks,
    state: Dict,
    device: torch.device,
) -> Dict:
    """SGD linear head on frozen mean-pool encoder features.

    Functionally identical to linear_probe_eval() in eval_linear.py but:
    - No on_run_start / on_run_end calls (handled by the orchestrator)
    - Metric keys prefixed probe/linear_*
    - Returns {"val_acc1": float, "best_acc1": float}
    """
    amp = bool(getattr(cfg.task, "amp", True)) and (device.type == "cuda")
    epochs = int(cfg.train.epochs)
    log_every = int(getattr(cfg.train, "log_every", 50))

    head = nn.Linear(int(cfg.model.embed_dim), int(num_classes)).to(device)
    model = LinearProbeModel(
        encoder=encoder,
        head=head,
        pool=str(getattr(cfg.task, "pool", "mean")),
    ).to(device)

    # Freeze encoder before DDP wrapping: with feature caching the encoder is never
    # called inside the DDP-wrapped forward, so unfrozen params would cause a
    # DDP allreduce deadlock (find_unused_parameters=False).
    unwrap_model(model).encoder.requires_grad_(False)
    unwrap_model(model).encoder.eval()

    if is_distributed():
        from torch.nn.parallel import DistributedDataParallel as DDP
        kwargs: Dict = dict(
            broadcast_buffers=False,
            find_unused_parameters=bool(
                getattr(getattr(cfg, "distributed", None), "find_unused_parameters", False)
            ),
        )
        if device.type == "cuda":
            kwargs.update(device_ids=[device.index], output_device=device.index)
        model = DDP(model, **kwargs)

    opt = torch.optim.SGD(
        unwrap_model(model).head.parameters(),
        lr=float(cfg.train.lr),
        momentum=0.9,
        weight_decay=float(cfg.train.weight_decay),
    )
    sched = _build_scheduler(opt, getattr(cfg.task, "sched", None), epochs)
    scaler = GradScaler("cuda", enabled=amp)

    # Pre-extract features — encoder is frozen throughout the probe.
    def encode_fn(x):
        t = unwrap_model(model)._features(x)
        return F.layer_norm(t, (t.shape[-1],))

    feats_tr, labs_tr   = _extract_features(encode_fn, train_loader, device, amp)
    feats_val, labs_val = _extract_features(encode_fn, val_loader,   device, amp)

    bsz = train_loader.batch_size
    train_cache = _TDL(TensorDataset(feats_tr, labs_tr),   batch_size=bsz, shuffle=True,  drop_last=True)
    val_cache   = _TDL(TensorDataset(feats_val, labs_val), batch_size=bsz, shuffle=False)

    val_acc1 = 0.0
    best_acc1 = 0.0

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)

        unwrap_model(model).head.train()

        loss_meter = AverageMeter()
        correct_sum = torch.zeros((), device=device, dtype=torch.long)
        total_sum = torch.zeros((), device=device, dtype=torch.long)

        for feat, y in train_cache:
            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = model(None, feat=feat)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=feat.size(0))
            state["global_step"] += 1

            with torch.no_grad():
                c, t = _acc_top1(logits, y)
                correct_sum += c
                total_sum += t

            if state["global_step"] % log_every == 0 and is_rank0():
                callbacks.on_step_end(
                    cfg=cfg,
                    state=state,
                    metrics={
                        "probe/linear_train_loss": float(loss_meter.avg),
                        "probe/linear_lr": float(opt.param_groups[0]["lr"]),
                        "probe/epoch": float(epoch),
                    },
                )

        if sched is not None:
            sched.step()

        correct_sum = all_reduce_sum(correct_sum.float())
        total_sum = all_reduce_sum(total_sum.float())
        train_acc1 = (correct_sum / total_sum.clamp(min=1.0)).item()

        # Validate
        unwrap_model(model).eval()
        val_loss_sum = torch.zeros((), device=device)
        val_correct = torch.zeros((), device=device, dtype=torch.long)
        val_total = torch.zeros((), device=device, dtype=torch.long)

        with torch.no_grad():
            for feat, y in val_cache:
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = model(None, feat=feat)
                    loss = F.cross_entropy(logits, y)
                val_loss_sum += loss.detach() * feat.size(0)
                val_correct += _acc_top1(logits, y)[0]
                val_total += y.numel()

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_correct = all_reduce_sum(val_correct.float())
        val_total = all_reduce_sum(val_total.float())

        val_loss = (val_loss_sum / val_total.clamp(min=1.0)).item()
        val_acc1 = (val_correct / val_total.clamp(min=1.0)).item()

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "probe/linear_train_loss": float(loss_meter.avg),
                    "probe/linear_train_acc1": float(train_acc1),
                    "probe/linear_val_loss": float(val_loss),
                    "probe/linear_val_acc1": float(val_acc1),
                    "probe/linear_lr": float(opt.param_groups[0]["lr"]),
                    "probe/epoch": float(epoch),
                },
            )

    return {"val_acc1": val_acc1, "best_acc1": best_acc1}


# ---------------------------------------------------------------------------
# Mode: masker_probe
# ---------------------------------------------------------------------------

def _run_masker_probe(
    cfg,
    encoder: nn.Module,
    masker: nn.Module,
    train_loader,
    val_loader,
    num_classes: int,
    callbacks,
    state: Dict,
    device: torch.device,
) -> Dict:
    """Learnable (λ, α) probed through a frozen pretrained masker.

    Gradient path:
        log_lam, log_alpha
        → rates (B, 2)
        → masker.rates_proj (frozen weights, live input grad)
        → pos_embed_rates → transformer → softmax → p_ctx (B, N)
        → soft-weighted pool of detached encoder tokens
        → head → cross-entropy → backward
    """
    mp_cfg = getattr(cfg.task, "masker_probe", None)
    lr           = float(getattr(mp_cfg, "lr",           1e-3)) if mp_cfg else 1e-3
    rates_lr     = float(getattr(mp_cfg, "rates_lr",     1e-2)) if mp_cfg else 1e-2
    epochs       = int(getattr(mp_cfg,   "epochs",       20))   if mp_cfg else 20
    weight_decay = float(getattr(mp_cfg, "weight_decay", 0.0))  if mp_cfg else 0.0
    log_every    = int(getattr(cfg.train, "log_every", 50))

    # Initialise log_lam / log_alpha at log-midpoint of config ranges
    latent_cfg = getattr(cfg.masking, "latent", None)
    lam_min   = float(getattr(latent_cfg, "lam_min",   0.01)) if latent_cfg else 0.01
    lam_max   = float(getattr(latent_cfg, "lam_max",   1.0))  if latent_cfg else 1.0
    alpha_min = float(getattr(latent_cfg, "alpha_min", 0.01)) if latent_cfg else 0.01
    alpha_max = float(getattr(latent_cfg, "alpha_max", 0.5))  if latent_cfg else 0.5

    lam_init   = math.exp(0.5 * (math.log(lam_min)   + math.log(lam_max)))
    alpha_init = math.exp(0.5 * (math.log(alpha_min) + math.log(alpha_max)))
    log_lam   = nn.Parameter(torch.tensor(math.log(lam_init),   device=device))
    log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init), device=device))

    head = nn.Linear(int(cfg.model.embed_dim), int(num_classes)).to(device)

    optimizer = torch.optim.AdamW([
        {"params": head.parameters(),    "lr": lr,       "weight_decay": weight_decay},
        {"params": [log_lam, log_alpha], "lr": rates_lr, "weight_decay": 0.0},
    ])

    masker.eval()
    val_acc1 = 0.0

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)

        head.train()
        loss_meter = AverageMeter()

        for batch in train_loader:
            x = batch["images"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            B = x.size(0)

            with torch.no_grad():
                ema_tokens = encoder(x)  # (B, N, D) — detached

            lam_val   = log_lam.exp().clamp(min=1e-6)
            alpha_val = log_alpha.exp().clamp(min=1e-6)
            rates = torch.stack(
                [lam_val.expand(B), alpha_val.expand(B)], dim=-1
            )  # (B, 2)

            mask_out = masker(ema_tokens, ema_full=None, rates=rates)
            p_ctx = mask_out.context_soft  # (B, N) — grad flows through rates_proj

            w = p_ctx / (p_ctx.sum(1, keepdim=True) + 1e-6)
            repr_vec = (w.unsqueeze(-1) * ema_tokens).sum(1)  # (B, D)

            logits = head(repr_vec)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_meter.update(float(loss.item()), n=B)
            state["global_step"] += 1

            if state["global_step"] % log_every == 0 and is_rank0():
                callbacks.on_step_end(
                    cfg=cfg,
                    state=state,
                    metrics={
                        "probe/masker_train_loss": float(loss_meter.avg),
                        "probe/masker_lam":        float(log_lam.exp().item()),
                        "probe/masker_alpha":      float(log_alpha.exp().item()),
                    },
                )

        # Validate
        head.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["images"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                B = x.size(0)

                ema_tokens = encoder(x)
                lam_val   = log_lam.exp().clamp(min=1e-6)
                alpha_val = log_alpha.exp().clamp(min=1e-6)
                rates = torch.stack(
                    [lam_val.expand(B), alpha_val.expand(B)], dim=-1
                )
                mask_out = masker(ema_tokens, ema_full=None, rates=rates)
                p_ctx = mask_out.context_soft
                w = p_ctx / (p_ctx.sum(1, keepdim=True) + 1e-6)
                repr_vec = (w.unsqueeze(-1) * ema_tokens).sum(1)

                logits = head(repr_vec)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc1 = val_correct / max(val_total, 1)

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "probe/masker_val_acc1":   float(val_acc1),
                    "probe/masker_train_loss": float(loss_meter.avg),
                    "probe/masker_lam":        float(log_lam.exp().item()),
                    "probe/masker_alpha":      float(log_alpha.exp().item()),
                    "probe/epoch":             float(epoch),
                },
            )

    return {
        "val_acc1": val_acc1,
        "lam":      float(log_lam.exp().item()),
        "alpha":    float(log_alpha.exp().item()),
    }


# ---------------------------------------------------------------------------
# Mode: logreg
# ---------------------------------------------------------------------------

def _run_logreg(
    cfg,
    encoder: nn.Module,
    train_loader,
    val_loader,
    callbacks,
    state: Dict,
    device: torch.device,
) -> Dict:
    """sklearn LogisticRegression on mean-pool encoder features.

    Single pass through train + val loaders under torch.no_grad().
    In DDP mode, each rank processes its own shard; sklearn fits on
    the local shard (full data in single-rank runs).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    lr_cfg  = getattr(cfg.task, "logreg", None)
    max_iter = int(getattr(lr_cfg, "max_iter", 2000)) if lr_cfg else 2000
    C        = float(getattr(lr_cfg, "C",        1.0)) if lr_cfg else 1.0

    encoder.eval()

    def _extract(loader) -> tuple[np.ndarray, np.ndarray]:
        embs, labels = [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["images"].to(device, non_blocking=True)
                y = batch["labels"]
                tokens = encoder(x)          # (B, N, D)
                feat = tokens.mean(dim=1)    # (B, D) — mean-pool
                embs.append(feat.cpu())
                labels.append(y.cpu())
        X = torch.cat(embs, dim=0).numpy()
        y = torch.cat(labels, dim=0).numpy()
        return X, y

    X_train, y_train = _extract(train_loader)
    X_val,   y_val   = _extract(val_loader)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, C=C, solver="lbfgs"),
    )
    clf.fit(X_train, y_train)

    val_acc1 = float((clf.predict(X_val) == y_val).mean())

    if is_rank0():
        callbacks.on_epoch_end(
            cfg=cfg,
            state=state,
            metrics={"probe/logreg_val_acc": val_acc1},
        )

    return {"val_acc1": val_acc1}


# ---------------------------------------------------------------------------
# Mode: masker_hard_probe
# ---------------------------------------------------------------------------

def _parse_pool_bins(pool_str: str) -> frozenset:
    """Parse pool spec string into a frozenset of bin names.

    Examples:
        "ctx"         → frozenset({"ctx"})
        "tgt"         → frozenset({"tgt"})
        "ctx+tgt"     → frozenset({"ctx", "tgt"})
        "ctx+tgt+ign" → frozenset({"ctx", "tgt", "ign"})
    """
    valid = {"ctx", "tgt", "ign"}
    bins = frozenset(b.strip() for b in pool_str.split("+"))
    unknown = bins - valid
    if unknown:
        raise ValueError(
            f"Unknown pool bin(s): {unknown}. Valid: {valid}. "
            f"Combine with '+', e.g. 'ctx+tgt'."
        )
    return bins


def _run_masker_hard_probe(
    cfg,
    encoder: nn.Module,
    masker: nn.Module,
    train_loader,
    val_loader,
    num_classes: int,
    callbacks,
    state: Dict,
    device: torch.device,
) -> Dict:
    """Linear probe on a chosen subset of the masker's 3-way token partition.

    The masker runs with fixed (λ, α); its hard assignments split N patches
    into ctx / tgt / ign bins. `pool` selects which bins to mean-pool:

        pool="ctx"         — context tokens only  (current default)
        pool="tgt"         — target tokens only
        pool="ign"         — ignored tokens only  (sanity check)
        pool="ctx+tgt"     — non-ignored tokens
        pool="ctx+tgt+ign" — all tokens (≡ mean-pool baseline)

    Multiple bins are union-pooled: every token in any listed bin is included.
    Token counts (nctx, ntgt, nign) are always logged regardless of pool choice.
    """
    mhp_cfg   = getattr(cfg.task, "masker_hard_probe", None)
    lam       = float(getattr(mhp_cfg, "lam",   0.1))   if mhp_cfg else 0.1
    alpha     = float(getattr(mhp_cfg, "alpha", 0.07))  if mhp_cfg else 0.07
    pool_str  = str(getattr(mhp_cfg,   "pool",  "ctx"))  if mhp_cfg else "ctx"
    amp       = bool(getattr(cfg.task, "amp", True)) and (device.type == "cuda")
    epochs    = int(cfg.train.epochs)
    log_every = int(getattr(cfg.train, "log_every", 50))

    bins = _parse_pool_bins(pool_str)
    masker.eval()

    rates_1 = torch.tensor([[lam, alpha]], device=device)  # (1, 2) — fixed

    # Per-batch token counts (updated on every forward, logged each step/epoch)
    counts = {"nctx": 0, "ntgt": 0, "nign": 0}

    @torch.no_grad()
    def _hard_pool(x: torch.Tensor) -> torch.Tensor:
        B, N = x.size(0), (x.shape[-1] // 1)  # N resolved after encoder
        tokens   = encoder(x)                                   # (B, N, D)
        N        = tokens.shape[1]
        mask_out = masker(tokens, ema_full=None,
                          rates=rates_1.expand(B, -1))

        ctx_idx = mask_out.context_idx   # (B, nctx)
        tgt_idx = mask_out.target_idx    # (B, ntgt)

        # Build per-bin boolean masks  (B, N)
        ctx_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        tgt_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        ctx_mask.scatter_(1, ctx_idx, True)
        tgt_mask.scatter_(1, tgt_idx, True)
        ign_mask = ~(ctx_mask | tgt_mask)

        counts["nctx"] = int(ctx_mask[0].sum())
        counts["ntgt"] = int(tgt_mask[0].sum())
        counts["nign"] = int(ign_mask[0].sum())

        # Union of requested bins
        pool_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if "ctx" in bins:
            pool_mask |= ctx_mask
        if "tgt" in bins:
            pool_mask |= tgt_mask
        if "ign" in bins:
            pool_mask |= ign_mask

        n_kept = pool_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return (tokens * pool_mask.unsqueeze(-1)).sum(dim=1) / n_kept  # (B, D)

    head = nn.Linear(int(cfg.model.embed_dim), int(num_classes)).to(device)
    opt = torch.optim.SGD(
        head.parameters(),
        lr=float(cfg.train.lr),
        momentum=0.9,
        weight_decay=float(cfg.train.weight_decay),
    )
    sched = _build_scheduler(opt, getattr(cfg.task, "sched", None), epochs)
    scaler = GradScaler("cuda", enabled=amp)

    val_acc1  = 0.0
    best_acc1 = 0.0
    train_sampler = getattr(train_loader, "sampler", None)

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        head.train()
        loss_meter = AverageMeter()
        correct_sum = torch.zeros((), device=device, dtype=torch.long)
        total_sum   = torch.zeros((), device=device, dtype=torch.long)

        for batch in train_loader:
            x = batch["images"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            feat = _hard_pool(x)  # (B, D) — fully detached

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = head(feat)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=x.size(0))
            state["global_step"] += 1

            with torch.no_grad():
                c, t = _acc_top1(logits, y)
                correct_sum += c
                total_sum   += t

            if state["global_step"] % log_every == 0 and is_rank0():
                callbacks.on_step_end(
                    cfg=cfg,
                    state=state,
                    metrics={
                        "probe/hard_train_loss": float(loss_meter.avg),
                        "probe/hard_lr":         float(opt.param_groups[0]["lr"]),
                        "probe/hard_lam":        lam,
                        "probe/hard_alpha":      alpha,
                        "probe/hard_nctx":       float(counts["nctx"]),
                        "probe/hard_ntgt":       float(counts["ntgt"]),
                        "probe/hard_nign":       float(counts["nign"]),
                    },
                )

        if sched is not None:
            sched.step()

        correct_sum = all_reduce_sum(correct_sum.float())
        total_sum   = all_reduce_sum(total_sum.float())
        train_acc1  = (correct_sum / total_sum.clamp(min=1.0)).item()

        # Validate
        head.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_correct  = torch.zeros((), device=device, dtype=torch.long)
        val_total    = torch.zeros((), device=device, dtype=torch.long)

        with torch.no_grad():
            for batch in val_loader:
                x = batch["images"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                feat = _hard_pool(x)
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = head(feat)
                    loss = F.cross_entropy(logits, y)
                val_loss_sum += loss.detach() * x.size(0)
                val_correct  += _acc_top1(logits, y)[0]
                val_total    += y.numel()

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_correct  = all_reduce_sum(val_correct.float())
        val_total    = all_reduce_sum(val_total.float())

        val_loss = (val_loss_sum / val_total.clamp(min=1.0)).item()
        val_acc1 = (val_correct  / val_total.clamp(min=1.0)).item()

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "probe/hard_val_acc1":   float(val_acc1),
                    "probe/hard_train_acc1": float(train_acc1),
                    "probe/hard_val_loss":   float(val_loss),
                    "probe/hard_train_loss": float(loss_meter.avg),
                    "probe/hard_lr":         float(opt.param_groups[0]["lr"]),
                    "probe/hard_lam":        lam,
                    "probe/hard_alpha":      alpha,
                    "probe/hard_nctx":       float(counts["nctx"]),
                    "probe/hard_ntgt":       float(counts["ntgt"]),
                    "probe/hard_nign":       float(counts["nign"]),
                    "probe/epoch":           float(epoch),
                },
            )

    return {
        "val_acc1":  val_acc1,
        "best_acc1": best_acc1,
        "lam":       lam,
        "alpha":     alpha,
        "pool":      pool_str,
        "nctx":      counts["nctx"],
        "ntgt":      counts["ntgt"],
        "nign":      counts["nign"],
    }


# ---------------------------------------------------------------------------
# Mode: spatial_probe
# ---------------------------------------------------------------------------

def _run_spatial_probe(
    cfg,
    encoder: nn.Module,
    train_loader,
    val_loader,
    num_classes: int,
    callbacks,
    state: Dict,
    device: torch.device,
) -> Dict:
    """Fixed spatial patch selection — no masker, no learned parameters.

    pool=center : middle vertical third of patch-grid rows.
                  For img_size=96, patch_size=8 → 12×12 grid → rows 4–7 → 48 patches.
                  Falsification baseline: if masker_hard_probe(pool=ign) merely learns
                  a center-crop prior, center and ign should match.

    pool=random : same count of patches drawn uniformly at random per image.
                  Null hypothesis: any fixed-size subset performs equally well (position
                  does not matter, only the number of tokens).
    """
    sp_cfg    = getattr(cfg.task, "spatial_probe", None)
    pool_str  = str(getattr(sp_cfg, "pool", "center")) if sp_cfg else "center"
    amp       = bool(getattr(cfg.task, "amp", True)) and (device.type == "cuda")
    epochs    = int(cfg.train.epochs)
    log_every = int(getattr(cfg.train, "log_every", 50))

    img_size   = int(cfg.model.image_size)
    patch_size = int(cfg.model.patch_size)
    grid_h  = img_size // patch_size   # e.g. 12
    grid_w  = grid_h                   # square grid assumed
    row_start = grid_h // 3            # 4  for 12×12
    row_end   = 2 * grid_h // 3        # 8  for 12×12
    n_select  = (row_end - row_start) * grid_w  # 48

    if pool_str == "center":
        center_idx = torch.tensor(
            [r * grid_w + c for r in range(row_start, row_end) for c in range(grid_w)],
            device=device, dtype=torch.long,
        )  # (n_select,)

        @torch.no_grad()
        def _pool(x: torch.Tensor) -> torch.Tensor:
            tokens = encoder(x)                        # (B, N, D)
            return tokens[:, center_idx, :].mean(1)   # (B, D)

    elif pool_str == "random":
        @torch.no_grad()
        def _pool(x: torch.Tensor) -> torch.Tensor:
            tokens = encoder(x)                                        # (B, N, D)
            B, N, D = tokens.shape
            idx = torch.randint(N, (B, n_select), device=device)      # (B, n_select)
            sel = tokens.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
            return sel.mean(1)                                         # (B, D)

    else:
        raise ValueError(
            f"spatial_probe.pool must be 'center' or 'random', got {pool_str!r}"
        )

    head = nn.Linear(int(cfg.model.embed_dim), int(num_classes)).to(device)
    opt = torch.optim.SGD(
        head.parameters(),
        lr=float(cfg.train.lr),
        momentum=0.9,
        weight_decay=float(cfg.train.weight_decay),
    )
    sched = _build_scheduler(opt, getattr(cfg.task, "sched", None), epochs)
    scaler = GradScaler("cuda", enabled=amp)

    val_acc1  = 0.0
    best_acc1 = 0.0
    train_sampler = getattr(train_loader, "sampler", None)

    for epoch in range(epochs):
        state["epoch"] = epoch
        callbacks.on_epoch_start(cfg=cfg, state=state)

        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        head.train()
        loss_meter  = AverageMeter()
        correct_sum = torch.zeros((), device=device, dtype=torch.long)
        total_sum   = torch.zeros((), device=device, dtype=torch.long)

        for batch in train_loader:
            x = batch["images"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)

            feat = _pool(x)  # (B, D) — fully detached

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits = head(feat)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(float(loss.item()), n=x.size(0))
            state["global_step"] += 1

            with torch.no_grad():
                c, t = _acc_top1(logits, y)
                correct_sum += c
                total_sum   += t

            if state["global_step"] % log_every == 0 and is_rank0():
                callbacks.on_step_end(
                    cfg=cfg,
                    state=state,
                    metrics={
                        "probe/spatial_train_loss": float(loss_meter.avg),
                        "probe/spatial_lr":         float(opt.param_groups[0]["lr"]),
                    },
                )

        if sched is not None:
            sched.step()

        correct_sum = all_reduce_sum(correct_sum.float())
        total_sum   = all_reduce_sum(total_sum.float())
        train_acc1  = (correct_sum / total_sum.clamp(min=1.0)).item()

        # Validate
        head.eval()
        val_loss_sum = torch.zeros((), device=device)
        val_correct  = torch.zeros((), device=device, dtype=torch.long)
        val_total    = torch.zeros((), device=device, dtype=torch.long)

        with torch.no_grad():
            for batch in val_loader:
                x = batch["images"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
                feat = _pool(x)
                with autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    logits = head(feat)
                    loss = F.cross_entropy(logits, y)
                val_loss_sum += loss.detach() * x.size(0)
                val_correct  += _acc_top1(logits, y)[0]
                val_total    += y.numel()

        val_loss_sum = all_reduce_sum(val_loss_sum)
        val_correct  = all_reduce_sum(val_correct.float())
        val_total    = all_reduce_sum(val_total.float())

        val_loss = (val_loss_sum / val_total.clamp(min=1.0)).item()
        val_acc1 = (val_correct  / val_total.clamp(min=1.0)).item()

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1

        if is_rank0():
            callbacks.on_epoch_end(
                cfg=cfg,
                state=state,
                metrics={
                    "probe/spatial_val_acc1":   float(val_acc1),
                    "probe/spatial_train_acc1": float(train_acc1),
                    "probe/spatial_val_loss":   float(val_loss),
                    "probe/spatial_train_loss": float(loss_meter.avg),
                    "probe/spatial_lr":         float(opt.param_groups[0]["lr"]),
                    "probe/epoch":              float(epoch),
                },
            )

    return {
        "val_acc1":   val_acc1,
        "best_acc1":  best_acc1,
        "pool":       pool_str,
        "n_patches":  n_select,
    }
