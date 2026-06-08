from __future__ import annotations

import math
import re
from collections.abc import Sequence
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ijepa_lite.callbacks.ckpt_cb import CheckpointCallback
from ijepa_lite.callbacks.handler import CallbackHandler
from ijepa_lite.callbacks.progress_cb import ProgressCallback
from ijepa_lite.data.classes import infer_num_classes
from ijepa_lite.data.collate import IJEPACollate, SegmentationCollate, SupervisedCollate
from ijepa_lite.data.datasets import build_dataset, build_segmentation_dataset
from ijepa_lite.data.transforms import (
    build_linear_probe_transforms,
    build_pretrain_transform,
    build_segmentation_transforms,
)
from ijepa_lite.engine.checkpoint import load_checkpoint_if_available, load_model_weights
from ijepa_lite.losses.rd_loss import RateDistSurpriseLoss
from ijepa_lite.losses.sigreg import SIGRegLoss
from ijepa_lite.losses.vanilla import VanillaTokenLoss
from ijepa_lite.masking.base import CollateMasker, LatentMasker
from ijepa_lite.masking.block_mask import BlockMaskGenerator
from ijepa_lite.masking.compressor import TokenCompressor
from ijepa_lite.masking.multiblock_mask import MultiBlockMaskGenerator
from ijepa_lite.models.ijepa import IJEPAModel
from ijepa_lite.models.predictor import Predictor
from ijepa_lite.models.vit_tokens import build_torchvision_vit_tokens
from ijepa_lite.utils.dist import get_rank, get_world_size, is_distributed
from ijepa_lite.utils.seed import seed_worker


# ------------------------------------------------------------------
# Masker builders
# ------------------------------------------------------------------

def _build_collate_masker(cfg) -> Optional[CollateMasker]:
    """
    Build a CPU-side CollateMasker from config.

    Returns None when a LatentMasker is configured (collate stays dumb).
    """
    # If a latent masker is configured, the collate should not produce masks.
    latent_cfg = getattr(cfg.masking, "latent", None)
    latent_name = str(getattr(latent_cfg, "name", "none")).lower() if latent_cfg else "none"
    if latent_name not in ("none", "null", ""):
        return None

    name = str(cfg.masking.name)

    if name == "block":
        area = cfg.masking.block_area_ratio  # [min, max]
        return BlockMaskGenerator(
            image_size=int(cfg.model.image_size),
            patch_size=int(cfg.model.patch_size),
            target_ratio=float(cfg.masking.target_ratio),
            context_ratio=float(cfg.masking.context_ratio),
            min_block_area_ratio=float(area[0]),
            max_block_area_ratio=float(area[1]),
        )

    if name == "multiblock":
        tgt_scale = cfg.masking.tgt_scale
        tgt_aspect = cfg.masking.tgt_aspect
        ctx_scale = cfg.masking.ctx_scale
        ctx_aspect = cfg.masking.ctx_aspect
        return MultiBlockMaskGenerator(
            image_size=int(cfg.model.image_size),
            patch_size=int(cfg.model.patch_size),
            target_ratio=float(cfg.masking.target_ratio),
            context_ratio=float(cfg.masking.context_ratio),
            num_target_blocks=int(cfg.masking.num_target_blocks),
            tgt_min_scale=float(tgt_scale[0]),
            tgt_max_scale=float(tgt_scale[1]),
            tgt_min_aspect=float(tgt_aspect[0]),
            tgt_max_aspect=float(tgt_aspect[1]),
            ctx_min_scale=float(ctx_scale[0]),
            ctx_max_scale=float(ctx_scale[1]),
            ctx_min_aspect=float(ctx_aspect[0]),
            ctx_max_aspect=float(ctx_aspect[1]),
            allow_overlap=bool(getattr(cfg.masking, "allow_overlap", False)),
            min_keep=int(getattr(cfg.masking, "min_keep", 10)),
            unclaimed=str(getattr(cfg.masking, "unclaimed", "ignore")),
        )

    raise ValueError(f"Unknown masking.name={name!r}")


def _build_compressor(cfg) -> Optional[TokenCompressor]:
    """
    Build a TokenCompressor from cfg.masking.compressor.

    Returns None if no compressor section is present in the config.
    A compressor is only required when cfg.masking.latent.name is set.
    """
    comp_cfg = getattr(cfg.masking, "compressor", None)
    if comp_cfg is None:
        return None

    mode = str(getattr(comp_cfg, "mode", "full"))
    embed_dim = int(cfg.model.embed_dim)

    return TokenCompressor(
        mode=mode,
        dim=embed_dim,
        stride=int(getattr(comp_cfg, "stride", 2)),
        num_queries=int(getattr(comp_cfg, "num_queries", 4)),
        num_heads=int(getattr(comp_cfg, "num_heads", 8)),
    )


def _build_latent_masker(
    cfg,
    compressor: Optional[TokenCompressor],
) -> Optional[LatentMasker]:
    """
    Build a LatentMasker from cfg.masking.latent using the masker registry.

    Returns None if masking.latent is absent or masking.latent.name is "none".

    The masker class must be registered via @register("name") and its module
    imported before this function is called.  The simplest way to ensure this
    is to import your masker module at the top of run.py:
        import ijepa_lite.masking.example_latent_masker  # noqa: F401

    All fields under cfg.masking.latent (except "name") are forwarded to the
    masker constructor as keyword arguments, along with these automatically
    inferred values:
        dim          : cfg.model.embed_dim
        num_patches  : (image_size // patch_size) ** 2
        target_ratio : cfg.masking.target_ratio
        context_ratio: cfg.masking.context_ratio

    Constructor keyword arguments take precedence over the auto-inferred ones
    if the masker config explicitly provides them.
    """
    latent_cfg = getattr(cfg.masking, "latent", None)
    if latent_cfg is None:
        return None

    name = str(getattr(latent_cfg, "name", "none")).lower()
    if name in ("none", "null", ""):
        return None

    # Lazy import to avoid importing user code unless a latent masker is used.
    from ijepa_lite.masking.registry import build_latent_masker

    num_patches = (int(cfg.model.image_size) // int(cfg.model.patch_size)) ** 2

    # Build kwargs: auto-inferred defaults, overridable by the latent config.
    #
    # Predictor-compatible fields are pulled from cfg.predictor so that
    # "predictor_based" maskers match predictor capacity with no config
    # duplication.  Any field can be overridden in masking.latent.
    pred_cfg = getattr(cfg, "predictor", None)
    auto_kwargs = {
        "dim": int(cfg.model.embed_dim),
        "num_patches": num_patches,
        "total_epochs": int(cfg.train.epochs),
        "image_size": int(cfg.model.image_size),
        "patch_size": int(cfg.model.patch_size),
        "target_ratio": float(getattr(cfg.masking, "target_ratio", 0.25)),
        "context_ratio": float(getattr(cfg.masking, "context_ratio", 0.75)),
        "num_target_blocks": int(getattr(cfg.masking, "num_target_blocks", 4)),
        "allow_overlap": bool(getattr(cfg.masking, "allow_overlap", False)),
        "min_keep": int(getattr(cfg.masking, "min_keep", 10)),
        "ctx_scale": list(getattr(cfg.masking, "ctx_scale", [0.85, 1.00])),
        "ctx_aspect": list(getattr(cfg.masking, "ctx_aspect", [0.75, 1.50])),
        "tgt_scale": list(getattr(cfg.masking, "tgt_scale", [0.15, 0.20])),
        "tgt_aspect": list(getattr(cfg.masking, "tgt_aspect", [0.75, 1.50])),
        # Predictor-compatible defaults (ignored by maskers that don't use them)
        "predictor_dim": int(getattr(pred_cfg, "predictor_dim", 192)),
        "depth": int(getattr(pred_cfg, "depth", 2)),
        "num_heads": int(getattr(pred_cfg, "num_heads", 6)),
        "mlp_ratio": float(getattr(pred_cfg, "mlp_ratio", 4.0)),
        "dropout": float(getattr(pred_cfg, "dropout", 0.0)),
        # Positional embedding kind — shared with encoder and predictor
        "pos_embed_kind": str(getattr(cfg.model, "pos_embed_kind", "learned")),
        # RD masker: distortion function (registry.py filters these for non-RD maskers)
        "base_kind": str(getattr(cfg.loss, "base_kind", getattr(cfg.loss, "kind", "smooth_l1"))),
        "normalize": bool(getattr(cfg.loss, "normalize", False)),
        # RD masker: λ distribution bounds (not used by MI masker)
        "lam_min": float(getattr(getattr(cfg.masking, "latent", None) or cfg.masking, "lam_min", 1e-3)),
        "lam_max": float(getattr(getattr(cfg.masking, "latent", None) or cfg.masking, "lam_max", 1.0)),
        # RD masker: α, β, λ_tgt LogUniform bounds (not used by MI masker)
        "alpha_min": float(getattr(latent_cfg, "alpha_min",
                           getattr(latent_cfg, "alpha", 0.01) / 10
                           if latent_cfg is not None else 0.01)),
        "alpha_max": float(getattr(latent_cfg, "alpha_max",
                           getattr(latent_cfg, "alpha", 0.5)
                           if latent_cfg is not None else 0.5)),
        "beta_min":     float(getattr(latent_cfg, "beta_min", 0.01)  if latent_cfg is not None else 0.01),
        "beta_max":     float(getattr(latent_cfg, "beta_max", 0.5)   if latent_cfg is not None else 0.5),
        "lam_tgt_min":  float(getattr(latent_cfg, "lam_tgt_min", 1e-3) if latent_cfg is not None else 1e-3),
        "lam_tgt_max":  float(getattr(latent_cfg, "lam_tgt_max", 0.1)  if latent_cfg is not None else 0.1),
    }

    # Collect all fields from the latent config (excluding "name")
    from omegaconf import OmegaConf
    latent_dict = (
        dict(OmegaConf.to_container(latent_cfg, resolve=True))
        if hasattr(latent_cfg, "_metadata")  # OmegaConf DictConfig
        else {k: v for k, v in vars(latent_cfg).items() if not k.startswith("_")}
    )
    latent_dict.pop("name", None)

    # Config overrides auto-inferred values
    merged_kwargs = {**auto_kwargs, **latent_dict}

    return build_latent_masker(name, **merged_kwargs)


# ------------------------------------------------------------------
# Callbacks, DDP, sampler (unchanged)
# ------------------------------------------------------------------

def build_callbacks(cfg):
    cbs = []
    cbs.append(ProgressCallback(log_every=int(getattr(cfg.train, "log_every", 50))))

    save_every = int(
        getattr(cfg.train, "save_every", getattr(cfg.train, "save_every_epochs", 1))
    )
    ckpt_dir = str(getattr(cfg.train, "ckpt_dir", "checkpoints"))
    ckpt_name = str(getattr(cfg.train, "ckpt_name", "last.pt"))

    cbs.append(
        CheckpointCallback(
            save_every=save_every,
            ckpt_dir=ckpt_dir,
            ckpt_name=ckpt_name,
        )
    )

    # Opt-in inline eval (before WandbCallback so metrics are visible to wandb)
    eval_every = int(getattr(cfg.train, "eval_every_epochs", 0))
    eval_every_steps = int(getattr(cfg.train, "eval_every_steps", 0))
    if eval_every > 0 or eval_every_steps > 0:
        from ijepa_lite.callbacks.eval_cb import InlineEvalCallback
        cbs.append(InlineEvalCallback())

    # Opt-in visualization (piggybacks on save_every cadence unless overridden)
    if bool(getattr(cfg.train, "viz_enabled", False)):
        from ijepa_lite.callbacks.viz_cb import VizCallback
        cbs.append(VizCallback())

    logger_cfg = getattr(cfg, "logger", None)
    logger_name = (
        str(getattr(logger_cfg, "name", "none")).lower() if logger_cfg else "none"
    )

    if logger_name == "wandb":
        try:
            from ijepa_lite.callbacks.wandb_cb import WandbCallback
        except Exception as e:
            raise RuntimeError(
                "cfg.logger.name='wandb' but WandbCallback could not be imported. "
                "Make sure the optional dependency 'wandb' is installed."
            ) from e
        cbs.append(WandbCallback(logger_cfg=logger_cfg))
    elif logger_name in ("none", "null", ""):
        pass
    else:
        raise ValueError(f"Unknown logger.name={logger_name!r}")

    return CallbackHandler(cbs)


def maybe_wrap_ddp(
    cfg, model: torch.nn.Module, device: torch.device
) -> torch.nn.Module:
    if not is_distributed():
        return model

    kwargs = dict(
        broadcast_buffers=False,
        find_unused_parameters=bool(
            getattr(getattr(cfg, "distributed", None), "find_unused_parameters", False)
        ),
    )
    if device.type == "cuda":
        kwargs.update(device_ids=[device.index], output_device=device.index)

    return DDP(model, **kwargs)


def _build_sampler(ds, shuffle: bool, drop_last: bool):
    if not is_distributed():
        return None

    from torch.utils.data.distributed import DistributedSampler

    return DistributedSampler(
        ds,
        num_replicas=get_world_size(),
        rank=get_rank(),
        shuffle=shuffle,
        drop_last=drop_last,
    )


# ------------------------------------------------------------------
# Loss builder
# ------------------------------------------------------------------

def _build_loss(cfg) -> VanillaTokenLoss:
    """
    Build the token-level loss function for use in ijepa.py.

    For standard maskers: VanillaTokenLoss with cfg.loss.kind.
    For rd_3way masker:   VanillaTokenLoss with cfg.loss.base_kind.
                          The RateDistSurpriseLoss combination lives on the masker;
                          this function only provides the per-patch distortion
                          that ijepa.py computes and passes into aux_loss.
    """
    normalize = bool(getattr(cfg.loss, "normalize", False))
    kind = str(getattr(cfg.loss, "kind", "mse"))

    if kind in ("rd_3way", "mi_3way", "mi_nway"):
        base_kind = str(getattr(cfg.loss, "base_kind", "smooth_l1"))
        return VanillaTokenLoss(normalize=normalize, kind=base_kind)

    return VanillaTokenLoss(normalize=normalize, kind=kind)


def _build_sigreg_loss(cfg) -> SIGRegLoss | None:
    sigreg_cfg = getattr(getattr(cfg, "loss", None), "sigreg", None)
    if sigreg_cfg is None:
        return None
    if not bool(getattr(sigreg_cfg, "enabled", False)):
        return None

    return SIGRegLoss(
        input_dim=int(cfg.model.embed_dim),
        num_slices=int(getattr(sigreg_cfg, "num_slices", 16)),
        num_t=int(getattr(sigreg_cfg, "num_t", 16)),
        t_max=float(getattr(sigreg_cfg, "t_max", 4.0)),
        standardize=bool(getattr(sigreg_cfg, "standardize", False)),
        eps=float(getattr(sigreg_cfg, "eps", 1e-6)),
        projection_seed=int(getattr(sigreg_cfg, "projection_seed", 0)),
        projector_hidden_dim=int(getattr(sigreg_cfg, "projector_hidden_dim", 2048)),
        projector_output_dim=int(getattr(sigreg_cfg, "projector_output_dim", 512)),
        use_projector=bool(getattr(sigreg_cfg, "use_projector", True)),
    )


# ------------------------------------------------------------------
# Pretrain model
# ------------------------------------------------------------------

def build_pretrain_model(cfg) -> torch.nn.Module:
    context = build_torchvision_vit_tokens(cfg.model)
    target_mode = str(getattr(cfg.model, "target_mode", "ema")).lower()
    if target_mode == "ema":
        target = build_torchvision_vit_tokens(cfg.model)
        target.load_state_dict(context.state_dict(), strict=True)
    elif target_mode == "shared":
        target = None
    else:
        raise ValueError(f"Unsupported model.target_mode={target_mode!r}")

    num_patches = (int(cfg.model.image_size) // int(cfg.model.patch_size)) ** 2

    pos_embed_kind = str(getattr(cfg.model, "pos_embed_kind", "learned"))

    pred = Predictor(
        dim=int(cfg.model.embed_dim),
        predictor_dim=int(cfg.predictor.predictor_dim),
        depth=int(cfg.predictor.depth),
        num_heads=int(cfg.predictor.num_heads),
        mlp_ratio=float(getattr(cfg.predictor, "mlp_ratio", 4.0)),
        dropout=float(getattr(cfg.predictor, "dropout", 0.0)),
        num_patches=num_patches,
        pos_embed_kind=pos_embed_kind,
    )

    loss_fn = _build_loss(cfg)

    ema_m = float(getattr(cfg.model, "ema_momentum", [0.0, 0.0])[0])

    compressor = _build_compressor(cfg)
    latent_masker = _build_latent_masker(cfg, compressor)
    if target_mode == "shared" and latent_masker is not None:
        raise ValueError(
            "target_mode='shared' is not yet supported with masking.latent.*. "
            "Use deterministic masking for the initial SIGReg integration."
        )

    sigreg_loss = _build_sigreg_loss(cfg)
    sigreg_weight = getattr(getattr(cfg.loss, "sigreg", None), "weight", 0.0)
    if isinstance(sigreg_weight, Sequence) and not isinstance(sigreg_weight, (str, bytes)):
        if len(sigreg_weight) != 2:
            raise ValueError(
                "loss.sigreg.weight must be a scalar or a [start, end] pair."
            )
        sigreg_weight_max = max(float(sigreg_weight[0]), float(sigreg_weight[1]))
    else:
        sigreg_weight_max = float(sigreg_weight)

    if target_mode != "shared" and sigreg_loss is not None and sigreg_weight_max > 0.0:
        raise ValueError(
            "loss.sigreg is currently supported only with model.target_mode='shared'."
        )
    if target_mode == "shared" and (sigreg_loss is None or sigreg_weight_max <= 0.0):
        raise ValueError(
            "target_mode='shared' requires loss.sigreg.enabled=true and loss.sigreg.weight > 0."
        )

    # Context loss config (V-JEPA 2.1-style visible token supervision).
    ctx_cfg = getattr(cfg, "ctx_loss", None)
    ctx_kw = {}
    if ctx_cfg is not None:
        ctx_kw = dict(
            ctx_loss_weight=float(getattr(ctx_cfg, "weight", 0.0)),
            ctx_loss_gamma=float(getattr(ctx_cfg, "gamma", 0.7)),
            ctx_loss_warmup_start=int(getattr(ctx_cfg, "warmup_start", 0)),
            ctx_loss_warmup_end=int(getattr(ctx_cfg, "warmup_end", 0)),
        )

    return IJEPAModel(
        context_encoder=context,
        target_encoder=target,
        predictor=pred,
        loss_fn=loss_fn,
        ema_momentum=ema_m,
        mask_generator=None,
        latent_masker=latent_masker,
        token_compressor=compressor,
        predict_blocks_jointly=bool(getattr(cfg.model, "predict_blocks_jointly", True)),
        grid_size=int(cfg.model.image_size) // int(cfg.model.patch_size),
        target_mode=target_mode,
        sigreg_loss=sigreg_loss,
        sigreg_weight=sigreg_weight,
        sigreg_weight_schedule=str(
            getattr(getattr(cfg.loss, "sigreg", None), "weight_schedule", "constant")
        ),
        total_epochs=int(cfg.train.epochs),
        **ctx_kw,
    )


# ------------------------------------------------------------------
# Pretrain loader
# ------------------------------------------------------------------

def build_pretrain_loader(cfg):
    tfm = build_pretrain_transform(cfg)

    # CollateMasker is None when a LatentMasker is configured.
    # In that case IJEPACollate is dumb (images only) and masking
    # happens inside IJEPAModel.forward on GPU.
    masker = _build_collate_masker(cfg)

    pretrain_split = str(getattr(cfg.data, "pretrain_split", "train"))
    ds = build_dataset(cfg.data, split=pretrain_split, transform=tfm)

    sampler = _build_sampler(ds, shuffle=True, drop_last=True)

    collate = IJEPACollate(masker=masker)

    from torch.utils.data import DataLoader

    loader = DataLoader(
        ds,
        batch_size=int(cfg.data.batch_size),
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=int(cfg.data.num_workers),
        pin_memory=bool(getattr(cfg.data, "pin_memory", True)),
        persistent_workers=(
            bool(getattr(cfg.data, "persistent_workers", True))
            if int(cfg.data.num_workers) > 0
            else False
        ),
        prefetch_factor=(
            int(getattr(cfg.data, "prefetch_factor", 2))
            if int(cfg.data.num_workers) > 0
            else None
        ),
        worker_init_fn=seed_worker,
        collate_fn=collate,
        drop_last=True,
    )
    return loader


# ------------------------------------------------------------------
# Optimiser + scheduler (unchanged)
# ------------------------------------------------------------------

def build_pretrain_optim_sched(cfg, model: torch.nn.Module):
    wd_start = float(cfg.optim.weight_decay)
    wd_end = float(getattr(cfg.optim, "final_weight_decay", wd_start))

    lr = float(cfg.optim.lr)
    betas = tuple(float(x) for x in cfg.optim.betas)
    eps = float(cfg.optim.eps)
    masker_cfg = getattr(cfg.optim, "masker", None)

    def _masker_opt_value(nested_key: str, legacy_key: str, default):
        if masker_cfg is not None and hasattr(masker_cfg, nested_key):
            return getattr(masker_cfg, nested_key)
        return getattr(cfg.optim, legacy_key, default)

    masker_lr_scale = float(_masker_opt_value("lr_scale", "masker_lr_scale", 1.0))
    separate_masker_opt = bool(
        _masker_opt_value("separate_optimizer", "masker_separate_optimizer", False)
    )
    masker_betas = tuple(float(x) for x in _masker_opt_value("betas", "masker_betas", betas))
    masker_eps = float(_masker_opt_value("eps", "masker_eps", eps))
    masker_wd_start = float(
        _masker_opt_value("weight_decay", "masker_weight_decay", wd_start)
    )
    masker_wd_end = float(
        _masker_opt_value(
            "final_weight_decay", "masker_final_weight_decay", masker_wd_start
        )
    )

    # Split params: masker submodules optionally get a separate optimizer
    core = model.module if hasattr(model, "module") else model
    masker_modules = set()
    if getattr(core, "latent_masker", None) is not None:
        masker_modules.update(core.latent_masker.parameters())
    if getattr(core, "token_compressor", None) is not None:
        masker_modules.update(core.token_compressor.parameters())

    base_params = []
    masker_params = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p in masker_modules:
            masker_params.append(p)
        else:
            base_params.append(p)

    if separate_masker_opt and masker_params:
        # Separate optimizer: base_opt gets encoder/predictor only
        param_groups = [{"params": base_params, "lr": lr}]
    else:
        # Shared optimizer: masker params go into a second param group (current behavior)
        param_groups = [{"params": base_params, "lr": lr}]
        if masker_params:
            param_groups.append({
                "params": masker_params,
                "lr": lr * masker_lr_scale,
            })

    opt = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=wd_start,
    )

    sched = None
    _lr_lambda = None
    if getattr(cfg, "sched", None) is not None:
        name = str(getattr(cfg.sched, "name", "warmup_cosine")).lower()
        if name == "warmup_cosine":
            interval = str(getattr(cfg.sched, "interval", "epoch")).lower()
            warmup_epochs = int(getattr(cfg.sched, "warmup_epochs", 0))
            warmup_steps = int(getattr(cfg.sched, "warmup_steps", 0))
            configured_total_steps = int(getattr(cfg.sched, "total_steps", 0))
            min_lr = float(getattr(cfg.sched, "min_lr", 0.0))
            if interval == "step":
                warmup = warmup_steps
                total = configured_total_steps
                if total <= 0:
                    total = int(getattr(cfg.train, "max_steps", 0))
                if total <= 0:
                    raise ValueError(
                        "Step-based warmup_cosine requires sched.total_steps "
                        "or train.max_steps > 0."
                    )
            elif interval == "epoch":
                if warmup_steps > 0 or configured_total_steps > 0:
                    raise ValueError(
                        "sched.warmup_steps and sched.total_steps require "
                        "sched.interval=step. Epoch scheduling uses "
                        "sched.warmup_epochs and train.epochs."
                    )
                warmup = warmup_epochs
                total = int(cfg.train.epochs)
            else:
                raise ValueError(
                    "sched.interval must be 'epoch' or 'step' "
                    f"(got {interval!r})."
                )

            def _lr_lambda(unit: int):
                if warmup > 0 and unit < warmup:
                    return float(unit + 1) / float(max(1, warmup))
                progress = (unit - warmup) / float(max(1, total - warmup))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (min_lr / lr) + (1.0 - (min_lr / lr)) * cosine

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        elif name == "none":
            sched = None
        else:
            raise ValueError(f"Unknown sched.name={name}")

    # Build separate masker optimizer when opted in
    masker_opt = None
    masker_sched = None
    if separate_masker_opt and masker_params:
        masker_opt = torch.optim.AdamW(
            masker_params,
            lr=lr * masker_lr_scale,
            betas=masker_betas,
            eps=masker_eps,
            weight_decay=masker_wd_start,
        )
        if _lr_lambda is not None:
            masker_sched = torch.optim.lr_scheduler.LambdaLR(masker_opt, lr_lambda=_lr_lambda)

    return opt, masker_opt, sched, masker_sched, wd_start, wd_end, masker_wd_start, masker_wd_end


# ------------------------------------------------------------------
# Eval masker (for eval_suite masker_probe mode)
# ------------------------------------------------------------------

def build_eval_masker(cfg, sd_full: dict, device: torch.device):
    """
    Build a frozen LatentMasker for the masker_probe eval mode.

    Returns None when:
    - masking.latent is absent or masking.latent.name is "none"
    - the checkpoint has no 'latent_masker.*' keys

    On success: loads weights (strict=False), freezes all params, moves to device.
    """
    latent_cfg = getattr(cfg.masking, "latent", None)
    latent_name = str(getattr(latent_cfg, "name", "none")).lower() if latent_cfg else "none"
    if latent_name in ("none", "null", ""):
        return None

    masker = _build_latent_masker(cfg, compressor=None)
    if masker is None:
        return None

    sd_masker = {
        k[len("latent_masker."):]: v
        for k, v in sd_full.items()
        if k.startswith("latent_masker.")
    }
    if not sd_masker:
        import warnings
        warnings.warn(
            "[build_eval_masker] No 'latent_masker.*' keys found in checkpoint;"
            " masker_probe mode will be skipped."
        )
        return None

    known_optional: set[str] = set()
    missing, unexpected = masker.load_state_dict(sd_masker, strict=False)
    real_missing = [k for k in missing if k not in known_optional]
    if real_missing:
        import warnings
        warnings.warn(
            f"[build_eval_masker] Missing unexpected keys: {real_missing}."
            " Check that the checkpoint architecture matches the config."
        )
    if missing:
        print(f"[build_eval_masker] missing keys (using defaults): {missing}")

    masker.requires_grad_(False)
    masker.to(device)
    return masker


# ------------------------------------------------------------------
# Linear probe (unchanged)
# ------------------------------------------------------------------


def _normalize_probe_checkpoint_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    nested_keys = ("target_encoder", "context_encoder", "encoder")
    for key in nested_keys:
        nested = sd.get(key)
        if isinstance(nested, dict):
            sd = nested
            break

    normalized: Dict[str, torch.Tensor] = {}
    for key, value in sd.items():
        if not isinstance(value, torch.Tensor):
            continue
        if key.startswith("module."):
            key = key[len("module."):]
        normalized[key] = value
    return normalized


def _extract_probe_encoder_state_dict(
    sd: Dict[str, torch.Tensor],
    prefer: str,
) -> Dict[str, torch.Tensor]:
    if prefer == "auto":
        prefixes = (
            "target_encoder.",
            "ema_encoder.",
            "context_encoder.",
            "encoder.",
            "backbone.",
        )
    elif prefer in ("target", "ema"):
        prefixes = (
            "target_encoder.",
            "ema_encoder.",
            "context_encoder.",
            "encoder.",
            "backbone.",
        )
    elif prefer in ("context", "shared"):
        prefixes = (
            "context_encoder.",
            "encoder.",
            "target_encoder.",
            "ema_encoder.",
            "backbone.",
        )
    else:
        raise ValueError(f"Unknown task.encoder={prefer!r}")

    for prefix in prefixes:
        enc_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if enc_sd:
            return enc_sd
    return sd


def _resize_torchvision_pos_embedding(
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
) -> torch.Tensor:
    if src_pos.ndim != 3 or dst_pos.ndim != 3:
        raise ValueError("Expected 3D positional embeddings for torchvision ViT.")
    if src_pos.shape[0] != 1 or dst_pos.shape[0] != 1:
        raise ValueError("Expected batch dimension 1 for positional embeddings.")
    if src_pos.shape[2] != dst_pos.shape[2]:
        raise ValueError(
            "Cannot resize positional embeddings with different channel dimensions: "
            f"{tuple(src_pos.shape)} vs {tuple(dst_pos.shape)}."
        )

    src_tokens = src_pos.shape[1] - 1
    dst_tokens = dst_pos.shape[1] - 1
    src_grid = int(math.isqrt(src_tokens))
    dst_grid = int(math.isqrt(dst_tokens))
    if src_grid * src_grid != src_tokens or dst_grid * dst_grid != dst_tokens:
        raise ValueError(
            "Positional embedding resize requires square patch grids after stripping CLS."
        )

    cls_pos = src_pos[:, :1, :]
    patch_pos = src_pos[:, 1:, :]
    patch_pos = patch_pos.reshape(1, src_grid, src_grid, src_pos.shape[2]).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos,
        size=(dst_grid, dst_grid),
        mode="bicubic",
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, dst_tokens, src_pos.shape[2])
    return torch.cat([cls_pos, patch_pos], dim=1)


def _resize_patch_only_pos_embedding(
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
) -> torch.Tensor:
    if src_pos.ndim != 3 or dst_pos.ndim != 3:
        raise ValueError("Expected 3D patch-only positional embeddings.")
    if src_pos.shape[0] != 1 or dst_pos.shape[0] != 1:
        raise ValueError("Expected batch dimension 1 for positional embeddings.")
    if src_pos.shape[2] != dst_pos.shape[2]:
        raise ValueError(
            "Cannot resize positional embeddings with different channel dimensions: "
            f"{tuple(src_pos.shape)} vs {tuple(dst_pos.shape)}."
        )

    src_tokens = src_pos.shape[1]
    dst_tokens = dst_pos.shape[1]
    src_grid = int(math.isqrt(src_tokens))
    dst_grid = int(math.isqrt(dst_tokens))
    if src_grid * src_grid != src_tokens or dst_grid * dst_grid != dst_tokens:
        raise ValueError("Patch-only positional embedding resize requires square patch grids.")

    patch_pos = src_pos.reshape(1, src_grid, src_grid, src_pos.shape[2]).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(
        patch_pos,
        size=(dst_grid, dst_grid),
        mode="bicubic",
        align_corners=False,
    )
    return patch_pos.permute(0, 2, 3, 1).reshape(1, dst_tokens, src_pos.shape[2])


def _is_square_token_count(tokens: int) -> bool:
    grid = int(math.isqrt(int(tokens)))
    return grid * grid == int(tokens)


def _resize_probe_pos_embedding(
    src_pos: torch.Tensor,
    dst_pos: torch.Tensor,
) -> torch.Tensor:
    src_tokens = int(src_pos.shape[1])
    dst_tokens = int(dst_pos.shape[1])
    if _is_square_token_count(src_tokens) and _is_square_token_count(dst_tokens):
        return _resize_patch_only_pos_embedding(src_pos, dst_pos)
    if _is_square_token_count(src_tokens - 1) and _is_square_token_count(dst_tokens - 1):
        return _resize_torchvision_pos_embedding(src_pos, dst_pos)
    raise ValueError(
        "Cannot resize positional embeddings because neither patch-only nor "
        "CLS-token square-grid layouts match: "
        f"ckpt{tuple(src_pos.shape)} -> model{tuple(dst_pos.shape)}."
    )


def _looks_like_original_ijepa_encoder_state_dict(enc_sd: Dict[str, torch.Tensor]) -> bool:
    return (
        "patch_embed.proj.weight" in enc_sd
        and "pos_embed" in enc_sd
        and any(k.startswith("blocks.") for k in enc_sd)
    )


def _set_vit_tokens_patch_only_mode(
    encoder: torch.nn.Module,
    *,
    pos_tokens: int,
) -> None:
    if not hasattr(encoder, "vit") or not hasattr(encoder.vit, "encoder"):
        raise ValueError("Expected a ViTTokens encoder backed by torchvision VisionTransformer.")
    embed_dim = int(encoder.vit.encoder.pos_embedding.shape[-1])
    device = encoder.vit.encoder.pos_embedding.device
    dtype = encoder.vit.encoder.pos_embedding.dtype
    encoder.use_cls_token = False
    encoder.vit.encoder.pos_embedding = nn.Parameter(
        torch.zeros(1, pos_tokens, embed_dim, device=device, dtype=dtype)
    )


def _map_original_ijepa_key(key: str) -> str | None:
    if key == "patch_embed.proj.weight":
        return "vit.conv_proj.weight"
    if key == "patch_embed.proj.bias":
        return "vit.conv_proj.bias"
    if key == "pos_embed":
        return "vit.encoder.pos_embedding"
    if key == "norm.weight":
        return "vit.encoder.ln.weight"
    if key == "norm.bias":
        return "vit.encoder.ln.bias"

    match = re.match(r"blocks\.(\d+)\.(.+)", key)
    if match is None:
        return None

    idx = int(match.group(1))
    suffix = match.group(2)
    prefix = f"vit.encoder.layers.encoder_layer_{idx}"

    if suffix == "norm1.weight":
        return f"{prefix}.ln_1.weight"
    if suffix == "norm1.bias":
        return f"{prefix}.ln_1.bias"
    if suffix == "norm2.weight":
        return f"{prefix}.ln_2.weight"
    if suffix == "norm2.bias":
        return f"{prefix}.ln_2.bias"
    if suffix == "attn.qkv.weight":
        return f"{prefix}.self_attention.in_proj_weight"
    if suffix == "attn.qkv.bias":
        return f"{prefix}.self_attention.in_proj_bias"
    if suffix == "attn.proj.weight":
        return f"{prefix}.self_attention.out_proj.weight"
    if suffix == "attn.proj.bias":
        return f"{prefix}.self_attention.out_proj.bias"
    if suffix == "mlp.fc1.weight":
        return f"{prefix}.mlp.0.weight"
    if suffix == "mlp.fc1.bias":
        return f"{prefix}.mlp.0.bias"
    if suffix == "mlp.fc2.weight":
        return f"{prefix}.mlp.3.weight"
    if suffix == "mlp.fc2.bias":
        return f"{prefix}.mlp.3.bias"
    return None


def _adapt_original_ijepa_encoder_state_dict(
    enc_sd: Dict[str, torch.Tensor],
    encoder: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    pos_embed = enc_sd.get("pos_embed")
    if pos_embed is None or pos_embed.ndim != 3:
        raise ValueError("Original I-JEPA checkpoint is missing a valid patch-only pos_embed.")

    _set_vit_tokens_patch_only_mode(encoder, pos_tokens=int(pos_embed.shape[1]))
    target_sd = encoder.state_dict()
    adapted: Dict[str, torch.Tensor] = {}
    skipped: list[str] = []

    for key, value in enc_sd.items():
        mapped = _map_original_ijepa_key(key)
        if mapped is None:
            skipped.append(key)
            continue
        if mapped not in target_sd:
            raise ValueError(f"Mapped original-IJEPA key {key!r} -> {mapped!r} not found in target encoder.")
        dst = target_sd[mapped]
        if value.shape != dst.shape:
            if mapped == "vit.encoder.pos_embedding":
                value = _resize_patch_only_pos_embedding(value, dst)
            else:
                raise ValueError(
                    "Original I-JEPA tensor shape mismatch after mapping: "
                    f"{key} {tuple(value.shape)} -> {mapped} {tuple(dst.shape)}"
                )
        adapted[mapped] = value

    essential = (
        "vit.conv_proj.weight",
        "vit.encoder.pos_embedding",
        "vit.encoder.ln.weight",
        "vit.encoder.ln.bias",
    )
    missing_essential = [key for key in essential if key not in adapted]
    if missing_essential:
        raise ValueError(
            "Original I-JEPA adapter did not populate required encoder tensors: "
            f"{missing_essential}"
        )
    return adapted


def _map_original_ijepa_predictor_key(key: str) -> str | None:
    if key == "predictor_embed.weight":
        return "proj_in.weight"
    if key == "predictor_embed.bias":
        return "proj_in.bias"
    if key == "predictor_proj.weight":
        return "proj_out.weight"
    if key == "predictor_proj.bias":
        return "proj_out.bias"
    if key == "predictor_pos_embed":
        return "pos_embed"
    if key == "mask_token":
        return "mask_token"
    if key == "predictor_norm.weight":
        return "norm.weight"
    if key == "predictor_norm.bias":
        return "norm.bias"

    match = re.match(r"predictor_blocks\.(\d+)\.(.+)", key)
    if match is None:
        return None

    idx = int(match.group(1))
    suffix = match.group(2)
    prefix = f"blocks.layers.{idx}"

    if suffix == "norm1.weight":
        return f"{prefix}.norm1.weight"
    if suffix == "norm1.bias":
        return f"{prefix}.norm1.bias"
    if suffix == "norm2.weight":
        return f"{prefix}.norm2.weight"
    if suffix == "norm2.bias":
        return f"{prefix}.norm2.bias"
    if suffix == "attn.qkv.weight":
        return f"{prefix}.self_attn.in_proj_weight"
    if suffix == "attn.qkv.bias":
        return f"{prefix}.self_attn.in_proj_bias"
    if suffix == "attn.proj.weight":
        return f"{prefix}.self_attn.out_proj.weight"
    if suffix == "attn.proj.bias":
        return f"{prefix}.self_attn.out_proj.bias"
    if suffix == "mlp.fc1.weight":
        return f"{prefix}.linear1.weight"
    if suffix == "mlp.fc1.bias":
        return f"{prefix}.linear1.bias"
    if suffix == "mlp.fc2.weight":
        return f"{prefix}.linear2.weight"
    if suffix == "mlp.fc2.bias":
        return f"{prefix}.linear2.bias"
    return None


def _adapt_original_ijepa_predictor_state_dict(
    pred_sd: Dict[str, torch.Tensor],
    predictor: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    target_sd = predictor.state_dict()
    adapted: Dict[str, torch.Tensor] = {}

    for key, value in pred_sd.items():
        mapped = _map_original_ijepa_predictor_key(key)
        if mapped is None:
            continue
        if mapped not in target_sd:
            raise ValueError(
                f"Mapped original-IJEPA predictor key {key!r} -> {mapped!r} "
                "not found in target predictor."
            )
        dst = target_sd[mapped]
        if value.shape != dst.shape:
            if mapped == "pos_embed":
                value = _resize_patch_only_pos_embedding(value, dst)
            else:
                raise ValueError(
                    "Original I-JEPA predictor tensor shape mismatch after mapping: "
                    f"{key} {tuple(value.shape)} -> {mapped} {tuple(dst.shape)}"
                )
        adapted[mapped] = value

    essential = (
        "proj_in.weight",
        "proj_out.weight",
        "pos_embed",
        "mask_token",
        "norm.weight",
        "norm.bias",
    )
    missing_essential = [key for key in essential if key not in adapted]
    if missing_essential:
        raise ValueError(
            "Original I-JEPA adapter did not populate required predictor tensors: "
            f"{missing_essential}"
        )
    return adapted


def _adapt_probe_encoder_state_dict(
    enc_sd: Dict[str, torch.Tensor],
    encoder: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    if _looks_like_original_ijepa_encoder_state_dict(enc_sd):
        enc_sd = _adapt_original_ijepa_encoder_state_dict(enc_sd, encoder)

    target_sd = encoder.state_dict()
    overlap = [key for key in enc_sd if key in target_sd]
    if not overlap:
        raise ValueError(
            "Checkpoint encoder weights do not match the torchvision downstream encoder format. "
            "No compatible encoder subtree was found for the current downstream probe model."
        )

    adapted = dict(enc_sd)
    incompatible: list[str] = []
    for key in overlap:
        src = adapted[key]
        dst = target_sd[key]
        if src.shape == dst.shape:
            continue
        if key == "vit.encoder.pos_embedding":
            adapted[key] = _resize_probe_pos_embedding(src, dst)
            continue
        if key == "vit.conv_proj.weight":
            raise ValueError(
                "Patch embedding shape mismatch between checkpoint and downstream model: "
                f"{tuple(src.shape)} vs {tuple(dst.shape)}. "
                "For downstream probes, model.patch_size and model.embed_dim must match "
                "the pretrained encoder."
            )
        incompatible.append(f"{key}: ckpt{tuple(src.shape)} != model{tuple(dst.shape)}")

    if incompatible:
        msg = "; ".join(incompatible[:4])
        if len(incompatible) > 4:
            msg += "; ..."
        raise ValueError(f"Incompatible downstream encoder checkpoint shapes: {msg}")
    return adapted

def build_linear_probe_model(cfg) -> torch.nn.Module:
    encoder = build_torchvision_vit_tokens(cfg.model)

    ckpt = getattr(cfg.task, "pretrained_ckpt", None)
    if ckpt:
        payload = torch.load(str(ckpt), map_location="cpu", weights_only=True)
        sd = (
            payload["model"]
            if isinstance(payload, dict) and "model" in payload
            else payload
        )
        sd = _normalize_probe_checkpoint_state_dict(sd)

        prefer = str(getattr(cfg.task, "encoder", "auto")).lower()
        enc_sd = _extract_probe_encoder_state_dict(sd, prefer)
        enc_sd = _adapt_probe_encoder_state_dict(enc_sd, encoder)
        encoder.load_state_dict(enc_sd, strict=False)

    for p in encoder.parameters():
        p.requires_grad = False

    return encoder


def build_linear_probe_loaders(cfg):
    train_tfm, val_tfm = build_linear_probe_transforms(cfg)

    train_split = str(getattr(cfg.data, "train_split", "train"))
    val_split = str(getattr(cfg.data, "val_split", "val"))

    ds_train = build_dataset(cfg.data, split=train_split, transform=train_tfm)
    ds_val = build_dataset(cfg.data, split=val_split, transform=val_tfm)

    sampler_train = _build_sampler(ds_train, shuffle=True, drop_last=True)
    sampler_val = _build_sampler(ds_val, shuffle=False, drop_last=False)

    collate = SupervisedCollate()

    from torch.utils.data import DataLoader

    def _loader(ds, sampler, shuffle, drop_last):
        return DataLoader(
            ds,
            batch_size=int(cfg.data.batch_size),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(getattr(cfg.data, "pin_memory", True)),
            persistent_workers=(
                bool(getattr(cfg.data, "persistent_workers", True))
                if int(cfg.data.num_workers) > 0
                else False
            ),
            prefetch_factor=(
                int(getattr(cfg.data, "prefetch_factor", 2))
                if int(cfg.data.num_workers) > 0
                else None
            ),
            worker_init_fn=seed_worker,
            collate_fn=collate,
            drop_last=drop_last,
        )

    train_loader = _loader(
        ds_train, sampler_train, shuffle=(sampler_train is None), drop_last=True
    )
    val_loader = _loader(ds_val, sampler_val, shuffle=False, drop_last=False)

    num_classes = infer_num_classes(cfg, ds_train=ds_train)

    return train_loader, val_loader, num_classes


def build_segmentation_probe_loaders(cfg):
    train_tfm, val_tfm = build_segmentation_transforms(cfg)

    train_split = str(getattr(cfg.data, "train_split", "train"))
    val_split = str(getattr(cfg.data, "val_split", "val"))

    ds_train = build_segmentation_dataset(cfg.data, split=train_split, transforms=train_tfm)
    ds_val = build_segmentation_dataset(cfg.data, split=val_split, transforms=val_tfm)

    sampler_train = _build_sampler(ds_train, shuffle=True, drop_last=True)
    sampler_val = _build_sampler(ds_val, shuffle=False, drop_last=False)

    collate = SegmentationCollate()

    from torch.utils.data import DataLoader

    def _loader(ds, sampler, shuffle, drop_last):
        return DataLoader(
            ds,
            batch_size=int(cfg.data.batch_size),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(cfg.data.num_workers),
            pin_memory=bool(getattr(cfg.data, "pin_memory", True)),
            persistent_workers=(
                bool(getattr(cfg.data, "persistent_workers", True))
                if int(cfg.data.num_workers) > 0
                else False
            ),
            prefetch_factor=(
                int(getattr(cfg.data, "prefetch_factor", 2))
                if int(cfg.data.num_workers) > 0
                else None
            ),
            worker_init_fn=seed_worker,
            collate_fn=collate,
            drop_last=drop_last,
        )

    train_loader = _loader(
        ds_train, sampler_train, shuffle=(sampler_train is None), drop_last=True
    )
    val_loader = _loader(ds_val, sampler_val, shuffle=False, drop_last=False)

    num_classes = infer_num_classes(cfg, ds_train=ds_train)

    return train_loader, val_loader, num_classes


# ------------------------------------------------------------------
# Top-level entry point (unchanged)
# ------------------------------------------------------------------

def build_for_task(cfg, device: torch.device) -> Dict[str, Any]:
    task = str(cfg.task.name)
    callbacks = build_callbacks(cfg)

    if task == "pretrain":
        model = build_pretrain_model(cfg).to(device)

        if bool(getattr(cfg, "compile", False)) and hasattr(torch, "compile"):
            model = torch.compile(model, dynamic=True)

        model = maybe_wrap_ddp(cfg, model, device)

        loader = build_pretrain_loader(cfg)
        optim, masker_optim, sched, masker_sched, wd_start, wd_end, masker_wd_start, masker_wd_end = build_pretrain_optim_sched(cfg, model)

        if cfg.resume and getattr(cfg, "init_weights", None):
            raise ValueError("Set either cfg.resume or cfg.init_weights, not both.")

        resumed_state: dict | None = None
        if cfg.resume:
            resumed_state = (
                load_checkpoint_if_available(
                    str(cfg.resume), model=model, optimizer=optim, scheduler=sched,
                    masker_optimizer=masker_optim, masker_scheduler=masker_sched,
                )
                or None
            )
        elif getattr(cfg, "init_weights", None):
            load_model_weights(
                str(cfg.init_weights),
                model=model,
                strict=bool(getattr(cfg, "init_weights_strict", False)),
            )

        return {
            "model": model,
            "loader": loader,
            "optimizer": optim,
            "masker_optimizer": masker_optim,
            "scheduler": sched,
            "masker_scheduler": masker_sched,
            "callbacks": callbacks,
            "device": device,
            "resumed_state": resumed_state,
            "wd_start": wd_start,
            "wd_end": wd_end,
            "masker_wd_start": masker_wd_start,
            "masker_wd_end": masker_wd_end,
        }

    if task == "linear_probe":
        encoder = build_linear_probe_model(cfg).to(device)
        train_loader, val_loader, num_classes = build_linear_probe_loaders(cfg)

        return {
            "encoder": encoder,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "num_classes": num_classes,
            "callbacks": callbacks,
            "device": device,
        }

    if task == "eval_suite":
        encoder = build_linear_probe_model(cfg).to(device)
        train_loader, val_loader, num_classes = build_linear_probe_loaders(cfg)

        masker = None
        modes = getattr(cfg.task, "eval_modes", {})
        needs_masker = (
            getattr(modes, "masker_probe", False)
            or getattr(modes, "masker_hard_probe", False)
        )
        if needs_masker:
            ckpt = getattr(cfg.task, "pretrained_ckpt", None)
            if ckpt:
                sd_full = torch.load(str(ckpt), map_location="cpu", weights_only=True)
                sd_full = sd_full.get("model", sd_full)
                masker = build_eval_masker(cfg, sd_full, device)
            else:
                print(
                    "[build_for_task] masker mode enabled but task.pretrained_ckpt is null;"
                    " masker modes will be skipped."
                )

        return {
            "encoder": encoder,
            "masker": masker,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "num_classes": num_classes,
            "callbacks": callbacks,
            "device": device,
        }

    if task == "segmentation_probe":
        encoder = build_linear_probe_model(cfg).to(device)
        train_loader, val_loader, num_classes = build_segmentation_probe_loaders(cfg)

        return {
            "encoder": encoder,
            "train_loader": train_loader,
            "val_loader": val_loader,
            "num_classes": num_classes,
            "callbacks": callbacks,
            "device": device,
        }

    raise ValueError(f"Unknown task.name={task}")
