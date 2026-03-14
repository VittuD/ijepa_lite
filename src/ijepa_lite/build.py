from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from ijepa_lite.callbacks.ckpt_cb import CheckpointCallback
from ijepa_lite.callbacks.handler import CallbackHandler
from ijepa_lite.callbacks.progress_cb import ProgressCallback
from ijepa_lite.data.classes import infer_num_classes
from ijepa_lite.data.collate import IJEPACollate, SupervisedCollate
from ijepa_lite.data.datasets import build_dataset
from ijepa_lite.data.transforms import (
    build_linear_probe_transforms,
    build_pretrain_transform,
)
from ijepa_lite.engine.checkpoint import load_checkpoint_if_available
from ijepa_lite.losses.rd_loss import RateDistSurpriseLoss
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
        "target_ratio": float(getattr(cfg.masking, "target_ratio", 0.25)),
        "context_ratio": float(getattr(cfg.masking, "context_ratio", 0.75)),
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
    if eval_every > 0:
        from ijepa_lite.callbacks.eval_cb import InlineEvalCallback
        cbs.append(InlineEvalCallback())

    # Opt-in visualization (piggybacks on save_every cadence)
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

    kwargs = dict(broadcast_buffers=False)
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

    if kind in ("rd_3way", "mi_3way"):
        base_kind = str(getattr(cfg.loss, "base_kind", "smooth_l1"))
        return VanillaTokenLoss(normalize=normalize, kind=base_kind)

    return VanillaTokenLoss(normalize=normalize, kind=kind)


# ------------------------------------------------------------------
# Pretrain model
# ------------------------------------------------------------------

def build_pretrain_model(cfg) -> torch.nn.Module:
    context = build_torchvision_vit_tokens(cfg.model)
    target = build_torchvision_vit_tokens(cfg.model)
    target.load_state_dict(context.state_dict(), strict=True)

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

    ema_m = float(cfg.model.ema_momentum[0])

    compressor = _build_compressor(cfg)
    latent_masker = _build_latent_masker(cfg, compressor)

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
    masker_lr_scale = float(getattr(cfg.optim, "masker_lr_scale", 1.0))

    # Split params: masker submodules get a separate LR group
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
    if getattr(cfg, "sched", None) is not None:
        name = str(getattr(cfg.sched, "name", "warmup_cosine")).lower()
        if name == "warmup_cosine":
            warmup = int(getattr(cfg.sched, "warmup_epochs", 0))
            min_lr = float(getattr(cfg.sched, "min_lr", 0.0))
            total = int(cfg.train.epochs)

            def _lr_lambda(epoch: int):
                if warmup > 0 and epoch < warmup:
                    return float(epoch + 1) / float(max(1, warmup))
                progress = (epoch - warmup) / float(max(1, total - warmup))
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return (min_lr / lr) + (1.0 - (min_lr / lr)) * cosine

            sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)
        elif name == "none":
            sched = None
        else:
            raise ValueError(f"Unknown sched.name={name}")

    return opt, sched, wd_start, wd_end


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

        prefer = str(getattr(cfg.task, "encoder", "target")).lower()
        prefix = (
            "target_encoder." if prefer in ("target", "ema") else "context_encoder."
        )

        enc_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        if not enc_sd:
            enc_sd = sd

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
        optim, sched, wd_start, wd_end = build_pretrain_optim_sched(cfg, model)

        resumed_state: dict | None = None
        if cfg.resume:
            resumed_state = (
                load_checkpoint_if_available(
                    str(cfg.resume), model=model, optimizer=optim, scheduler=sched
                )
                or None
            )

        return {
            "model": model,
            "loader": loader,
            "optimizer": optim,
            "scheduler": sched,
            "callbacks": callbacks,
            "device": device,
            "resumed_state": resumed_state,
            "wd_start": wd_start,
            "wd_end": wd_end,
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

    raise ValueError(f"Unknown task.name={task}")