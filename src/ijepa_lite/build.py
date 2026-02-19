from __future__ import annotations

import math
from typing import Any, Dict

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
from ijepa_lite.losses.vanilla import VanillaTokenLoss
from ijepa_lite.masking.block_mask import BlockMaskGenerator
from ijepa_lite.masking.multiblock_mask import MultiBlockMaskGenerator
from ijepa_lite.models.ijepa import IJEPAModel
from ijepa_lite.models.predictor import Predictor
from ijepa_lite.models.vit_tokens import build_torchvision_vit_tokens
from ijepa_lite.utils.dist import get_rank, get_world_size, is_distributed
from ijepa_lite.utils.seed import seed_worker


def _build_masker(cfg):
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
        )

    raise ValueError(f"Unknown masking.name={name}")


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

    # Optional logger (W&B)
    logger_cfg = getattr(cfg, "logger", None)
    logger_name = (
        str(getattr(logger_cfg, "name", "none")).lower() if logger_cfg else "none"
    )

    if logger_name == "wandb":
        # Lazy import so logger=none doesn't require wandb installed.
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


def build_pretrain_model(cfg) -> torch.nn.Module:
    context = build_torchvision_vit_tokens(cfg.model)
    target = build_torchvision_vit_tokens(cfg.model)
    target.load_state_dict(context.state_dict(), strict=True)

    num_patches = (int(cfg.model.image_size) // int(cfg.model.patch_size)) ** 2

    pred = Predictor(
        dim=int(cfg.model.embed_dim),
        predictor_dim=int(cfg.predictor.predictor_dim),
        depth=int(cfg.predictor.depth),
        num_heads=int(cfg.predictor.num_heads),
        mlp_ratio=float(getattr(cfg.predictor, "mlp_ratio", 4.0)),
        dropout=float(getattr(cfg.predictor, "dropout", 0.0)),
        num_patches=num_patches,
    )

    loss_fn = VanillaTokenLoss(
        normalize=bool(getattr(cfg.loss, "normalize", False)),
        kind=str(getattr(cfg.loss, "kind", "mse")),
    )

    ema_m = float(cfg.model.ema_momentum[0])

    return IJEPAModel(
        context_encoder=context,
        target_encoder=target,
        predictor=pred,
        loss_fn=loss_fn,
        ema_momentum=ema_m,
        mask_generator=None,
    )


def build_pretrain_loader(cfg):
    tfm = build_pretrain_transform(cfg)
    masker = _build_masker(cfg)

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


def build_pretrain_optim_sched(cfg, model: torch.nn.Module):
    wd_start = float(cfg.optim.weight_decay)
    wd_end = float(getattr(cfg.optim, "final_weight_decay", wd_start))

    lr = float(cfg.optim.lr)
    betas = tuple(float(x) for x in cfg.optim.betas)
    eps = float(cfg.optim.eps)

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
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

        enc_sd = {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
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

    # IMPORTANT: pass full cfg (needs cfg.task + cfg.data)
    num_classes = infer_num_classes(cfg, ds_train=ds_train)

    return train_loader, val_loader, num_classes


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

    raise ValueError(f"Unknown task.name={task}")
