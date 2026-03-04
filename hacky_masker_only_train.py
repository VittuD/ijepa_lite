#!/usr/bin/env python3
"""
hacky_masker_only_train.py

Trains ONLY the masker + predictor with a fully frozen encoder on STL-10
unlabeled data. The encoder is never updated (no EMA, no gradient).

Phase-2 diagnostic: if the masker learns content-adaptive assignments when
paired with a FROZEN encoder, that encoder IS providing content-informative
features that a fresh masker can learn to read. If it re-converges to
horizontal thirds, the observed behavior is a co-adaptation artifact — it
only arises during joint optimization, not from either component alone.

Key metric logged: spatial_pos_std
  Per-position within-batch standard deviation of bin assignments (0=ctx,
  1=tgt, 2=ign), averaged over all 144 patch positions.
  Near 0 → purely positional (same position always same bin).
  Higher  → content-adaptive (same position gets different bins per image).

The saved checkpoint uses latent_masker.* key prefix so it is directly
compatible with hacky_visualize_ign_patches.py via:
  python hacky_visualize_ign_patches.py \\
      --encoder-ckpt /path/to/original_encoder_ckpt.pt \\
      --masker-ckpt  <out_dir>/epoch_XXXX.pt

Usage:
  # Jointly-trained encoder (main diagnostic)
  python hacky_masker_only_train.py \\
      --encoder-ckpt /path/to/mi_coupled_last.pt \\
      --out-dir masker_only_joint_enc

  # Vanilla JEPA encoder (control)
  python hacky_masker_only_train.py \\
      --encoder-ckpt /path/to/vanilla_last.pt \\
      --out-dir masker_only_vanilla_enc
"""
import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as tv_datasets, transforms
from torchvision.models.vision_transformer import VisionTransformer

import ijepa_lite.masking.mi_masker  # noqa: F401 — populates registry
from ijepa_lite.masking.mi_masker import MIRateMasker
from ijepa_lite.models.predictor import Predictor
from ijepa_lite.models.vit_tokens import ViTTokens, _remove_classifier_head

# ---------------------------------------------------------------------------
# Architecture constants (stl10 ViT-S/8)
# ---------------------------------------------------------------------------
IMG_SIZE   = 96
PATCH_SIZE = 8
EMBED_DIM  = 384
DEPTH      = 12
NUM_HEADS  = 6
PRED_DIM   = 192
PRED_DEPTH = 2
PRED_HEADS = 6
N_PATCHES  = (IMG_SIZE // PATCH_SIZE) ** 2   # 144

SMOOTH_L1_BETA = 2.0
IMAGENET_MEAN  = (0.485, 0.456, 0.406)
IMAGENET_STD   = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_encoder(ckpt_path: str, device: torch.device) -> ViTTokens:
    vit = VisionTransformer(
        image_size=IMG_SIZE, patch_size=PATCH_SIZE, num_layers=DEPTH,
        num_heads=NUM_HEADS, hidden_dim=EMBED_DIM, mlp_dim=EMBED_DIM * 4,
        num_classes=1000,
    )
    _remove_classifier_head(vit)
    enc = ViTTokens(vit)

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = sd.get("model", sd)
    prefix = "target_encoder."
    enc_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, _ = enc.load_state_dict(enc_sd, strict=False)
    if missing:
        print(f"  [encoder] missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    enc.requires_grad_(False)
    return enc.to(device).eval()


def build_fresh_masker(device: torch.device, alpha_min: float = 0.01) -> MIRateMasker:
    return MIRateMasker(
        dim=EMBED_DIM, predictor_dim=PRED_DIM, depth=PRED_DEPTH,
        num_heads=PRED_HEADS, mlp_ratio=4.0, dropout=0.0,
        num_patches=N_PATCHES,
        lam_min=0.01, lam_max=1.0, alpha_min=alpha_min, alpha_max=0.5,
        coupled_scalarization=True, ratio_logit_std=1.0,
        h_floor=0.1, floor_weight=5.0, lam_warmup_epochs=0,
    ).to(device)


def build_fresh_predictor(device: torch.device) -> Predictor:
    return Predictor(
        dim=EMBED_DIM, predictor_dim=PRED_DIM, depth=PRED_DEPTH,
        num_heads=PRED_HEADS, mlp_ratio=4.0, dropout=0.0,
        num_patches=N_PATCHES,
    ).to(device)


def load_predictor_weights(predictor: Predictor, ckpt_path: str) -> None:
    """Load predictor weights from a checkpoint.

    Handles two formats:
      - Main JEPA checkpoint: model["predictor.*"]
      - Masker-only checkpoint: ckpt["predictor"] (flat state_dict)
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model" in sd:
        pred_sd = {k[len("predictor."):]: v
                   for k, v in sd["model"].items() if k.startswith("predictor.")}
    elif "predictor" in sd:
        pred_sd = sd["predictor"]
    else:
        pred_sd = {}

    if not pred_sd:
        print("  [predictor] no predictor keys found — using fresh init")
        return

    missing, _ = predictor.load_state_dict(pred_sd, strict=False)
    if missing:
        print(f"  [predictor] missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    print(f"  [predictor] loaded {len(pred_sd)} tensors from {ckpt_path}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_loader(data_root: str, batch_size: int):
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(
            IMG_SIZE, scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = tv_datasets.STL10(data_root, split="unlabeled", download=False, transform=tfm)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=8, pin_memory=True,
    )

# ---------------------------------------------------------------------------
# Spatial content-adaptivity metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def spatial_pos_std(
    ctx_idx: torch.Tensor,  # (B, nctx)
    tgt_idx: torch.Tensor,  # (B, ntgt)
    device: torch.device,
) -> float:
    """
    Average within-position std of bin assignment across the batch.

    For each patch position j, compute std(bin_j) across B images where
    bin ∈ {0=ctx, 1=tgt, 2=ign}. Average the N per-position stds.

    Near 0 → purely positional masker (same position → same bin for all images).
    Higher  → content-adaptive (same position varies across images).
    """
    B = ctx_idx.shape[0]
    bin_map = torch.full((B, N_PATCHES), 2.0, device=device)  # default: ign
    ctx_mask = torch.zeros(B, N_PATCHES, dtype=torch.bool, device=device)
    tgt_mask = torch.zeros(B, N_PATCHES, dtype=torch.bool, device=device)
    ctx_mask.scatter_(1, ctx_idx, True)
    tgt_mask.scatter_(1, tgt_idx, True)
    bin_map[ctx_mask] = 0.0
    bin_map[tgt_mask] = 1.0
    # std across batch for each position → (N,) → mean
    return float(bin_map.std(dim=0).mean().item())

# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(out_dir: Path, epoch: int, masker, predictor, optimizer):
    """Save in latent_masker.* format for viz script compatibility."""
    ckpt = {
        "model": {
            **{"latent_masker." + k: v for k, v in masker.state_dict().items()},
        },
        "predictor": predictor.state_dict(),
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
    }
    path = out_dir / f"epoch_{epoch:04d}.pt"
    torch.save(ckpt, path)
    # Keep only most-recent + epoch 0 to avoid disk bloat
    for old in sorted(out_dir.glob("epoch_*.pt")):
        if old != path and old.stem != "epoch_0000":
            old.unlink(missing_ok=True)
    print(f"  Checkpoint saved → {path}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder-ckpt",   required=True,
                        help="Frozen encoder: target_encoder.* from any JEPA checkpoint")
    parser.add_argument("--predictor-ckpt", default=None,
                        help="Warm-start predictor from predictor.* in any JEPA or "
                             "masker-only checkpoint. Fresh init if omitted.")
    parser.add_argument("--alpha-min",    type=float, default=0.01,
                        help="Minimum context ratio for masker (default 0.01 ≈ 1 patch). "
                             "Raise to prevent nCtx=1 collapse, e.g. 0.15 (~22 patches) "
                             "or 0.25 (~36 patches).")
    parser.add_argument("--data-root",    default=os.environ.get("FAST", "/scratch") + "/datasets/")
    parser.add_argument("--out-dir",      default="masker_only_run")
    parser.add_argument("--epochs",       type=int,   default=400)
    parser.add_argument("--batch-size",   type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--log-every",    type=int,   default=50,  help="Steps between log lines")
    parser.add_argument("--save-every",   type=int,   default=50,  help="Epochs between ckpt saves")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    amp    = device.type == "cuda"

    print(f"Loading encoder from : {args.encoder_ckpt}")
    encoder = build_encoder(args.encoder_ckpt, device)

    print("Building fresh masker + predictor ...")
    masker    = build_fresh_masker(device, alpha_min=args.alpha_min)
    print(f"  alpha_min={args.alpha_min}  (min nCtx ≈ {int(args.alpha_min * N_PATCHES)})")
    predictor = build_fresh_predictor(device)
    if args.predictor_ckpt:
        print(f"Warm-starting predictor from : {args.predictor_ckpt}")
        load_predictor_weights(predictor, args.predictor_ckpt)
    n_params  = sum(p.numel() for p in list(masker.parameters()) + list(predictor.parameters()))
    print(f"  Trainable params : {n_params:,}  (encoder frozen)")

    optimizer = torch.optim.AdamW(
        list(masker.parameters()) + list(predictor.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    loader      = build_loader(args.data_root, args.batch_size)
    total_steps = args.epochs * len(loader)
    print(f"\nSTL-10 unlabeled: {len(loader.dataset):,} images  "
          f"steps/epoch={len(loader)}  total_steps={total_steps}")
    print(f"lr={args.lr}  wd={args.weight_decay}  batch={args.batch_size}  epochs={args.epochs}\n")

    global_step = 0
    acc: dict[str, float] = dict(loss=0, recon=0, mi=0, surprise=0, sps=0)
    acc_n = 0

    for epoch in range(args.epochs):
        masker.train()
        predictor.train()

        for batch in loader:
            images = batch[0].to(device, non_blocking=True)  # STL-10: (img, label=-1)
            B = images.size(0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):

                # --- Frozen encoder: full token bank (no grad) ---
                with torch.no_grad():
                    ema_tokens = encoder(images)  # (B, N, D) — ViT internal LN applied

                # --- Masker forward (grads flow through masker params) ---
                mask_output = masker(ema_tokens, ema_full=ema_tokens)
                ctx_idx = mask_output.context_idx  # (B, nctx)
                tgt_idx = mask_output.target_idx   # (B, ntgt)

                # --- Context encoder: selected patches only (no grad) ---
                with torch.no_grad():
                    ctx_tokens = encoder(images, keep_idx=ctx_idx)  # (B, nctx, D)
                    # Extra LN on target tokens only — matches ijepa.py line 160-162
                    tgt_all = F.layer_norm(ema_tokens, (EMBED_DIM,))
                    tgt_tokens = tgt_all.gather(
                        1, tgt_idx.unsqueeze(-1).expand(-1, -1, EMBED_DIM)
                    )  # (B, ntgt, D)

                # --- Predictor forward (grads flow) ---
                pred = predictor(ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx)

                # --- Reconstruction loss ---
                patch_loss = F.smooth_l1_loss(
                    pred, tgt_tokens, beta=SMOOTH_L1_BETA, reduction="none"
                )  # (B, ntgt)
                recon_loss = patch_loss.mean()

                # --- Full objective via masker aux_loss (owns_loss=True) ---
                total_loss = masker.aux_loss(
                    mask_output, recon_loss, patch_loss=patch_loss.detach()
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(masker.parameters()) + list(predictor.parameters()), 1.0
            )
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            acc["loss"]    += float(total_loss.detach())
            acc["recon"]   += float(recon_loss.detach())
            acc["mi"]      += float(mask_output.aux.get("mi_rate", 0.0))
            acc["surprise"] += float(mask_output.aux.get("surprise_mean", 0.0))
            acc["sps"]     += spatial_pos_std(ctx_idx.detach(), tgt_idx.detach(), device)
            acc_n          += 1

            if global_step % args.log_every == 0:
                nctx = int(ctx_idx.shape[1])
                ntgt = int(tgt_idx.shape[1])
                nign = N_PATCHES - nctx - ntgt
                print(
                    f"  ep {epoch:3d}  step {global_step:6d}"
                    f"  loss={acc['loss']/acc_n:.4f}"
                    f"  recon={acc['recon']/acc_n:.4f}"
                    f"  mi_rate={acc['mi']/acc_n:.4f}"
                    f"  surprise={acc['surprise']/acc_n:.4f}"
                    f"  spatial_pos_std={acc['sps']/acc_n:.4f}"
                    f"  nctx={nctx} ntgt={ntgt} nign={nign}"
                )
                acc = dict(loss=0, recon=0, mi=0, surprise=0, sps=0)
                acc_n = 0

        sched.step()

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(out_dir, epoch, masker, predictor, optimizer)

    print(f"\nDone. Checkpoints in ./{out_dir}/")
    print("Visualise with:")
    print(f"  python hacky_visualize_ign_patches.py \\")
    print(f"      --encoder-ckpt {args.encoder_ckpt} \\")
    print(f"      --masker-ckpt  {out_dir}/epoch_{args.epochs-1:04d}.pt \\")
    print(f"      --out-dir      ign_viz_masker_only")


if __name__ == "__main__":
    main()
