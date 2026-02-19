# ijepa_lite

A minimal, readable PyTorch implementation of **i-JEPA-style token prediction** with:
- **CPU-side mask generation** in DataLoader workers (overlapped with GPU compute)
- **EMA target encoder** updates (online/context → target)
- **Hydra** configuration
- Optional **DDP** + **Weights & Biases**
- A built-in **linear probe** evaluation loop (frozen encoder + linear head)

## What this repo does

**Pretraining (`task.name=pretrain`)**
- A **context encoder** sees only *context* patch tokens (targets are removed before attention).
- A **target encoder** sees the full image (no grads; updated by EMA).
- A small **predictor** uses context tokens + learned mask tokens to predict target embeddings.
- Loss is token-level regression (MSE by default).

**Linear probing (`task.name=linear_probe`)**
- Loads either the pretrained **target** or **context** encoder from a checkpoint.
- Freezes encoder, trains a **linear head** on labeled data.
- Reports top-1 accuracy.

---

## Install

~~~bash
# (optional) create env
python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install torch torchvision hydra-core omegaconf wandb
~~~

> Notes:
> - `wandb` is optional (only used when `logger.name=wandb`).
> - For multi-GPU, use `torchrun` (see below).

---

## Project layout

~~~text
ijepa_lite/
  build.py                 # builds model/optim/loader for each task
  run.py                   # Hydra entrypoint
  data/                    # datasets, transforms, collate
  masking/                 # block + multiblock mask generators
  models/                  # ViT wrapper + predictor + EMA helpers
  engine/                  # train loop + linear probe loop
  callbacks/               # progress, checkpoint, wandb
  utils/                   # dist utils, seeding, meters, metrics
configs/
  config.yaml              # Hydra root config (expected)
~~~

---

## Quickstart: Pretraining

### 1) Choose a dataset

Currently supported in `ijepa_lite/data/datasets.py`:
- `cifar10` (torchvision download supported)
- `stl10` (torchvision download supported)
- `imagenet` (expects an ImageFolder-style directory; **no auto-download**)

**ImageNet on disk layout (required):**
~~~text
/path/to/imagenet/
  train/
    n01440764/
    ...
  val/
    n01440764/
    ...
~~~

### 2) Run pretraining

Single GPU:
~~~bash
python ijepa_lite/run.py \
  task.name=pretrain \
  data.name=stl10 data.root=/scratch/ijepa_lite/data data.download=true \
  model.arch=vit_small_16 model.image_size=96 model.patch_size=8 \
  model.embed_dim=512 model.depth=6 model.num_heads=8 \
  predictor.depth=1 predictor.num_heads=2 predictor.predictor_dim=192 \
  masking.name=multiblock masking.target_ratio=0.25 masking.context_ratio=0.75 masking.num_target_blocks=4 \
  train.epochs=100 train.log_every=50
~~~

Multi-GPU (DDP):
~~~bash
torchrun --nproc_per_node=8 ijepa_lite/run.py \
  task.name=pretrain \
  distributed.backend=nccl \
  data.name=stl10 data.root=/scratch/ijepa_lite/data data.download=true \
  model.arch=vit_small_16 model.image_size=96 model.patch_size=8 \
  train.epochs=100
~~~

Or create an experiment config file (e.g. `configs/exp01.yaml`) with your desired settings and run:
~~~bash
python ijepa_lite/run.py \
  --config-name exp01
~~~
---

## Quickstart: Linear probe

Linear probing expects a JEPA checkpoint containing:
- `model` state dict with keys like `target_encoder.*` and `context_encoder.*`

Run:
~~~bash
python ijepa_lite/run.py \
  task.name=linear_probe \
  task.pretrained_ckpt=checkpoints/last.pt \
  task.encoder=target \
  data.name=cifar10 data.root=/scratch/ijepa_lite/data data.download=true \
  train.epochs=90 train.lr=0.1 train.weight_decay=0.0
~~~

Key options:
- `task.encoder`: `target` (default) or `context`
- `task.pool`: pooling over patch tokens (currently supports `mean`)
- `task.amp`: AMP for linear probe (default true on CUDA)
- `task.sched`: optional LR schedule for head (`step` or `none`)

Example step schedule:
~~~yaml
task:
  sched:
    name: step
    step_size: 30
    gamma: 0.1
~~~

Linear probe checkpoints are saved to:
~~~text
checkpoints/linear_probe_best.pt
checkpoints/linear_probe_last.pt
~~~

---

## Masking

Mask generation happens inside the **DataLoader collate fn** (`IJEPACollate`) on CPU, so it overlaps GPU compute.

Two masking modes:

### 1) `masking.name=block`
Single target block sampled per image.

Returns:
- `context_idx`: `(B, Nctx)`
- `target_idx`: `(B, Ntgt)`

### 2) `masking.name=multiblock`
Multiple target blocks + one context rectangle.

Returns:
- `context_idx`: `(B, Nctx)`
- `target_idx`: `(B, M, K)` where `M=num_target_blocks`

Important behavior:
- When `allow_overlap=false`, target patches are removed from the context rectangle.

---

## Logging (stdout + W&B)

By default you get stdout logs via `ProgressCallback`.

Enable W&B:
~~~bash
python ijepa_lite/run.py \
  logger.name=wandb \
  logger.project=my_project \
  logger.run_name=exp01 \
  logger.mode=online
~~~

Useful W&B knobs:
- `logger.entity`
- `logger.group`
- `logger.tags=[...]`
- `logger.notes`
- `logger.log_model=true|false` (logs checkpoint artifact)

---

## Checkpointing and resume

Pretraining saves:
- `checkpoints/last.pt`

Resume training:
~~~bash
python ijepa_lite/run.py \
  task.name=pretrain \
  resume=checkpoints/last.pt
~~~

The checkpoint includes:
- `model`, `optimizer`, `scheduler`, `scaler`
- training `state` (`epoch`, `global_step`, etc.)
- `ema_start` so EMA scheduling resumes correctly

---

## Config reference (high-level)

This repo expects these top-level config groups/keys (see `ijepa_lite/build.py` and `ijepa_lite/run.py`):

- `task.name`: `pretrain` | `linear_probe`
- `model.*`: `arch`, `image_size`, `patch_size`, `embed_dim`, `depth`, `num_heads`, `ema_momentum=[start,end]`
- `predictor.*`: `depth`, `num_heads`, `predictor_dim`, optional `mlp_ratio`, `dropout`
- `data.*`: `name`, `root`, `download`, `batch_size`, `num_workers`, `pin_memory`, `prefetch_factor`, `persistent_workers`, and split names
- `masking.*`: `name`, `target_ratio`, `context_ratio`, plus block/multiblock fields
- `optim.*`: AdamW params (`lr`, `weight_decay`, `betas`, `eps`, optional `final_weight_decay`)
- `sched.*`: warmup + cosine LR (`warmup_epochs`, optional `min_lr`)
- `train.*`: `epochs`, `log_every`, `amp`, `grad_clip_norm`, `save_every_epochs`
- `distributed.*`: `backend`, `timeout_minutes`, optional `find_unused_parameters`
- `logger.*`: W&B configuration when enabled
- `seed`, `device`, `compile`, `exp_name`, `resume`

---

## Adding more datasets (CIFAR100, ImageNet100, HF ImageNet-1k 128x128)

Out of the box, `build_dataset()` currently handles **cifar10**, **stl10**, and **imagenet** (ImageFolder). If you want to hardcode more choices (like `cifar100`, `imagenet100`, or an HF dataset), extend:

- `ijepa_lite/data/datasets.py` (add a `specs[...]` entry + optional download path)
- `ijepa_lite/data/classes.py` (num classes mapping is already prepared for several names)

Minimal pattern to add a dataset entry:
~~~python
specs["cifar100"] = (
    ("train", "val", "test"),
    lambda r, s, t: datasets.CIFAR100(
        root=r, train=(s == "train"), download=False, transform=t
    ),
)
~~~

For Hugging Face datasets, you’d typically create a tiny custom Dataset wrapper (so transforms work consistently) and then add it to `specs`.

---

## Troubleshooting

- **DDP hangs at start**: ensure `torchrun` is used and `distributed.backend=nccl` only when CUDA is available.
- **ImageNet not found**: verify the `root/train` and `root/val` folder structure.
- **Slow dataloading**: increase `data.num_workers`, set `persistent_workers=true`, and keep mask generation in collate (default).
- **OOM**: reduce `data.batch_size`, reduce model size (`embed_dim`, `depth`), or reduce `image_size`.

---

# TODOs
- [ ] Add HF ImageNet 128x128 dataset
- [ ] Add more mask related logging metrics (e.g. mask distribution stats)
- [ ] Add non-linear probe eval (e.g. Attentive Probing)
- [ ] Use Python `logging` module instead of print statements

---

## License

MIT License (see `LICENSE` file).
