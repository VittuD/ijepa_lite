# Contributing to `ijepa_lite`

This repo is optimized for research iteration, not framework cleverness. The fastest way to contribute well is to preserve that property: add new ideas through the existing seams, keep the main execution path explicit, and make experiments easy to reproduce from config and launchers.

## Start Here

Before editing, read these files in this order:

1. `src/ijepa_lite/run.py`
2. `src/ijepa_lite/build.py`
3. `src/ijepa_lite/models/ijepa.py`
4. `src/ijepa_lite/engine/train_loop.py`
5. The module you plan to extend (`masking/`, `losses/`, `callbacks/`, `data/`, or `configs/`)

That sequence gives you the repo's control flow without needing to read everything.

## Local Workflow

Install the repo in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Main entrypoint:

```bash
python -m ijepa_lite.run experiment=debug
```

Useful contributor checks:

```bash
python -m compileall src/ijepa_lite
python -m ijepa_lite.run experiment=debug logger=none
```

Hydra changes the working directory into `outputs/.../rank<rank>`, so keep that in mind when introducing new relative paths. Dataset roots and checkpoint paths are usually better supplied explicitly from config or launcher overrides.

## Core Design Philosophy

### 1. Keep the top-level flow thin and explicit

- `run.py` should stay small: initialize runtime concerns, build a task bundle, dispatch to one loop.
- `build.py` is the composition layer. Most wiring changes belong there.
- `engine/train_loop.py` should remain generic. Avoid adding feature-specific branching unless the feature truly changes optimization semantics.

### 2. Prefer extension points over invasive edits

The repo already has clear seams:

- New latent masker: `src/ijepa_lite/masking/`
- New deterministic collate masker: `src/ijepa_lite/masking/`
- New atomic masker term: `src/ijepa_lite/losses/terms.py`
- New callback or logging behavior: `src/ijepa_lite/callbacks/`
- New dataset or transform: `src/ijepa_lite/data/`
- New experiment recipe: `configs/experiment/`
- New cluster launcher: `slurm/`

If a change can be localized to one seam, do that instead of teaching unrelated modules about it.

### 3. Config is part of the architecture

Hydra composition is not an afterthought here. A feature is not fully integrated until contributors can:

- enable it from config,
- override it from CLI or sbatch,
- and tell from logs which variant ran.

In practice this means code changes usually come with one or more of:

- a config group entry,
- an experiment config,
- a launcher override,
- new metrics under a stable namespace.

### 4. Separate optimization semantics from monitoring semantics

This repo frequently logs values that are not the optimized objective.

Examples:

- `train/loss` is the full objective being optimized.
- `train/reconstruction_loss` is often kept as a monitoring scalar even when a learned masker owns the real loss.
- `mask/*` metrics are diagnostics and ablation handles, not necessarily supervision targets.

When adding a feature, be explicit about:

- what is optimized,
- what is only monitored,
- and what should remain comparable across runs.

### 5. Respect the CPU-mask / GPU-mask split

There are two masking paths by design:

- Deterministic maskers run in `IJEPACollate` on CPU and output hard indices.
- Learned maskers run inside `IJEPAModel.forward()` on GPU and may backprop through soft assignments.

Do not blur the two:

- CPU collate maskers must be picklable and data-loader safe.
- Learned maskers are `nn.Module`s and belong in the model path.
- If a learned masker is active, collate should stay dumb.

### 6. Side effects must be DDP-safe and resumable

The repo assumes:

- rank-0 owns side effects such as logging and artifact handling,
- checkpoints carry enough state to resume correctly,
- callback runtime objects are passed through `state` private keys,
- and barriers are used deliberately when rank skew could hang DDP.

If you add logging, checkpoint state, warmup behavior, or eval hooks, think about distributed execution first.

## Repo Mental Model

Use this as the ownership map for changes:

| Path | Responsibility | Edit when... |
| --- | --- | --- |
| `src/ijepa_lite/run.py` | entrypoint, task dispatch, built-in masker imports | adding a new task or registering a new built-in latent masker import |
| `src/ijepa_lite/build.py` | object construction and task bundles | wiring models, loaders, optimizers, callbacks, config-driven behavior |
| `src/ijepa_lite/models/ijepa.py` | JEPA forward pass and loss assembly | changing core pretraining semantics |
| `src/ijepa_lite/engine/train_loop.py` | optimization, EMA, metrics emission, callback cadence | changing optimization flow or training-time bookkeeping |
| `src/ijepa_lite/masking/` | mask generation and learned maskers | adding or modifying masking strategies |
| `src/ijepa_lite/losses/` | reconstruction, RD/MI objectives, atomic terms | adding loss logic or diagnostics tied to loss structure |
| `src/ijepa_lite/callbacks/` | progress, checkpoints, W&B, inline eval, viz | adding orthogonal runtime behavior |
| `src/ijepa_lite/data/` | datasets, transforms, collate | new datasets or pretrain/eval data behavior |
| `configs/` | Hydra composition surface | exposing or documenting a feature |
| `slurm/` | reproducible launch recipes | making a runnable cluster variant |

## Working Rules

### Keep changes opt-in by default

New research behavior should usually be enabled by config, not silently change the default path.

Good:

- new term with weight `[0, 0]` until selected,
- new callback gated by `train.*` or `logger.*`,
- new launcher as a sibling of an existing one.

Risky:

- changing the default experiment behavior without a narrow reason,
- adding hidden logic that activates from unrelated configs,
- moving metrics or renaming keys casually.

### Keep naming and logging stable

Metric namespaces already encode intent:

- `train/*`: optimization and training scalars
- `mask/*`: masking diagnostics and learned-masker internals
- `ema/*`: EMA alignment and momentum
- `_hist/*`: histogram payloads for W&B conversion

Reuse those namespaces instead of inventing new ad hoc prefixes.

### Preserve checkpoint semantics

There are two distinct weight-loading modes:

- `resume`: restore optimizer/scheduler/scaler/training state
- `init_weights`: load model weights only, then train with fresh optimization state

Do not overload one to behave like the other.

### Prefer composition over duplication

Common pattern in this repo:

- define a solid base experiment in `configs/experiment/`,
- override only the delta in sbatch or CLI,
- keep launchers comparable by changing the smallest possible set of knobs.

Avoid copying a full experiment or launcher unless the variant is genuinely independent.

### Comments should explain invariants, not syntax

The codebase already uses comments well around non-obvious behavior:

- DDP/barrier reasoning,
- warmup semantics,
- loss ownership,
- callback ordering,
- monitoring-vs-objective distinctions.

Match that style. Add comments where future contributors would otherwise make the wrong edit.

## Practical Contribution Recipes

### Add a new atomic masker term

Use this path when the masker architecture stays the same but the compositional objective changes.

1. Add a `MaskerTerm` subclass in `src/ijepa_lite/losses/terms.py`.
2. Return `(scalar_to_minimize, logs_dict)`.
3. Register it in `TERM_REGISTRY`.
4. Expose a config stanza in a latent masker config such as `configs/masking/latent/mi_3way.yaml`.
5. Keep it disabled by default with `weight: [0, 0]` unless it is replacing an existing term intentionally.
6. Log diagnostics under meaningful `mask/*` keys.

Usually you do not need to touch `build.py` for a term-only addition. `CompositeMaskerLoss` already instantiates active terms from config.

### Add a new learned latent masker

Use this path when mask generation itself changes and should depend on model features.

1. Create a new module in `src/ijepa_lite/masking/`.
2. Subclass `LatentMasker`.
3. Decorate it with `@register("your_name")`.
4. Import the module in `src/ijepa_lite/run.py` so registration happens.
5. Add a Hydra config in `configs/masking/latent/your_name.yaml`.
6. If it needs compressed tokens, make sure the compressor config is compatible.
7. If it defines its own objective, document whether `owns_loss` is `True` or `False`.

If the constructor needs new config fields, prefer declaring explicit `__init__` args and letting `masking.registry.build_latent_masker()` filter the global kwarg superset for you.

### Add a new deterministic collate masker

Use this only when masks can be generated without encoder features.

1. Add a `CollateMasker` implementation under `src/ijepa_lite/masking/`.
2. Wire it into `_build_collate_masker()` in `src/ijepa_lite/build.py`.
3. Keep its output contract to hard index tensors through `MaskOutput`.
4. Do not introduce GPU tensors, model dependencies, or non-picklable state.

If the collate path becomes feature-dependent, it should be a latent masker instead.

### Add a new callback or runtime hook

Use callbacks for orthogonal behavior, not for core forward semantics.

1. Add a class under `src/ijepa_lite/callbacks/`.
2. Keep it side-effect safe on non-zero ranks.
3. Wire it in `build_callbacks()` in `src/ijepa_lite/build.py`.
4. If it needs runtime objects, read them from the shared `state` dict.
5. Use private state keys prefixed with `_` for non-serializable runtime values.

Examples already in the repo:

- progress printing,
- checkpointing,
- W&B logging,
- inline eval,
- visualization.

### Add a new task mode

Use this when the repo needs a new top-level execution mode, not just a pretrain variant.

1. Add or extend the loop/evaluator under `src/ijepa_lite/engine/`.
2. Teach `build_for_task()` in `src/ijepa_lite/build.py` to construct the required bundle.
3. Add a `configs/task/<name>.yaml`.
4. Dispatch it explicitly from `src/ijepa_lite/run.py`.
5. Keep its contract narrow: builder assembles objects, runner dispatches, engine executes.

If the change can fit inside `pretrain`, `linear_probe`, or `eval_suite`, prefer that over creating a new task.

### Add a new dataset or transform path

1. Put dataset creation in `src/ijepa_lite/data/datasets.py`.
2. Put augmentation logic in `src/ijepa_lite/data/transforms.py`.
3. Keep collate responsibilities narrow: stacking plus optional deterministic masking.
4. Expose dataset choices through `configs/data/`.

Avoid burying dataset-specific behavior inside the training loop.

### Add a new experiment and launcher

Preferred pattern:

1. Compose a reusable base experiment in `configs/experiment/`.
2. Put durable defaults there.
3. Put cluster-, dataset-, or run-specific overrides in a sibling `slurm/*.sbatch`.

Anchor launchers to an existing experiment when possible. For example, several ImageNet sbatches reuse `experiment=stl10_vits_ps8_mi` and override only the geometry, dataset, term weights, and runtime cadence.

## Minimal Examples

### Example 1: Add a new MI term

If you want a new diversity penalty for the MI masker:

- implement the term in `src/ijepa_lite/losses/terms.py`,
- add it to `TERM_REGISTRY`,
- add a disabled stanza in `configs/masking/latent/mi_3way.yaml`,
- enable it from CLI or sbatch with a weight override.

You should not need to edit `train_loop.py`.

### Example 2: Add a new learned masking strategy

If you want a predictor-conditioned masking policy:

- implement it as a `LatentMasker`,
- register it,
- import it in `run.py`,
- define `configs/masking/latent/<name>.yaml`,
- select it from an experiment config.

You should not route it through `IJEPACollate`.

### Example 3: Add new logging around an existing feature

If you need extra run-time diagnostics:

- prefer computing them where the tensors already exist,
- add them to `out["mask_stats"]` or `out["model_stats"]`,
- let `train_loop.py` forward them into callbacks,
- and let `WandbCallback` handle histogram conversion when needed.

Do not add direct W&B calls inside core model code.

## Validation Before Opening a PR

At minimum:

1. Make sure the config composes cleanly for the path you changed.
2. Run `python -m compileall src/ijepa_lite`.
3. If you changed config wiring, sanity-check the target entrypoint you touched, usually `python -m ijepa_lite.run ...`.
4. If you changed DDP, callbacks, checkpointing, or warmup logic, reason through rank-0 behavior and resume behavior explicitly.
5. If you added a launcher, verify that its overrides stay minimal and comparable to the intended baseline.

## Final Checklist

Before you consider a contribution done, ask:

- Does this change live in the narrowest possible module?
- Is the feature selectable from config instead of hard-coded?
- Are the metrics clear about objective vs monitoring?
- Is the behavior safe under DDP and resume?
- Would a new contributor know how to run the variant from config plus sbatch?

If the answer to any of those is no, the change probably needs one more pass.
