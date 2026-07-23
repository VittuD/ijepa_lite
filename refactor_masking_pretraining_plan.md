# Masking and Pretraining Refactor Plan

Status: implementation complete; remote validation pending

Stable baseline: `stable-pre-refactor-2026-07-22` (`0f27e61`)

## Objective

Prepare the masking and pretraining implementation for publication by making
configuration errors explicit, replacing semantic data hidden in
`MaskOutput.aux` with typed structures, and reducing `IJEPAModel.forward()` to
a stable entry point over a stateless pretraining runtime.

## Compatibility Invariants

- Keep `IJEPAModel.forward()` arguments and returned dictionary keys stable.
- Keep model ownership and checkpoint state-dict keys stable.
- Keep existing `mask/*` logging names stable.
- Keep deterministic, multiblock, Goldilocks, RD, three-way MI, and N-way MI
  masking paths supported.
- Do not change masking policies or objective equations during structural work.
- Never silently ignore an explicit latent-masker configuration field.
- Keep experimental diagnostics extensible without using them for objective
  inputs or partition semantics.

## Validation Constraint

`CLAUDE.md` forbids local Python imports, `pytest`, and training/evaluation
commands. Tests are added locally but must be executed through the supplied
debug SLURM launcher on the HPC environment.

## Scheduled Changes

### 1. Characterize Current Behavior

- [x] Add CPU-only public-interface tests for mask construction and model
  pretraining outputs.
- [x] Cover deterministic single-block, deterministic multiblock, and learned
  masking paths.
- [x] Assert loss composition, tensor shapes, gradient ownership, target-token
  caching, output keys, and checkpoint key stability.
- [x] Add a debug SLURM launcher that runs the focused test suite remotely.

### 2. Make Latent-Masker Configuration Strict

- [x] Separate automatically inferred constructor values from explicit Hydra
  configuration values.
- [x] Continue filtering unused inferred values.
- [x] Reject explicit fields unsupported by the selected masker constructor,
  naming both the masker and invalid fields in the error.
- [x] Add regression coverage for valid overrides and unsupported fields.

### 3. Retire the Unsupported Goldilocks EMA Surface

- [x] Remove runnable Hydra and SLURM artifacts that advertise the unimplemented
  `ema_signal` path.
- [x] Preserve the proposal as a non-runnable research note.
- [x] Keep implementation of the EMA teaching signal out of this refactor.

### 4. Introduce Typed Mask Semantics

- [x] Add `MaskPartition` for hard indices and per-block counts.
- [x] Add explicit assignment variants for two-way probabilities, three-way
  probabilities, N-way probabilities, and Goldilocks target scores.
- [x] Add typed objective-state payloads for RD and MI maskers.
- [x] Keep an open diagnostics mapping only for logging and visual inspection.
- [x] Migrate collate, deterministic maskers, learned maskers, diagnostics, and
  visualization consumers.
- [x] Remove `MaskOutput.aux` after all semantic consumers are migrated.

### 5. Extract the Stateless Pretraining Runtime

- [x] Add typed pretraining request/result structures.
- [x] Move forward-pass orchestration into a stateless runtime function.
- [x] Keep all `nn.Module` ownership on `IJEPAModel`.
- [x] Keep `IJEPAModel.forward()` as the DDP-facing delegation point.
- [x] Preserve the existing dictionary output through a compatibility adapter.

### 6. Verify and Publish the Refactor

- [x] Run static checks that do not import project Python code.
- [ ] Run the focused test launcher on the remote HPC environment.
- [ ] Compare emitted metric names and state-dict keys with the baseline.
- [ ] Commit in reviewable slices after remote validation.

## Explicitly Out of Scope

- Implementing `ema_signal` or running new scientific experiments.
- Changing Goldilocks, RD, MI, SIGReg, or context-loss equations.
- Reorganizing `build.py` outside latent-masker argument validation.
- Refactoring trainer lifecycle hooks.
- Renaming existing metrics or checkpoint parameters.

## Known Semantic Decision

Goldilocks documentation says `patch_loss` is detached before entering the
Goldilocks target computation, while the current model passes the live tensor.
That changes gradient behavior and therefore experimental semantics. This
refactor will characterize and expose the behavior but will not silently change
it; any correction must be a separately labelled scientific change.
