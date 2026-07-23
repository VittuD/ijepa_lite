# Goldilocks EMA Teaching Signal Proposal

Status: proposed, not implemented

The repository previously exposed Hydra configurations and SLURM launchers for
an `ema_signal` Goldilocks variant. `GoldilocksTeacherMasker` never accepted or
implemented that field, while the latent-masker registry silently discarded it.
Those jobs therefore ran the ordinary Goldilocks method under an EMA-specific
experiment name.

The runnable artifacts were removed before publication. The proposal is kept
here so that a future implementation starts from an explicit method rather than
reviving the misleading configuration surface.

## Intended Method

The proposed variant would derive the per-patch Goldilocks teaching signal from
EMA encoder features with stop-gradient on both sides. This is intended to
separate the difficulty target from the online context encoder and reduce the
co-adaptation loop between target selection and reconstruction error.

The old launchers anticipated:

- an additional predictor pass used to compute EMA-based patch losses;
- an `ema_patch_loss` tensor with the same shape as the online `patch_loss`;
- no gradient from the EMA difficulty target into either encoder;
- comparison against ordinary Goldilocks under identical context and target
  budgets;
- monitoring `mask/marginal_score_std`, `mask/batch_iou`,
  `mask/tgt_pos_std_norm`, and `mask/masker_loss`.

## Decisions Required Before Implementation

1. Define exactly which parameters produce the EMA-side prediction.
2. Define whether the predictor itself is copied by EMA or shared.
3. Define the stop-gradient boundary in the objective equation.
4. Decide whether online reconstruction remains part of the full objective or
   is used only for monitoring.
5. Establish a fixed-budget diagnostic protocol before full pretraining.

Only after those decisions are documented should `ema_signal` return as a
constructor argument, Hydra option, or runnable SLURM experiment.
