from __future__ import annotations

import torch
import torch.nn as nn


class LinearSTEScorer(nn.Module):
    """
    Learnable patch scorer using a single linear projection + Straight-Through
    Estimator (STE) for gradient flow through the discrete top-k selection.

    Architecture
    ------------
    Given EMA output ``z`` of shape (B, N, F) (post layer-norm):

        scores = z @ W                          # (B, N)
        pos_mean = scores.mean(dim=0)           # (N,)  batch-level spatial prior
        scores_centered = scores - pos_mean     # (B, N)  residual content signal

    Selection is performed on ``scores_centered`` so the spatial prior is
    removed at every forward pass.  W can therefore only move the selection
    by learning directions in feature space that vary *within* positions across
    images, i.e. content-driven variation.

    The STE bridge is built on the centered scores so gradients flow through
    the quantity that actually drives selection.

    Notes on the prior subtraction
    --------------------------------
    * The mean is computed over the current batch (not a running average), so
      the centering is exact for the batch but noisy for small batch sizes.
      With batch sizes >= 128 this is stable in practice.
    * During eval / single-image inference the batch mean collapses to the
      image's own scores and centering has no effect — this path is only
      meaningful during training where B > 1.
    * W is initialised near zero (std=0.02) so early scores are near-uniform
      and the selection is close to random at the start of training.
    * No bias: shift-invariant projection consistent with post-LN features.
    """

    def __init__(self, feat_dim: int):
        super().__init__()
        # W: (F, 1) — one scalar score per patch
        self.W = nn.Linear(feat_dim, 1, bias=False)
        nn.init.trunc_normal_(self.W.weight, std=0.02)

    def forward(
        self,
        ema_out: torch.Tensor,   # (B, N, F)  post layer-norm EMA features
        k: int,                  # number of patches to select as targets
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns
        -------
        target_idx : (B, k)   long — indices of the k highest centered-scoring patches
        mask_ste   : (B, N)   float — STE mask built on centered scores
        stats      : dict     — scalar diagnostics (all detached):

          Existing scalars (now computed on centered scores):
            mask/score_std     — per-image std of centered scores, mean over batch.
                                 Measures how sharply W differentiates patches
                                 *after* removing the spatial prior.
            mask/score_entropy — normalised entropy of softmax(centered scores),
                                 in [0, 1]. Same interpretation as before but
                                 now reflects content-driven concentration.

          New scalars (decompose raw scores into prior + residual):
            mask/prior_std     — std of pos_mean across positions (scalar).
                                 How strong the learned spatial prior is.
                                 Rising = W is still encoding positional bias.

            mask/residual_std  — mean over batch of per-image std of
                                 (scores - pos_mean). How much content-driven
                                 variation W produces after removing the prior.
                                 Should grow relative to prior_std over training.

            mask/prior_fraction — prior_std / (prior_std + residual_std).
                                  0 = fully content-driven, 1 = fully positional.
                                  Previous run: ~high. Goal: push this down.

          Private (tensor, not logged):
            _scores_centered  — centered scores for downstream correlation.
        """
        B, N, _ = ema_out.shape

        # ------------------------------------------------------------------ #
        # Raw scores then batch-level prior subtraction                       #
        # ------------------------------------------------------------------ #
        scores = self.W(ema_out).squeeze(-1)          # (B, N)

        # Per-position mean across the batch — the spatial prior W has learned.
        # Detached so the centering only affects which patches get selected
        # (forward pass), not the gradient flowing back to W (backward pass).
        # Without detach, the batch-mean subtraction weakens the L_W gradient
        # by cancelling consistent cross-image signal, causing W to grow large
        # weights chasing noisy residuals instead of learning difficulty.
        pos_mean = scores.mean(dim=0, keepdim=True).detach()   # (1, N)

        # Residual: what's left after removing the consistent positional signal.
        # No per-image std normalisation here — that approach distorts gradients
        # via a non-trivial Jacobian that can flip gradient directions and wash
        # out selection signal (entropy stuck near 1.0).  Magnitude is controlled
        # instead by weight decay on W in the optimiser (scorer param group).
        scores_centered = scores - pos_mean                     # (B, N)

        # ------------------------------------------------------------------ #
        # STE on centered scores                                              #
        # ------------------------------------------------------------------ #
        # Use topk for both indices and mask so they are consistent by
        # construction.  A threshold-based hard_mask with a separate argsort
        # can diverge when multiple patches share the k-th score (ties),
        # causing mask_ste to carry gradient for patches the predictor never
        # saw.  Scattering ones at exactly topk.indices avoids this entirely.
        topk_result = torch.topk(scores_centered, k, dim=1)
        target_idx  = topk_result.indices                          # (B, k)

        hard_mask = torch.zeros_like(scores_centered).scatter_(
            1, target_idx, 1.0
        )                                                          # (B, N)

        # Forward == hard_mask; backward == d(scores_centered)/d(W).
        # pos_mean is detached so d(scores_centered)/d(W) = d(scores)/d(W),
        # i.e. the full uncentered gradient flows to W — centering only affects
        # which patches are selected, not what signal W is trained on.
        mask_ste = scores_centered + (hard_mask - scores_centered).detach()

        # ------------------------------------------------------------------ #
        # Diagnostics                                                         #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            # --- On centered scores (what actually drives selection) ---
            score_std = scores_centered.std(dim=1).mean()          # scalar

            probs = torch.softmax(scores_centered.float(), dim=1)  # (B, N)
            log_n = torch.log(torch.tensor(float(N), device=scores.device))
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean() / log_n

            # --- Prior vs residual decomposition ---
            # prior_std: std of pos_mean across patch positions — how much
            # spatial structure is still in W's raw outputs (informational
            # only; pos_mean is detached so this no longer drives gradients).
            prior_std    = pos_mean.squeeze(0).std()               # scalar

            # residual_std: per-image std of the z-scored centered scores.
            # After normalisation this is ~1 by construction; if it deviates
            # significantly something is wrong with the normalisation.
            residual_std = scores_centered.std(dim=1).mean()       # scalar

            # prior_fraction still meaningful as a monitoring signal:
            # compares raw prior magnitude to raw (pre-normalised) residual.
            raw_residual_std = (scores - pos_mean).std(dim=1).mean()
            prior_fraction = prior_std / (prior_std + raw_residual_std + 1e-8)

        stats = {
            "mask/score_std":       float(score_std.item()),
            "mask/score_entropy":   float(entropy.item()),
            "mask/prior_std":       float(prior_std.item()),
            # Raw (pre-normalised) residual std — meaningful magnitude comparison
            # against prior_std. After z-scoring, residual_std is ~1 always.
            "mask/residual_std":    float(raw_residual_std.item()),
            "mask/prior_fraction":  float(prior_fraction.item()),
            # Centered+normalised scores for downstream correlation — not logged
            "_scores_centered":     scores_centered.detach(),
        }

        return target_idx, mask_ste, stats