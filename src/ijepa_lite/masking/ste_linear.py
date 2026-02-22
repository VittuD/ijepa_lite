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

        scores = z @ W          # (B, N, 1) -> squeeze -> (B, N)

    The top-k scoring patches are selected as targets.  Because argmax/topk
    are non-differentiable, we build a soft binary mask via STE:

        hard_mask = (scores >= threshold).float()               # (B, N), 0/1
        mask_ste  = scores + (hard_mask - scores).detach()      # STE bridge

    Forward pass  : mask_ste == hard_mask  (exactly 0 or 1 per patch).
    Backward pass : grad flows through scores as if mask_ste were scores,
                    so dL/dW = dL/d(mask_ste) * d(scores)/dW.

    The caller is responsible for:
      - using ``target_idx`` to gather target tokens and run the predictor,
      - using ``mask_ste`` to form the W-specific loss  L_W (see ijepa.py).

    Notes
    -----
    * W is initialised near zero (std=0.02) so early scores are near-uniform
      and the selection is close to random at the start of training.
    * Ties at the threshold boundary can produce slightly more than k ones in
      hard_mask; this is benign in practice because W outputs continuous values
      and exact ties are vanishingly rare after the first few steps.
    * No bias is used: the score is a pure dot-product with the EMA features,
      which keeps the scorer shift-invariant (consistent with post-LN features).
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
        target_idx : (B, k)   long — indices of the k highest-scoring patches
        mask_ste   : (B, N)   float — STE mask (1 at target_idx, 0 elsewhere
                                      in the forward; score gradient in backward)
        stats      : dict     — scalar diagnostics (detached, no grad):

            mask/score_std     — mean over batch of per-image score std across
                                 patches. Near-zero means W has collapsed and all
                                 patches score alike; selection is effectively
                                 random.

            mask/score_entropy — mean normalised entropy of softmax(scores),
                                 in [0, 1]. 1.0 = perfectly uniform (random
                                 masking). 0.0 = always selects the same patch.
                                 Healthy training: starts near 1.0, decreases
                                 gradually as W learns to concentrate on
                                 informative regions.
        """
        B, N, _ = ema_out.shape

        # (B, N, F) @ (F, 1) -> (B, N, 1) -> (B, N)
        scores = self.W(ema_out).squeeze(-1)

        # ------------------------------------------------------------------ #
        # STE: build a hard binary mask but let gradients flow through scores #
        # ------------------------------------------------------------------ #
        # threshold: the k-th largest score per sample  shape (B, 1)
        topk_vals = torch.topk(scores, k, dim=1).values   # (B, k)
        threshold = topk_vals[:, -1:]                      # (B, 1)

        hard_mask = (scores >= threshold).float()          # (B, N)

        # Forward == hard_mask; backward == d(scores)/d(W)
        mask_ste = scores + (hard_mask - scores).detach()  # (B, N)

        # Indices of selected patches — sorted descending by score
        target_idx = torch.argsort(scores, dim=1, descending=True)[:, :k]  # (B, k)

        # ------------------------------------------------------------------ #
        # Scalar diagnostics (no grad)                                        #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            score_std = scores.std(dim=1).mean()

            probs = torch.softmax(scores.float(), dim=1)             # (B, N)
            log_n = torch.log(torch.tensor(float(N), device=scores.device))
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=1).mean() / log_n

        stats = {
            "mask/score_std":     float(score_std.item()),
            "mask/score_entropy": float(entropy.item()),
            # Raw scores kept for downstream correlation metrics (detached).
            # Not a loggable scalar — consumed by ijepa.py and then dropped.
            "_scores":            scores.detach(),
        }

        return target_idx, mask_ste, stats