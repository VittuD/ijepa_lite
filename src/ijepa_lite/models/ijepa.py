from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.models.ema import ema_update


class IJEPAModel(nn.Module):
    """
    i-JEPA model.

    Mask generation has been moved out of this class and into the DataLoader
    collate function (``IJEPACollate``).  Masks must therefore always be
    supplied via the ``masks`` argument to ``forward``; passing ``None`` is
    still accepted as a convenience (e.g. unit tests) but will raise if no
    ``mask_generator`` was provided at construction time.

    Learnable masking (optional)
    ----------------------------
    When ``mask_scorer`` is provided (a ``LinearSTEScorer`` or any future
    variant), the target patch positions are decided at runtime from the EMA
    output rather than from the pre-computed ``masks["target_idx"]``.

    When learnable masking is active, ``loss_fn`` must support
    ``reduction="none"`` (i.e. be a ``PerTokenLoss`` instance).  The standard
    ``VanillaTokenLoss`` is still used unchanged for the non-learnable path.

    Specifically, for every forward pass the scorer:
      1. Runs the EMA encoder on the full image (would happen anyway).
      2. Scores each of the N patches with a learned weight vector W.
      3. Selects the top-k patches as targets (k taken from the collate mask
         shape so the masking ratio is still controlled by config).

    Two losses are then computed and summed:

      L_jepa  — standard i-JEPA token prediction loss (trains context encoder
                 + predictor, W is detached via e.detach() below).
      L_W     — scorer loss: −mean(mask_ste_selected · e.detach())
                 Rewards W for assigning high scores to high-error patches.
                 The negative sign is intentional: W maximises prediction
                 difficulty, the predictor minimises it.

    ``lm_loss_weight`` (λ) scales L_W relative to L_jepa.  Start with a small
    value (e.g. 0.1) and increase if the scorer converges too slowly.

    The multiblock forward path is left entirely unchanged — learnable masking
    only applies when tgt_idx is 2-D (single-block case).
    """

    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
        loss_fn: nn.Module,
        ema_momentum: float,
        # kept for backward-compat / unit-test convenience; not used in
        # normal training where masks arrive from the DataLoader.
        mask_generator=None,
        # learnable masking (optional)
        mask_scorer: Optional[nn.Module] = None,
        lm_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor
        self.loss_fn = loss_fn
        self.ema_momentum = float(ema_momentum)
        self._mask_generator = mask_generator  # fallback only

        self.mask_scorer = mask_scorer
        self.lm_loss_weight = float(lm_loss_weight)

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_target(self) -> None:
        ema_update(self.target_encoder, self.context_encoder, self.ema_momentum)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ema_forward(self, images: torch.Tensor) -> torch.Tensor:
        """Full-image EMA forward + layer norm. Always no-grad."""
        with torch.no_grad():
            tgt = self.target_encoder(images)              # (B, N, F)
            tgt = F.layer_norm(tgt, (tgt.shape[-1],))
        return tgt

    def _single_block_forward(
        self,
        images: torch.Tensor,
        ctx_idx: torch.Tensor,          # (B, Nctx)
        tgt_idx: torch.Tensor,          # (B, k)
        tgt_tokens_all: torch.Tensor,   # (B, N, F)  pre-computed EMA output
        mask_ste: Optional[torch.Tensor] = None,  # (B, N) — only with scorer
    ) -> Dict:
        # ctx_tokens is returned so the outer forward can reuse it for
        # encoder_agreement without running a second full forward pass.
        d = tgt_tokens_all.shape[-1]

        # Context encoder (masked)
        ctx_tokens = self.context_encoder(images, keep_idx=ctx_idx)  # (B, Nctx, F)

        # Gather target tokens at selected positions
        tgt_tokens = tgt_tokens_all.gather(
            1, tgt_idx.unsqueeze(-1).expand(-1, -1, d)
        )  # (B, k, F)

        # Predictor
        pred = self.predictor(
            ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx
        )  # (B, k, F)

        # ------------------------------------------------------------------
        # Losses
        # ------------------------------------------------------------------
        if mask_ste is not None:
            # Per-token errors: (B, k)  — mean over feature dim F
            e = self.loss_fn(pred, tgt_tokens, reduction="none")  # (B, k)

            L_jepa = e.mean()

            # mask_ste at the selected positions is 1.0 in the forward pass,
            # but carries d(scores)/d(W) in the backward pass — that's the STE.
            s_selected = mask_ste.gather(1, tgt_idx)              # (B, k)
            L_W = -(s_selected * e.detach()).mean()

            loss = L_jepa + self.lm_loss_weight * L_W
        else:
            e = None
            loss = self.loss_fn(pred, tgt_tokens)
            L_jepa = loss
            L_W = None

        return {
            "loss": loss,
            "pred": pred.detach(),
            "target": tgt_tokens.detach(),
            "ctx_tokens": ctx_tokens.detach(),          # (B, Nctx, F) for agreement metric
            "L_jepa": L_jepa.detach(),
            "L_W": L_W.detach() if L_W is not None else None,
            # Per-token errors returned so the outer forward can compute
            # score-error correlation without a second predictor pass.
            "per_token_error": e.detach() if e is not None else None,  # (B, k) or None
        }

    def forward(
        self,
        images: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        compute_agreement: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # ------------------------------------------------------------------
        # Masks come pre-computed from IJEPACollate (CPU, DataLoader workers).
        # Fallback to on-device generation only for unit tests / debugging.
        # ------------------------------------------------------------------
        if masks is None:
            if self._mask_generator is None:
                raise ValueError(
                    "masks=None but no mask_generator was provided to IJEPAModel. "
                    "Either pass pre-computed masks or supply a mask_generator."
                )
            masks = self._mask_generator(
                batch_size=images.shape[0], device=images.device
            )

        ctx_idx = masks["context_idx"].to(images.device, non_blocking=True)  # (B, Nctx)
        tgt_idx = masks["target_idx"].to(
            images.device, non_blocking=True
        )  # (B, k) or (B, M, K)

        # ------------------------------------------------------------------
        # EMA forward — always needed, and done once regardless of path
        # ------------------------------------------------------------------
        tgt_tokens_all = self._ema_forward(images)   # (B, N, F)
        b = images.shape[0]
        d = tgt_tokens_all.shape[-1]

        # ------------------------------------------------------------------
        # Learnable mask scoring (single-block path only)
        # ------------------------------------------------------------------
        mask_ste: Optional[torch.Tensor] = None
        if self.mask_scorer is not None and tgt_idx.dim() == 2:
            k = tgt_idx.shape[1]
            # tgt_tokens_all has no grad (computed under no_grad), but W
            # receives its gradient through mask_ste via the STE trick.
            tgt_idx, mask_ste, scorer_stats = self.mask_scorer(tgt_tokens_all, k)

        # ------------------------------------------------------------------
        # Mask stats
        # ------------------------------------------------------------------
        mask_stats: Dict[str, float] = {"mask/nctx": float(ctx_idx.shape[1])}
        if tgt_idx.dim() == 2:
            mask_stats["mask/ntgt"] = float(tgt_idx.shape[1])
            mask_stats["mask/nblocks"] = 1.0
        elif tgt_idx.dim() == 3:
            m, k_ = tgt_idx.shape[1], tgt_idx.shape[2]
            mask_stats["mask/ntgt_per_block"] = float(k_)
            mask_stats["mask/ntgt_total"] = float(m * k_)
            mask_stats["mask/nblocks"] = float(m)
        # Merge scorer diagnostics into mask_stats when scorer is active.
        # _scores is a tensor used later for correlation; exclude it from the
        # scalar stats dict so it doesn't reach the logger.
        if self.mask_scorer is not None and tgt_idx.dim() == 2:
            mask_stats.update({k: v for k, v in scorer_stats.items() if not k.startswith("_")})

        # ------------------------------------------------------------------
        # Optional encoder-agreement diagnostic
        # ------------------------------------------------------------------
        # NOTE: populated after _single_block_forward so we can reuse
        # ctx_tokens from the result rather than running the encoder again.

        # ------------------------------------------------------------------
        # Single block: tgt_idx (B, k)
        # ------------------------------------------------------------------
        if tgt_idx.dim() == 2:
            result = self._single_block_forward(
                images, ctx_idx, tgt_idx, tgt_tokens_all, mask_ste
            )
            out: Dict = {
                "loss": result["loss"],
                "pred": result["pred"],
                "target": result["target"],
                "mask_stats": mask_stats,
            }
            if result["L_W"] is not None:
                out["lm_loss"] = float(result["L_W"].item())
                out["jepa_loss"] = float(result["L_jepa"].item())
                # target_idx carried through for heatmap accumulation in train_loop
                out["target_idx"] = tgt_idx.detach()

                # ----------------------------------------------------------
                # Diagnostic 1 — score-error correlation
                # Tests whether W has learned to assign high scores to patches
                # that are genuinely hard to predict.
                #   > 0 : W is finding harder patches (intended behaviour)
                #   ≈ 0 : W is unaligned with difficulty
                #   < 0 : W is actively selecting easy patches (bad)
                # ----------------------------------------------------------
                with torch.no_grad():
                    scores_sel = scorer_stats["_scores"].gather(1, tgt_idx)  # (B, k)
                    e_flat = result["per_token_error"].float().reshape(-1)
                    s_flat = scores_sel.float().reshape(-1)
                    s_mu, e_mu = s_flat.mean(), e_flat.mean()
                    cov = ((s_flat - s_mu) * (e_flat - e_mu)).mean()
                    s_std = s_flat.std().clamp(min=1e-8)
                    e_std = e_flat.std().clamp(min=1e-8)
                    score_error_corr = cov / (s_std * e_std)
                mask_stats["mask/score_error_corr"] = float(score_error_corr.item())

                # ----------------------------------------------------------
                # Diagnostic 2 — selection diversity (Jaccard)
                # Measures whether selections vary per image (content-driven)
                # or are nearly identical across images (fixed spatial prior).
                # Computed on a subsample of 32 images to keep it cheap.
                # Baseline (random masking) ≈ 1 - k/N.
                # If this drops well below baseline, W has collapsed to a
                # fixed spatial prior regardless of image content.
                # ----------------------------------------------------------
                with torch.no_grad():
                    B_full = tgt_idx.shape[0]
                    k_val  = tgt_idx.shape[1]
                    N_val  = tgt_tokens_all.shape[1]
                    subsample = min(32, B_full)
                    idx_sub = tgt_idx[:subsample]              # (S, k)
                    # Build binary selection masks: (S, N)
                    sel_masks = torch.zeros(
                        subsample, N_val, device=tgt_idx.device
                    ).scatter_(1, idx_sub, 1.0)
                    # Pairwise Jaccard: intersection / union for each pair
                    inter = sel_masks @ sel_masks.t()          # (S, S)
                    union = k_val + k_val - inter              # |A|+|B|-|A∩B|
                    jaccard_sim = inter / union.clamp(min=1)
                    # Mean off-diagonal = mean pairwise similarity
                    mask_diag = torch.eye(
                        subsample, device=tgt_idx.device, dtype=torch.bool
                    )
                    mean_sim = jaccard_sim[~mask_diag].mean()
                    diversity = 1.0 - mean_sim
                    random_baseline = 1.0 - k_val / N_val
                mask_stats["mask/selection_diversity"] = float(diversity.item())
                mask_stats["mask/diversity_baseline"]  = float(random_baseline)
            if compute_agreement:
                # ctx_tokens: (B, Nctx, F) — masked context encoder output
                # tgt_at_ctx: (B, Nctx, F) — EMA output at those same positions
                # Both are Nctx-sized so encoder_agreement can do element-wise ops.
                out["ctx_tokens_all"] = result["ctx_tokens"]
                out["tgt_tokens_all"] = tgt_tokens_all.gather(
                    1, ctx_idx.unsqueeze(-1).expand(-1, -1, d)
                ).detach()
            return out

        # ------------------------------------------------------------------
        # Multiblock: tgt_idx (B, M, K) — learnable masking not applied here
        # ------------------------------------------------------------------
        if tgt_idx.dim() == 3:
            m = tgt_idx.shape[1]
            k = tgt_idx.shape[2]
            tgt_idx_cat = tgt_idx.reshape(b, m * k)  # (B, M*K)

            tgt_tokens_cat = tgt_tokens_all.gather(
                1, tgt_idx_cat.unsqueeze(-1).expand(-1, -1, d)
            )  # (B, M*K, F)
            tgt_tokens = tgt_tokens_cat.reshape(b, m, k, d)  # (B, M, K, F)

            ctx_tokens = self.context_encoder(images, keep_idx=ctx_idx)  # (B, Nctx, F)
            pred_cat = self.predictor(
                ctx_tokens, ctx_idx=ctx_idx, tgt_idx=tgt_idx_cat
            )  # (B, M*K, F)
            pred = pred_cat.reshape(b, m, k, d)  # (B, M, K, F)

            loss = self.loss_fn(pred, tgt_tokens)

            out: Dict = {
                "loss": loss,
                "pred": pred.detach(),
                "target": tgt_tokens.detach(),
                "mask_stats": mask_stats,
            }
            if compute_agreement:
                out["ctx_tokens_all"] = ctx_tokens.detach()
                out["tgt_tokens_all"] = tgt_tokens_all.gather(
                    1, ctx_idx.unsqueeze(-1).expand(-1, -1, d)
                ).detach()
            return out

        raise ValueError(
            f"Unsupported target_idx.dim()={tgt_idx.dim()}, expected 2 or 3."
        )