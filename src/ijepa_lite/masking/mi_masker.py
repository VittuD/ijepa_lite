"""
MIRateMasker: learned masker with modular compositional loss.

Architecture
------------
  proj_in → [compressed_ctx | selection_queries] → TransformerEncoder → proj_score

The loss is a weighted sum of independently toggleable atomic terms
(see ``losses/terms.py`` and ``losses/composite.py``). Per-step term
weights are sampled from LogUniform distributions with optional warmup.

No (λ, α) conditioning — the transformer sees only positional embeddings
and compressed context tokens, keeping the masker architecture agnostic
to the loss weighting.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ijepa_lite.losses.composite import CompositeMaskerLoss
from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register


@register("mi_3way")
class MIRateMasker(LatentMasker):
    """
    MI-rate 3-way masker with modular compositional loss.

    Args
    ----
    dim            : Encoder embedding dim.
    predictor_dim  : Internal transformer dim (matches Predictor).
    depth          : Transformer layers.
    num_heads      : Attention heads.
    mlp_ratio      : FFN expansion.
    dropout        : Dropout.
    num_patches    : N — total patch positions.
    terms          : Nested dict of term configs (see CompositeMaskerLoss).
    ntgt_min       : Hard floor on target count.
    nctx_min       : Hard floor on context count.
    hard_assignment: "topk", "argmax", or "gumbel" for hard reconstruction masks.
    gumbel_tau     : Temperature for Gumbel hard assignment.
    warmup_epochs  : Epochs to grow weight sampling range to full.
    """

    owns_loss: bool = True
    needs_full_tokens: bool = True

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        terms: dict,
        ntgt_min: int = 4,
        nctx_min: int = 1,
        hard_assignment: str = "topk",
        gumbel_tau: float = 1.0,
        warmup_epochs: int = 0,
        pos_embed_kind: str = "learned",
        # Unused — kept for build.py kwarg filtering
        base_kind: str = "smooth_l1",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.ntgt_min = max(1, int(ntgt_min))
        self.nctx_min = max(1, int(nctx_min))
        if hard_assignment not in ("topk", "argmax", "gumbel"):
            raise ValueError(
                "hard_assignment must be 'topk', 'argmax', or 'gumbel', "
                f"got {hard_assignment!r}"
            )
        self.hard_assignment = str(hard_assignment)
        self.gumbel_tau = float(gumbel_tau)
        self.warmup_epochs = int(warmup_epochs)

        # _progress in [0, 1]; initialised to 1.0 so unit tests use full range.
        self.register_buffer("_progress", torch.tensor(1.0), persistent=False)

        d = predictor_dim

        from ijepa_lite.models.pos_embed import build_pos_embed_2d

        grid_size = int(math.isqrt(num_patches))

        # ----------------------------------------------------------------
        # Transformer backbone
        # ----------------------------------------------------------------
        self.proj_in = nn.Linear(dim, d)

        self.pos_embed = build_pos_embed_2d(pos_embed_kind, grid_size, d)
        self.selection_token = nn.Parameter(torch.zeros(1, 1, d))

        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=num_heads,
            dim_feedforward=int(d * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d)

        # 3-way score head: d → [l_ctx, l_tgt, l_ign]
        self.proj_score = nn.Linear(d, 3)

        # Composite loss
        self.composite_loss = CompositeMaskerLoss(terms, num_patches)

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

    # ------------------------------------------------------------------
    # Warmup progress
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialise all weights. Called by train loop on masker reset trigger."""
        for m in self.modules():
            if m is self:
                continue
            reset_fn = getattr(m, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        if isinstance(self.pos_embed, nn.Parameter):
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

    def set_progress(self, fraction: float) -> None:
        """Update warmup progress. Call once per epoch."""
        self._progress.fill_(max(0.0, min(1.0, float(fraction))))

    # ------------------------------------------------------------------
    # Weight sampling
    # ------------------------------------------------------------------

    def _sample_weights(self, device: torch.device) -> dict[str, float]:
        """Sample per-term weights from LogUniform with warmup."""
        p = max(self._progress.item(), 1e-3)
        weights: dict[str, float] = {}

        for name, (lo, hi) in self.composite_loss.weight_ranges.items():
            if lo == hi:
                weights[name] = lo
            else:
                # Temperature-biased LogUniform: u^(1/p) concentrates near lo
                u = torch.rand(1, device=device).item() ** (1.0 / p)
                val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                weights[name] = val

        return weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,                     # (B, M, D)
        ema_full: Optional[torch.Tensor] = None,   # (B, N, D)
        rates: Optional[torch.Tensor] = None,      # unused, kept for interface compat
    ) -> MaskOutput:
        B = tokens.shape[0]

        # ----------------------------------------------------------------
        # Transformer input: compressed tokens
        # ----------------------------------------------------------------
        ctx = self.proj_in(tokens)   # (B, M, d)

        # ----------------------------------------------------------------
        # Selection queries + transformer (no rates conditioning)
        # ----------------------------------------------------------------
        queries = self.selection_token.expand(B, self.num_patches, -1) \
                  + self.pos_embed.expand(B, -1, -1)
        seq = torch.cat([ctx, queries], dim=1)   # (B, M+N, d)
        out = self.norm(self.blocks(seq))
        query_out = out[:, -self.num_patches:]   # (B, N, d)

        # ----------------------------------------------------------------
        # 3-way soft assignments
        # ----------------------------------------------------------------
        logits = self.proj_score(query_out)   # (B, N, 3)
        soft = F.softmax(logits, dim=-1)      # (B, N, 3)

        p_ctx = soft[..., 0]
        p_tgt = soft[..., 1]
        p_ign = soft[..., 2]

        # ----------------------------------------------------------------
        # Hard reconstruction masks
        # ----------------------------------------------------------------
        aux_counts = {}
        if self.hard_assignment in ("argmax", "gumbel"):
            if self.hard_assignment == "gumbel":
                # Gumbel-max samples one hard role per patch from the categorical.
                g = -torch.empty_like(logits).exponential_().log()
                winners = (logits / self.gumbel_tau + g).argmax(dim=-1)
            else:
                winners = soft.argmax(dim=-1)

            tgt_scores = (winners == 1).float()
            tgt_counts = tgt_scores.sum(dim=-1)
            needs_tgt_fallback = tgt_counts < self.ntgt_min
            if needs_tgt_fallback.any():
                tgt_scores[needs_tgt_fallback] = p_tgt[needs_tgt_fallback]

            ntgt = max(self.ntgt_min, int(tgt_counts.max().item()))
            # Scores are binary except fallback rows; topk avoids sorting all
            # patches while still selecting winners before non-winner fillers.
            _, tgt_idx = torch.topk(tgt_scores, ntgt, dim=-1, sorted=False)

            ctx_scores = (winners == 0).float()
            ctx_counts = ctx_scores.sum(dim=-1)
            needs_ctx_fallback = ctx_counts < self.nctx_min
            if needs_ctx_fallback.any():
                p_ctx_fb = p_ctx[needs_ctx_fallback].clone()
                p_ctx_fb.scatter_(1, tgt_idx[needs_ctx_fallback], 0.0)
                ctx_scores[needs_ctx_fallback] = p_ctx_fb

            nctx = max(self.nctx_min, int(ctx_counts.max().item()))
            _, ctx_idx = torch.topk(ctx_scores, nctx, dim=-1, sorted=False)

            aux_counts = {
                "hard_sampled_nctx": float(ctx_counts.float().mean().detach().item()),
                "hard_sampled_ntgt": float(tgt_counts.float().mean().detach().item()),
                "hard_sampled_nign": float((winners == 2).float().sum(dim=-1).mean().detach().item()),
            }
        else:
            ntgt = max(self.ntgt_min, int(round(p_tgt.sum(dim=-1).mean().item())))
            nctx = max(self.nctx_min, int(round(p_ctx.sum(dim=-1).mean().item())))

            _, tgt_idx = torch.topk(p_tgt, ntgt, dim=-1, sorted=False)
            p_ctx_masked = p_ctx.clone().scatter_(1, tgt_idx, 0.0)
            _, ctx_idx = torch.topk(p_ctx_masked, nctx, dim=-1, sorted=False)

        # Sample weights for this step
        weights = self._sample_weights(tokens.device)

        return MaskOutput(
            context_idx=ctx_idx,
            target_idx=tgt_idx,
            context_soft=p_ctx,
            target_soft=p_tgt,
            aux={
                "weights":  weights,
                "p_ign":    p_ign,
                "logits":   logits.detach(),
                "ema_full": ema_full,
                "hard_assignment": self.hard_assignment,
                **aux_counts,
            },
        )

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p_ctx    = mask_output.context_soft
        p_tgt    = mask_output.target_soft
        p_ign    = mask_output.aux["p_ign"]
        weights  = mask_output.aux["weights"]
        ema_full = mask_output.aux.get("ema_full")

        if ema_full is None:
            # Unit test fallback — compute only entropy terms
            device = p_ctx.device
            ema_full = torch.zeros(
                p_ctx.shape[0], p_ctx.shape[1], 1, device=device,
            )

        total, logs = self.composite_loss(
            weights=weights,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
        )

        # Write logs into aux for metrics.py to pick up
        mask_output.aux.update(logs)

        return reconstruction_loss + total


# ======================================================================
# N-way MI masker (cross-target surprise, no ctx-vs-tgt terms)
# ======================================================================


@register("mi_nway")
class MINWayMasker(LatentMasker):
    """
    N-way MI masker with (M+2)-way categorical assignments.

    Assigns each patch to (ctx, tgt₁, ..., tgt_M, ign). Loss terms only
    push target blocks apart (inter-target surprise). Context relevance
    emerges naturally without adversarial ctx-vs-tgt distance terms.
    """

    owns_loss: bool = True
    needs_full_tokens: bool = True

    def __init__(
        self,
        dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        num_patches: int,
        terms: dict,
        num_tgt_blocks: int = 4,
        ntgt_min_per_block: int = 4,
        max_total_tgt: int = 0,
        max_tgt_per_block: int = 0,
        nctx_min: int = 1,
        hard_assignment: str = "topk",
        arch: str = "transformer",
        gumbel_tau: float = 1.0,
        warmup_epochs: int = 0,
        pos_embed_kind: str = "learned",
        # k schedule (context mass fraction for KL marginal term)
        k_start: float = 0.0,
        k_min: float = 0.0,
        k_max: float = 0.0,
        k_warmup_epochs: int = 10,
        total_epochs: int = 1000,
        # Progressive role unlocking
        n_start_tgt: int = 2,
        phase_mode: str = "none",       # "none" | "deterministic" | "adaptive"
        phase_fractions: list | None = None,  # deterministic: fractions at which to add a role
        adaptive_eps: float = 0.01,     # adaptive: KL below this = converged
        adaptive_patience_epochs: int = 3,    # adaptive: epochs below eps before advance
        # Unused — kept for build.py kwarg filtering
        base_kind: str = "smooth_l1",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.M = int(num_tgt_blocks)
        self.ntgt_min_per_block = max(1, int(ntgt_min_per_block))
        self.max_total_tgt = int(max_total_tgt)
        self.max_tgt_per_block = int(max_tgt_per_block)
        self.nctx_min = max(1, int(nctx_min))
        self.hard_assignment = str(hard_assignment)
        self.arch = str(arch)
        self.gumbel_tau = float(gumbel_tau)
        self.warmup_epochs = int(warmup_epochs)

        # k schedule state: k_start → k_max (warmup) → k_min (cosine)
        self.k_start = float(k_start)
        self.k_min = float(k_min)
        self.k_max = float(k_max)
        self.k_warmup_epochs = int(k_warmup_epochs)
        self.total_epochs = int(total_epochs)
        self._k_enabled = self.k_start > 0 or self.k_min > 0 or self.k_max > 0

        self.register_buffer("_progress", torch.tensor(1.0), persistent=False)
        self.register_buffer("_current_k", torch.tensor(self.k_start if self._k_enabled else 0.0), persistent=False)

        # ------------------------------------------------------------------
        # Progressive role unlocking
        # ------------------------------------------------------------------
        assert phase_mode in ("none", "deterministic", "adaptive"), \
            f"phase_mode must be 'none', 'deterministic', or 'adaptive', got {phase_mode!r}"
        self._phase_mode = str(phase_mode)
        self._n_start_tgt = max(2, min(int(n_start_tgt), self.M))  # ≥2 for cross-surprise
        self._adaptive_eps = float(adaptive_eps)
        self._adaptive_patience = int(adaptive_patience_epochs)
        self._adaptive_kl_history: list[float] = []

        # Deterministic phase fractions: where in [0,1] of total steps to unlock next role.
        if phase_fractions is not None:
            _fracs = sorted(float(f) for f in phase_fractions)
        else:
            # Auto-uniform: n_transitions evenly spaced, all within the training window.
            n_transitions = self.M - self._n_start_tgt
            if n_transitions > 0:
                _fracs = [(i + 1) / (n_transitions + 1) for i in range(n_transitions)]
            else:
                _fracs = []
        self._phase_fractions: list[float] = _fracs

        # Persistent buffer so current phase survives checkpoint/resume.
        self.register_buffer(
            "_n_active_buf",
            torch.tensor(self._n_start_tgt if self._phase_mode != "none" else self.M,
                         dtype=torch.long),
            persistent=True,
        )

        d = predictor_dim

        # ----------------------------------------------------------------
        # Scoring backbone — arch selects complexity
        # ----------------------------------------------------------------
        if self.arch == "transformer":
            from ijepa_lite.models.pos_embed import build_pos_embed_2d
            grid_size = int(math.isqrt(num_patches))

            self.proj_in = nn.Linear(dim, d)
            self.pos_embed = build_pos_embed_2d(pos_embed_kind, grid_size, d)
            self.selection_token = nn.Parameter(torch.zeros(1, 1, d))

            layer = nn.TransformerEncoderLayer(
                d_model=d,
                nhead=num_heads,
                dim_feedforward=int(d * mlp_ratio),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(
                layer, num_layers=depth, enable_nested_tensor=False
            )
            self.norm = nn.LayerNorm(d)
            self.proj_score = nn.Linear(d, self.M + 2)

            nn.init.trunc_normal_(self.selection_token, std=0.02)

        elif self.arch == "mlp":
            self.proj_in = nn.Linear(dim, d)
            self.proj_score = nn.Linear(d, self.M + 2)

        elif self.arch == "linear":
            self.proj_score = nn.Linear(dim, self.M + 2)

        else:
            raise ValueError(f"Unknown arch: {self.arch!r}. Choose from: transformer, mlp, linear")

        # (M+2)-way score head init (shared across all archs)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

        # k-conditioning: learned logit bias from scalar k
        if self._k_enabled:
            self.k_logit_bias = nn.Sequential(
                nn.Linear(1, 32), nn.GELU(), nn.Linear(32, self.M + 2),
            )
        else:
            self.k_logit_bias = None

        # Composite loss
        self.composite_loss = CompositeMaskerLoss(
            terms, num_patches, num_tgt_blocks=self.M
        )

    # ------------------------------------------------------------------
    # Warmup progress
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        """Re-initialise all weights. Called by train loop on masker reset trigger."""
        for m in self.modules():
            if m is self:
                continue
            reset_fn = getattr(m, "reset_parameters", None)
            if callable(reset_fn):
                reset_fn()
        if self.arch == "transformer":
            nn.init.trunc_normal_(self.selection_token, std=0.02)
            if isinstance(self.pos_embed, nn.Parameter):
                nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)
        # Reset phase state so the curriculum restarts from the beginning
        self._n_active_buf.fill_(
            self._n_start_tgt if self._phase_mode != "none" else self.M
        )
        self._adaptive_kl_history.clear()

    def set_progress(self, fraction: float) -> None:
        self._progress.fill_(max(0.0, min(1.0, float(fraction))))

    def set_step(self, step: int, total_steps: int) -> None:
        """Update k schedule and (for deterministic mode) advance the active-role phase."""
        self._global_step = int(step)

        # Auto-set transition_steps for the progressive KL term on first call,
        # using the same inter-phase gap formula as the deterministic fractions:
        #   gap = total_steps / (n_transitions + 1)
        if not getattr(self, "_pklt_ts_initialized", False):
            self._pklt_ts_initialized = True
            pklt = self.composite_loss.terms._modules.get("nway_progressive_kl")
            if pklt is not None and pklt.transition_steps < 0:
                n_transitions = self.M - self._n_start_tgt
                if n_transitions > 0:
                    pklt.transition_steps = max(1, total_steps // (n_transitions + 1))
                else:
                    pklt.transition_steps = max(1, total_steps)

        if self._k_enabled:
            warmup_steps = self.k_warmup_epochs * max(1, total_steps // max(1, self.total_epochs))
            if warmup_steps > 0 and step < warmup_steps:
                frac = float(step) / float(warmup_steps)
                k = self.k_start + (self.k_max - self.k_start) * frac
            else:
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                progress = min(progress, 1.0)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                k = self.k_min + (self.k_max - self.k_min) * cosine
            self._current_k.fill_(k)

        # Deterministic phase advance: unlock next tgt role when step crosses threshold.
        if self._phase_mode == "deterministic":
            n_active = int(self._n_active_buf.item())
            if n_active < self.M:
                phase_idx = n_active - self._n_start_tgt
                if phase_idx < len(self._phase_fractions):
                    threshold = int(self._phase_fractions[phase_idx] * total_steps)
                    if step >= threshold:
                        self._n_active_buf.fill_(min(n_active + 1, self.M))

    def on_epoch_kl(self, kl_val: float) -> bool:
        """
        Called at epoch end with the per-epoch average progressive KL loss.
        For adaptive phase mode: advances to the next role when the KL has
        been below adaptive_eps for adaptive_patience_epochs consecutive epochs.
        Returns True if a new role was unlocked.
        """
        if self._phase_mode != "adaptive":
            return False
        n_active = int(self._n_active_buf.item())
        if n_active >= self.M:
            return False

        self._adaptive_kl_history.append(float(kl_val))
        if len(self._adaptive_kl_history) >= self._adaptive_patience:
            recent = self._adaptive_kl_history[-self._adaptive_patience:]
            if all(v < self._adaptive_eps for v in recent):
                self._n_active_buf.fill_(min(n_active + 1, self.M))
                self._adaptive_kl_history.clear()
                return True
        return False

    # ------------------------------------------------------------------
    # Weight sampling
    # ------------------------------------------------------------------

    def _allocate_block_counts(
        self,
        signal: torch.Tensor,
    ) -> torch.Tensor | None:
        """
        Allocate a per-block target budget under a global sum cap and an
        optional per-block ceiling.

        Returns
        -------
        counts : (M,) LongTensor when max_total_tgt > 0, otherwise None.
        """
        if self.max_total_tgt <= 0:
            return None

        device = signal.device
        counts = torch.full(
            (self.M,), self.ntgt_min_per_block, device=device, dtype=torch.long
        )
        min_total = int(counts.sum().item())
        total_budget = max(min_total, self.max_total_tgt)
        extra_budget = total_budget - min_total
        if extra_budget <= 0:
            return counts

        weights = signal.float().clamp(min=0.0)
        if float(weights.sum().item()) <= 0.0:
            weights[0] = 1.0

        max_per_block = self.num_patches
        if self.max_tgt_per_block > 0:
            max_per_block = min(
                max_per_block,
                max(self.ntgt_min_per_block, self.max_tgt_per_block),
            )
        while extra_budget > 0:
            capacity = max_per_block - counts
            active = capacity > 0
            if not bool(active.any()):
                break

            active_weights = weights.clone()
            active_weights[~active] = 0.0
            if float(active_weights.sum().item()) <= 0.0:
                active_weights = active.float()

            raw_extra = active_weights * (float(extra_budget) / float(active_weights.sum().item()))
            extra = torch.floor(raw_extra).to(torch.long)
            extra = torch.minimum(extra, capacity)

            added = int(extra.sum().item())
            if added == 0:
                order = torch.argsort(active_weights, descending=True)
                for idx in order.tolist():
                    if capacity[idx] > 0:
                        counts[idx] += 1
                        extra_budget -= 1
                        break
                continue

            counts += extra
            extra_budget -= added

            if extra_budget <= 0:
                break

            frac = raw_extra - torch.floor(raw_extra)
            frac[capacity <= extra] = -1.0
            order = torch.argsort(frac, descending=True)
            for idx in order.tolist():
                if extra_budget <= 0:
                    break
                if counts[idx] < max_per_block:
                    counts[idx] += 1
                    extra_budget -= 1

        return counts

    def _sample_weights(self, device: torch.device) -> dict[str, float]:
        p = max(self._progress.item(), 1e-3)
        weights: dict[str, float] = {}

        for name, (lo, hi) in self.composite_loss.weight_ranges.items():
            if lo == hi:
                weights[name] = lo
            else:
                u = torch.rand(1, device=device).item() ** (1.0 / p)
                val = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))
                weights[name] = val

        return weights

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,
        ema_full: Optional[torch.Tensor] = None,
        rates: Optional[torch.Tensor] = None,
    ) -> MaskOutput:
        B = tokens.shape[0]
        M = self.M

        # ----------------------------------------------------------------
        # Scoring backbone
        # ----------------------------------------------------------------
        if self.arch == "transformer":
            ctx = self.proj_in(tokens)
            queries = self.selection_token.expand(B, self.num_patches, -1) \
                      + self.pos_embed.expand(B, -1, -1)
            seq = torch.cat([ctx, queries], dim=1)
            out = self.norm(self.blocks(seq))
            logits = self.proj_score(out[:, -self.num_patches:])
        elif self.arch == "mlp":
            logits = self.proj_score(F.gelu(self.proj_in(tokens)))
        else:  # linear
            logits = self.proj_score(tokens)

        # ----------------------------------------------------------------
        # k-conditioning: shift logits toward target allocation q(k)
        # ----------------------------------------------------------------
        if self.k_logit_bias is not None:
            k_val = self._current_k.item()
            k_tensor = torch.tensor(
                [[k_val]], device=logits.device, dtype=logits.dtype,
            )
            logits = logits + self.k_logit_bias(k_tensor).unsqueeze(1)

        # ----------------------------------------------------------------
        # Progressive phase masking: suppress inactive tgt role channels
        # before softmax so they receive zero probability mass.
        # ----------------------------------------------------------------
        n_active = int(self._n_active_buf.item())
        if n_active < self.M:
            logits = logits.clone()
            logits[:, :, 1 + n_active: 1 + self.M] = float("-inf")

        # ----------------------------------------------------------------
        # (M+2)-way soft assignments
        # ----------------------------------------------------------------
        soft = F.softmax(logits, dim=-1)      # (B, N, M+2)

        p_ctx  = soft[..., 0]                 # (B, N)
        p_tgts = soft[..., 1:M+1]            # (B, N, M)
        p_ign  = soft[..., -1]                # (B, N)

        # ----------------------------------------------------------------
        # Hard indices → (B, M, K)
        # ----------------------------------------------------------------
        if self.hard_assignment in ("argmax", "gumbel"):
            # Each patch goes to its winning role; no topk budget constraint.
            # Blocks may have variable sizes → pad to max for rectangular tensor.
            if self.hard_assignment == "gumbel":
                # Gumbel-max trick: sample from the categorical instead of argmax
                g = -torch.empty_like(logits).exponential_().log()  # Gumbel(0,1)
                winners = (logits / self.gumbel_tau + g).argmax(dim=-1)
            else:
                winners = soft.argmax(dim=-1)  # (B, N)  values in [0, M+1]

            # --- Vectorized target block indices ---
            # Build (B, N, M) score tensor: 1.0 where patch won block k, else 0.0.
            # argsort descending puts winners first; remaining slots fill with
            # non-winning patches in stable index order (replaces cycling pad).
            K_max_cap = self.num_patches // 2
            if self.max_tgt_per_block > 0:
                K_max_cap = min(
                    K_max_cap,
                    max(self.ntgt_min_per_block, self.max_tgt_per_block),
                )
            all_masks = (
                winners.unsqueeze(-1) == torch.arange(1, M + 1, device=winners.device)
            )  # (B, N, M)
            scores = all_masks.float()  # (B, N, M)

            counts = scores.sum(1)  # (B, M) — winner count per block per sample
            needs_fallback = counts < self.ntgt_min_per_block  # (B, M)
            if needs_fallback.any():
                # Replace scores with soft probabilities for under-populated blocks
                fb_b, fb_k = needs_fallback.nonzero(as_tuple=True)
                scores[fb_b, :, fb_k] = p_tgts[fb_b, :, fb_k]
            alloc_counts = self._allocate_block_counts(counts.float().mean(dim=0))
            if alloc_counts is None:
                K = max(self.ntgt_min_per_block, min(K_max_cap, int(counts.max().item())))
                target_block_counts = torch.full(
                    (M,), K, device=scores.device, dtype=torch.long
                )
            else:
                target_block_counts = alloc_counts
                K = int(target_block_counts.max().item())

            sorted_idx = scores.argsort(dim=1, descending=True)  # (B, N, M)
            tgt_idx_list = []
            tgt_flat_parts = []
            for i in range(M):
                k_i = int(target_block_counts[i].item())
                idx_i = sorted_idx[:, :k_i, i]  # (B, k_i)
                tgt_flat_parts.append(idx_i)
                if k_i < K:
                    pad_val = idx_i[:, -1:] if k_i > 0 else torch.zeros(
                        B, 1, device=idx_i.device, dtype=idx_i.dtype
                    )
                    idx_i = torch.cat([idx_i, pad_val.expand(-1, K - k_i)], dim=1)
                tgt_idx_list.append(idx_i)
            tgt_idx = torch.stack(tgt_idx_list, dim=1)  # (B, M, K)
            tgt_flat = torch.cat(tgt_flat_parts, dim=1)

            # --- Vectorized context indices ---
            ctx_scores = (winners == 0).float()  # (B, N)
            ctx_counts = ctx_scores.sum(-1)       # (B,)
            needs_ctx_fallback = ctx_counts < self.nctx_min
            if needs_ctx_fallback.any():
                # For fallback samples: use p_ctx with target positions zeroed out
                p_ctx_fb = p_ctx[needs_ctx_fallback].clone()              # (n_fb, N)
                p_ctx_fb.scatter_(1, tgt_flat[needs_ctx_fallback], 0.0)
                ctx_scores[needs_ctx_fallback] = p_ctx_fb

            nctx = max(self.nctx_min, min(K_max_cap, int(ctx_counts.max().item())))
            ctx_idx = ctx_scores.argsort(dim=-1, descending=True)[:, :nctx]  # (B, nctx)
        else:
            # topk (default): fixed K per block from soft mass
            per_block_mass = p_tgts.sum(dim=1).mean(dim=0)  # (M,)
            alloc_counts = self._allocate_block_counts(per_block_mass)
            if alloc_counts is None:
                K = max(self.ntgt_min_per_block, int(round(per_block_mass.max().item())))
                if self.max_tgt_per_block > 0:
                    K = min(K, max(self.ntgt_min_per_block, self.max_tgt_per_block))
                target_block_counts = torch.full(
                    (M,), K, device=p_tgts.device, dtype=torch.long
                )
            else:
                target_block_counts = alloc_counts
                K = int(target_block_counts.max().item())

            tgt_idx_list = []
            tgt_flat_parts = []
            for k in range(M):
                k_i = int(target_block_counts[k].item())
                _, idx_k = torch.topk(p_tgts[..., k], k_i, dim=-1, sorted=False)
                tgt_flat_parts.append(idx_k)
                if k_i < K:
                    pad_val = idx_k[:, -1:] if k_i > 0 else torch.zeros(
                        B, 1, device=idx_k.device, dtype=idx_k.dtype
                    )
                    idx_k = torch.cat([idx_k, pad_val.expand(-1, K - k_i)], dim=1)
                tgt_idx_list.append(idx_k)
            tgt_idx = torch.stack(tgt_idx_list, dim=1)  # (B, M, K)
            tgt_flat = torch.cat(tgt_flat_parts, dim=1)

            # Context: topk on p_ctx after zeroing all target positions
            nctx = max(self.nctx_min, int(round(p_ctx.sum(dim=-1).mean().item())))
            p_ctx_masked = p_ctx.clone().scatter_(1, tgt_flat, 0.0)
            _, ctx_idx = torch.topk(p_ctx_masked, nctx, dim=-1, sorted=False)

        # Sample weights for this step
        weights = self._sample_weights(tokens.device)

        # target_soft = sum across blocks for backward compat metrics
        aux = {
            "weights":      weights,
            "p_ign":        p_ign,
            "soft":         soft,
            "logits":       logits.detach(),
            "ema_full":     ema_full,
            "n_active_tgt": n_active,
            "global_step":  getattr(self, "_global_step", 0),
            "target_block_counts": target_block_counts,
            "max_total_tgt": self.max_total_tgt,
            "max_tgt_per_block": self.max_tgt_per_block,
        }
        if self._k_enabled:
            aux["k"] = self._current_k.item()

        return MaskOutput(
            context_idx=ctx_idx,       # (B, Nctx)
            target_idx=tgt_idx,        # (B, M, K)
            context_soft=p_ctx,        # (B, N)
            target_soft=p_tgts.sum(-1),  # (B, N) — total target mass
            aux=aux,
        )

    # ------------------------------------------------------------------
    # Auxiliary loss
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        mask_output: MaskOutput,
        reconstruction_loss: torch.Tensor,
        patch_loss: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        p_ctx    = mask_output.context_soft
        p_tgt    = mask_output.target_soft
        p_ign    = mask_output.aux["p_ign"]
        weights  = mask_output.aux["weights"]
        ema_full = mask_output.aux.get("ema_full")
        soft     = mask_output.aux.get("soft")

        if ema_full is None:
            device = p_ctx.device
            ema_full = torch.zeros(
                p_ctx.shape[0], p_ctx.shape[1], 1, device=device,
            )

        extra_kw = {}
        k = mask_output.aux.get("k")
        if k is not None:
            extra_kw["k"] = k
        n_active = mask_output.aux.get("n_active_tgt")
        if n_active is not None:
            extra_kw["n_active_tgt"] = n_active
        global_step = mask_output.aux.get("global_step")
        if global_step is not None:
            extra_kw["global_step"] = global_step

        total, logs = self.composite_loss(
            weights=weights,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
            soft=soft,
            **extra_kw,
        )

        mask_output.aux.update(logs)

        return reconstruction_loss + total
