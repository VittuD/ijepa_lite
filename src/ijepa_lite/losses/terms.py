"""
Atomic masker loss terms for the compositional MI masker.

Each term is an ``nn.Module`` that receives the soft 3-way assignments
(p_ctx, p_tgt, p_ign) and EMA tokens, and returns
``(scalar_to_minimise, {log_key: value})``.

Terms are fully independent — no shared state, no cross-term coupling.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mean_sq_norm(x: torch.Tensor) -> torch.Tensor:
    """Return mean_d(x^2) without materializing x.pow(2)."""
    return torch.einsum("...d,...d->...", x, x) / x.shape[-1]


class MaskerTerm(nn.Module):
    """Base class for atomic masker loss terms."""

    name: str  # registry key

    def forward(
        self,
        *,
        p_ctx: torch.Tensor,    # (B, N)
        p_tgt: torch.Tensor,    # (B, N)
        p_ign: torch.Tensor,    # (B, N)
        ema_full: torch.Tensor,  # (B, N, D)
        **kw,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError


# ------------------------------------------------------------------
# H(Y|n) — conditional entropy (minimise → confident assignments)
# ------------------------------------------------------------------

class HCondTerm(MaskerTerm):
    name = "H_cond"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)      # (B, N, 3)
        H_cond = -(soft * (soft + 1e-8).log()).sum(-1).mean()   # scalar
        return H_cond, {"entropy_conditional": float(H_cond.detach().item())}


# ------------------------------------------------------------------
# −H(Y) — negative marginal entropy (minimise → maximise H(Y))
# ------------------------------------------------------------------

class NegHMargTerm(MaskerTerm):
    name = "neg_H_marg"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)      # (B, N, 3)
        p_bar = soft.mean(dim=1)                                # (B, 3)
        H_marg = -(p_bar * (p_bar + 1e-8).log()).sum(-1).mean()
        neg_H = -H_marg
        return neg_H, {"entropy_marginal": float(H_marg.detach().item())}


# ------------------------------------------------------------------
# −surprise — negative Bayesian surprise (minimise → maximise surprise)
# ------------------------------------------------------------------

class NegSurpriseTerm(MaskerTerm):
    name = "neg_surprise"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        # Soft context centroid blended with image mean (collapse-safe)
        image_mean = ema_full.mean(dim=1)                                    # (B, D)
        p_ctx_sum = p_ctx.sum(dim=1, keepdim=True)                           # (B, 1)
        ctx_weighted = (p_ctx.unsqueeze(-1) * ema_full).sum(dim=1)           # (B, D)
        virtual_w = (1.0 - p_ctx_sum).clamp(min=0.0)                        # (B, 1)
        ctx_centroid = (ctx_weighted + virtual_w * image_mean) \
                       / (p_ctx_sum + virtual_w).clamp(min=1e-6)             # (B, D)

        BS_all = (ema_full - ctx_centroid.unsqueeze(1)).pow(2).mean(-1)      # (B, N)
        p_tgt_sum = p_tgt.sum(dim=-1).clamp(min=1.0)                        # (B,)
        surprise = ((p_tgt * BS_all).sum(-1) / p_tgt_sum).mean()            # scalar

        return -surprise, {"surprise_mean": float(surprise.detach().item())}


# ------------------------------------------------------------------
# −cosine surprise — scale-invariant context/target dissimilarity
# ------------------------------------------------------------------

class NegCosSurpriseTerm(MaskerTerm):
    name = "neg_cos_surprise"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        # Same collapse-safe context centroid as raw BS, but score with cosine
        # distance so global feature-scale changes do not dominate the term.
        image_mean = ema_full.mean(dim=1)                                    # (B, D)
        p_ctx_sum = p_ctx.sum(dim=1, keepdim=True)                           # (B, 1)
        ctx_weighted = (p_ctx.unsqueeze(-1) * ema_full).sum(dim=1)           # (B, D)
        virtual_w = (1.0 - p_ctx_sum).clamp(min=0.0)                        # (B, 1)
        ctx_centroid = (ctx_weighted + virtual_w * image_mean) \
                       / (p_ctx_sum + virtual_w).clamp(min=1e-6)             # (B, D)

        cos = F.cosine_similarity(
            ema_full,
            ctx_centroid.unsqueeze(1),
            dim=-1,
            eps=1e-8,
        )                                                                    # (B, N)
        cos_surprise_all = 1.0 - cos
        p_tgt_sum = p_tgt.sum(dim=-1).clamp(min=1.0)                         # (B,)
        cos_surprise = ((p_tgt * cos_surprise_all).sum(-1) / p_tgt_sum).mean()

        return -cos_surprise, {
            "cos_surprise_mean": float(cos_surprise.detach().item()),
            "cos_surprise/objective": float(cos_surprise.detach().item()),
            "cos_surprise/tgt_to_ctx": float(cos_surprise.detach().item()),
            "cos_surprise/is_symmetric": 0.0,
        }


class NegSymmetricCosSurpriseTerm(MaskerTerm):
    name = "neg_symmetric_cos_surprise"

    def __init__(self, role_indices: list[int] | tuple[int, ...] = (0, 1)):
        super().__init__()
        if len(role_indices) < 2:
            raise ValueError("neg_symmetric_cos_surprise.role_indices needs at least two roles")
        self.role_indices = tuple(int(i) for i in role_indices)

    @staticmethod
    def _centroid(p: torch.Tensor, ema_full: torch.Tensor) -> torch.Tensor:
        image_mean = ema_full.mean(dim=1)                                    # (B, D)
        p_sum = p.sum(dim=1, keepdim=True)                                   # (B, 1)
        weighted = (p.unsqueeze(-1) * ema_full).sum(dim=1)                   # (B, D)
        virtual_w = (1.0 - p_sum).clamp(min=0.0)                             # (B, 1)
        return (weighted + virtual_w * image_mean) \
               / (p_sum + virtual_w).clamp(min=1e-6)                         # (B, D)

    @staticmethod
    def _weighted_cos_distance(
        p: torch.Tensor,
        ema_full: torch.Tensor,
        centroid: torch.Tensor,
    ) -> torch.Tensor:
        cos = F.cosine_similarity(
            ema_full,
            centroid.unsqueeze(1),
            dim=-1,
            eps=1e-8,
        )                                                                    # (B, N)
        dist = 1.0 - cos
        p_sum = p.sum(dim=-1).clamp(min=1.0)                                 # (B,)
        return ((p * dist).sum(-1) / p_sum).mean()

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")
        if soft is None:
            soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)  # (B, N, 3)
        n_roles = soft.shape[-1]
        bad = [idx for idx in self.role_indices if idx < 0 or idx >= n_roles]
        if bad:
            raise ValueError(
                f"neg_symmetric_cos_surprise.role_indices out of range for {n_roles} roles: {bad}"
            )

        role_probs = [soft[..., idx] for idx in self.role_indices]
        centroids = [self._centroid(p, ema_full) for p in role_probs]

        pair_vals: list[torch.Tensor] = []
        logs: dict[str, float] = {}
        for src_pos, src_role in enumerate(self.role_indices):
            for dst_pos, dst_role in enumerate(self.role_indices):
                if src_pos == dst_pos:
                    continue
                val = self._weighted_cos_distance(
                    role_probs[src_pos],
                    ema_full,
                    centroids[dst_pos],
                )
                pair_vals.append(val)
                logs[f"sym_cos_surprise_role_{src_role}_to_{dst_role}"] = float(
                    val.detach().item()
                )

        sym = torch.stack(pair_vals).mean()
        logs.update({
            "sym_cos_surprise_mean": float(sym.detach().item()),
            "cos_surprise/objective": float(sym.detach().item()),
            "cos_surprise/is_symmetric": 1.0,
            "cos_surprise/n_roles": float(len(self.role_indices)),
        })

        if 0 in self.role_indices and 1 in self.role_indices:
            role_to_pos = {role: pos for pos, role in enumerate(self.role_indices)}
            ctx_pos = role_to_pos[0]
            tgt_pos = role_to_pos[1]
            tgt_to_ctx = self._weighted_cos_distance(
                role_probs[tgt_pos],
                ema_full,
                centroids[ctx_pos],
            )
            ctx_to_tgt = self._weighted_cos_distance(
                role_probs[ctx_pos],
                ema_full,
                centroids[tgt_pos],
            )
            logs.update({
                "sym_cos_surprise_tgt_to_ctx": float(tgt_to_ctx.detach().item()),
                "sym_cos_surprise_ctx_to_tgt": float(ctx_to_tgt.detach().item()),
                "cos_surprise/tgt_to_ctx": float(tgt_to_ctx.detach().item()),
                "cos_surprise/ctx_to_tgt": float(ctx_to_tgt.detach().item()),
            })

        return -sym, logs


class NegSketchedOrthogonalCosSurpriseTerm(MaskerTerm):
    name = "neg_sketched_orthogonal_cos_surprise"

    def __init__(
        self,
        num_sketches: int = 8,
        assign_tau: float = 0.25,
        eps: float = 1e-6,
        sketch_seed: int = 0,
        detach_assignments: bool = True,
    ):
        super().__init__()
        self.num_sketches = int(num_sketches)
        if self.num_sketches <= 0:
            raise ValueError("neg_sketched_orthogonal_cos_surprise.num_sketches must be > 0")
        self.assign_tau = float(assign_tau)
        if self.assign_tau <= 0.0:
            raise ValueError("neg_sketched_orthogonal_cos_surprise.assign_tau must be > 0")
        self.eps = float(eps)
        self.sketch_seed = int(sketch_seed)
        self.detach_assignments = bool(detach_assignments)
        self.register_buffer("_sketch_dirs", torch.empty(0), persistent=False)

    def _get_sketch_dirs(self, dim: int, device: torch.device) -> torch.Tensor:
        if (
            self._sketch_dirs.numel() == 0
            or self._sketch_dirs.shape != (self.num_sketches, dim)
            or self._sketch_dirs.device != device
        ):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.sketch_seed)
            dirs = torch.randn(
                self.num_sketches,
                dim,
                generator=gen,
                dtype=torch.float32,
            )
            self._sketch_dirs = F.normalize(dirs, dim=-1, eps=self.eps).to(device=device)
        return self._sketch_dirs

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        del p_ign, kw

        z_raw = ema_full.float()                                              # (B, N, D)
        z = F.normalize(z_raw, dim=-1, eps=self.eps)                          # (B, N, D)
        dirs = self._get_sketch_dirs(z.shape[-1], z.device)                   # (K, D)

        assign_source = z.detach() if self.detach_assignments else z
        assign_logits = torch.einsum("bnd,kd->bnk", assign_source, dirs)      # (B, N, K)
        assign = F.softmax(assign_logits / self.assign_tau, dim=-1)           # (B, N, K)

        # Build one collapse-safe context prototype per fixed-random sketch
        # bucket. Low-mass buckets fall back toward the image mean instead of
        # creating unstable arbitrary directions.
        p_ctx_f = p_ctx.float()
        p_tgt_f = p_tgt.float()
        image_mean = z_raw.mean(dim=1, keepdim=True)                          # (B, 1, D)
        weights = p_ctx_f.unsqueeze(-1) * assign                              # (B, N, K)
        mass = weights.sum(dim=1)                                             # (B, K)
        weighted_sum = torch.einsum("bnk,bnd->bkd", weights, z_raw)           # (B, K, D)
        virtual_w = (1.0 - mass).clamp(min=0.0)                               # (B, K)
        prototypes = (
            weighted_sum + virtual_w.unsqueeze(-1) * image_mean
        ) / (mass + virtual_w).clamp(min=self.eps).unsqueeze(-1)              # (B, K, D)
        prototypes = F.normalize(prototypes, dim=-1, eps=self.eps)

        cos = torch.einsum("bnd,bkd->bnk", z, prototypes).clamp(min=-1.0, max=1.0)
        explained = cos.square().amax(dim=-1)                                 # (B, N)
        surprise_all = 1.0 - explained                                        # (B, N)
        p_tgt_sum = p_tgt_f.sum(dim=-1).clamp(min=1.0)                        # (B,)
        surprise = ((p_tgt_f * surprise_all).sum(dim=-1) / p_tgt_sum).mean()

        support = (mass / float(ema_full.shape[1])).clamp(min=0.0)
        return -surprise, {
            "sketched_orthogonal_cos_surprise_mean": float(surprise.detach().item()),
            "cos_surprise/objective": float(surprise.detach().item()),
            "cos_surprise/tgt_to_ctx": float(surprise.detach().item()),
            "cos_surprise/is_symmetric": 0.0,
            "cos_surprise/is_sketched_orthogonal": 1.0,
            "cos_surprise/num_sketches": float(self.num_sketches),
            "cos_surprise/assign_tau": float(self.assign_tau),
            "cos_surprise/explained_mean": float(explained.detach().mean().item()),
            "cos_surprise/sketch_ctx_mass_mean": float(mass.detach().mean().item()),
            "cos_surprise/sketch_ctx_mass_min": float(mass.detach().amin(dim=-1).mean().item()),
            "cos_surprise/sketch_ctx_support_mean": float(support.detach().mean().item()),
            "cos_surprise/sketch_ctx_support_min": float(
                support.detach().amin(dim=-1).mean().item()
            ),
        }


class NegPromptContrastTerm(MaskerTerm):
    name = "neg_prompt_contrast"

    def __init__(
        self,
        sample_mode: str = "uniform_distinct",
        sample_temperature: float = 0.5,
        exclude_prompts: bool = True,
        degenerate_cos_threshold: float = 0.9,
        eps: float = 1e-8,
    ):
        super().__init__()
        if sample_mode not in ("uniform_distinct", "feature_far"):
            raise ValueError(
                "neg_prompt_contrast.sample_mode must be "
                f"'uniform_distinct' or 'feature_far', got {sample_mode!r}"
            )
        self.sample_mode = str(sample_mode)
        self.sample_temperature = float(sample_temperature)
        if self.sample_temperature <= 0.0:
            raise ValueError("neg_prompt_contrast.sample_temperature must be > 0")
        self.exclude_prompts = bool(exclude_prompts)
        self.degenerate_cos_threshold = float(degenerate_cos_threshold)
        self.eps = float(eps)

    def _sample_prompt_indices(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample one context prompt and one target prompt per image."""
        B, N, _ = z.shape
        if N < 2:
            raise ValueError("neg_prompt_contrast requires at least two patches")

        device = z.device
        ctx_idx = torch.randint(N, (B,), device=device)

        if self.sample_mode == "uniform_distinct":
            tgt_idx = torch.randint(N - 1, (B,), device=device)
            tgt_idx = tgt_idx + (tgt_idx >= ctx_idx).long()
            entropy = torch.log(z.new_tensor(float(N - 1))).expand(B)
            return ctx_idx, tgt_idx, entropy

        with torch.no_grad():
            batch_idx = torch.arange(B, device=device)
            ctx_feat = z.detach()[batch_idx, ctx_idx]                       # (B, D)
            cos_to_ctx = torch.einsum("bnd,bd->bn", z.detach(), ctx_feat)   # (B, N)
            dist = (1.0 - cos_to_ctx).clamp(min=0.0)
            logits = dist / self.sample_temperature
            logits = logits.scatter(1, ctx_idx.unsqueeze(1), -torch.inf)
            probs = F.softmax(logits, dim=-1)
            tgt_idx = torch.multinomial(probs, num_samples=1).squeeze(1)
            entropy = -(probs * probs.clamp(min=self.eps).log()).sum(dim=-1)
        return ctx_idx, tgt_idx, entropy

    @staticmethod
    def _gather_prompt(z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch_idx = torch.arange(z.shape[0], device=z.device)
        return z[batch_idx, idx]

    def _exclude_prompt_mass(
        self,
        probs: torch.Tensor,
        idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        excluded = probs.gather(1, idx.unsqueeze(1)).squeeze(1)
        if not self.exclude_prompts:
            return probs.float(), excluded
        probs_eff = probs.float().scatter(1, idx.unsqueeze(1), 0.0)
        return probs_eff, excluded

    def _mass_aware_centroid(
        self,
        probs: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mass = probs.sum(dim=-1)                                            # (B,)
        # Deliberately clamp at one patch of soft mass, not eps. Below one
        # patch, the role contributes a mass-scaled partial sum rather than an
        # amplified average from a tiny anchor.
        denom = mass.clamp(min=1.0).unsqueeze(-1)                           # (B, 1)
        centroid = (probs.unsqueeze(-1) * z).sum(dim=1) / denom              # (B, D)
        return centroid, mass

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        del p_ign, kw

        z = F.normalize(ema_full.float(), dim=-1, eps=self.eps)              # (B, N, D)
        ctx_idx, tgt_idx, sampling_entropy = self._sample_prompt_indices(z)

        z_ctx = self._gather_prompt(z, ctx_idx)                              # (B, D)
        z_tgt = self._gather_prompt(z, tgt_idx)                              # (B, D)
        prompt_axis = z_tgt - z_ctx                                          # (B, D)

        p_ctx_eff, prompt_ctx_mass = self._exclude_prompt_mass(p_ctx, ctx_idx)
        p_tgt_eff, prompt_tgt_mass = self._exclude_prompt_mass(p_tgt, tgt_idx)

        mu_ctx, mass_ctx = self._mass_aware_centroid(p_ctx_eff, z)
        mu_tgt, mass_tgt = self._mass_aware_centroid(p_tgt_eff, z)
        role_delta = mu_tgt - mu_ctx

        tgt_projection = (mu_tgt * prompt_axis).sum(dim=-1)                  # (B,)
        ctx_projection = -(mu_ctx * prompt_axis).sum(dim=-1)                 # (B,)
        score_per_image = tgt_projection + ctx_projection
        score = score_per_image.mean()

        origin_cos = (z_ctx * z_tgt).sum(dim=-1).clamp(min=-1.0, max=1.0)
        orientation_cos = F.cosine_similarity(
            role_delta,
            prompt_axis,
            dim=-1,
            eps=self.eps,
        )
        role_delta_norm = role_delta.norm(dim=-1)
        prompt_axis_norm = prompt_axis.norm(dim=-1)
        alignment_scale = role_delta_norm * prompt_axis_norm
        max_sampling_entropy = torch.log(z.new_tensor(float(z.shape[1] - 1)))

        logs = {
            "prompt_contrast/score": float(score.detach().item()),
            "prompt_contrast/tgt_margin": float(tgt_projection.detach().mean().item()),
            "prompt_contrast/ctx_margin": float(ctx_projection.detach().mean().item()),
            "prompt_contrast/orientation_cos": float(orientation_cos.detach().mean().item()),
            "prompt_contrast/role_delta_norm": float(role_delta_norm.detach().mean().item()),
            "prompt_contrast/prompt_axis_norm": float(prompt_axis_norm.detach().mean().item()),
            "prompt_contrast/alignment_scale": float(alignment_scale.detach().mean().item()),
            "prompt_contrast/origin_cos": float(origin_cos.detach().mean().item()),
            "prompt_contrast/origin_distance": float((1.0 - origin_cos).detach().mean().item()),
            "prompt_contrast/degenerate_fraction": float(
                (origin_cos.detach() > self.degenerate_cos_threshold).float().mean().item()
            ),
            "prompt_contrast/sampling_entropy": float(sampling_entropy.detach().mean().item()),
            "prompt_contrast/sampling_entropy_norm": float(
                (
                    sampling_entropy.detach().mean()
                    / max_sampling_entropy.clamp(min=self.eps)
                ).item()
            ),
            "prompt_contrast/mass_ctx": float(mass_ctx.detach().mean().item()),
            "prompt_contrast/mass_tgt": float(mass_tgt.detach().mean().item()),
            "prompt_contrast/mass_ctx_lt_one_frac": float(
                (mass_ctx.detach() < 1.0).float().mean().item()
            ),
            "prompt_contrast/mass_tgt_lt_one_frac": float(
                (mass_tgt.detach() < 1.0).float().mean().item()
            ),
            "prompt_contrast/raw_mass_ctx": float(p_ctx.detach().sum(dim=-1).mean().item()),
            "prompt_contrast/raw_mass_tgt": float(p_tgt.detach().sum(dim=-1).mean().item()),
            "prompt_contrast/prompt_as_context_mass": float(
                prompt_ctx_mass.detach().mean().item()
            ),
            "prompt_contrast/prompt_as_target_mass": float(
                prompt_tgt_mass.detach().mean().item()
            ),
            "prompt_contrast/anchor_shortcut_signal": float(
                (
                    0.5
                    * (
                        prompt_ctx_mass.detach().mean()
                        + prompt_tgt_mass.detach().mean()
                    )
                ).item()
            ),
            "prompt_contrast/exclude_prompts": float(self.exclude_prompts),
            "prompt_contrast/sample_mode_id": 0.0
            if self.sample_mode == "uniform_distinct"
            else 1.0,
            "prompt_contrast/sample_temperature": self.sample_temperature,
        }

        return -score, logs


# ------------------------------------------------------------------
# −logdet diversity — role-wise sketched covariance volume
# ------------------------------------------------------------------

class _BaseLogDetDiversityTerm(MaskerTerm):
    name = "_base_logdet_diversity"
    log_prefix = "logdet"
    variant_id = -1
    uses_mass_scaled_cov = False
    uses_support_multiplier = False

    def __init__(
        self,
        role_indices: list[int] | tuple[int, ...] = (0, 1),
        sketch_dim: int = 32,
        alpha: float = 1.0,
        eps: float = 1e-6,
        projection_seed: int = 0,
    ):
        super().__init__()
        if len(role_indices) == 0:
            raise ValueError("neg_logdet_diversity.role_indices must be non-empty")
        self.role_indices = tuple(int(i) for i in role_indices)
        self.sketch_dim = int(sketch_dim)
        if self.sketch_dim <= 0:
            raise ValueError("neg_logdet_diversity.sketch_dim must be positive")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.projection_seed = int(projection_seed)
        self.register_buffer("_projection", torch.empty(0), persistent=False)

    def _get_projection(self, dim: int, device: torch.device) -> torch.Tensor:
        if (
            self._projection.numel() == 0
            or self._projection.shape != (self.sketch_dim, dim)
            or self._projection.device != device
        ):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.projection_seed)
            proj = torch.randn(
                self.sketch_dim,
                dim,
                generator=gen,
                dtype=torch.float32,
            )
            proj = proj * (self.sketch_dim ** -0.5)
            self._projection = proj.to(device=device)
        return self._projection

    def _compute_role_stats(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")
        if soft is None:
            soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)  # (B, N, 3)
        n_roles = soft.shape[-1]
        bad = [idx for idx in self.role_indices if idx < 0 or idx >= n_roles]
        if bad:
            raise ValueError(
                f"neg_logdet_diversity.role_indices out of range for {n_roles} roles: {bad}"
            )

        # Project normalized EMA tokens to a fixed low-dimensional sketch before
        # the covariance logdet; compute second-order stats in fp32 for stability.
        z = F.normalize(ema_full.float(), dim=-1, eps=self.eps)              # (B, N, D)
        proj = self._get_projection(z.shape[-1], z.device)                  # (S, D)
        u = F.normalize(torch.matmul(z, proj.t()), dim=-1, eps=self.eps)     # (B, N, S)

        role_probs = soft[..., list(self.role_indices)].float()              # (B, N, R)
        mass = role_probs.sum(dim=1)                                         # (B, R)
        weights = role_probs / mass.clamp(min=self.eps).unsqueeze(1)         # (B, N, R)

        mean = torch.einsum("bnr,bns->brs", weights, u)                     # (B, R, S)
        second = torch.einsum("bnr,bns,bnt->brst", weights, u, u)           # (B, R, S, S)
        cov = second - mean.unsqueeze(-1) * mean.unsqueeze(-2)              # (B, R, S, S)
        cov = 0.5 * (cov + cov.transpose(-1, -2))

        eye = torch.eye(self.sketch_dim, device=u.device, dtype=u.dtype)
        mat = eye.view(1, 1, self.sketch_dim, self.sketch_dim) + self.alpha * cov
        _, logdet = torch.linalg.slogdet(mat)                                # (B, R)
        score = logdet.mean()

        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)                       # (B, R)
        sum_w2 = weights.square().sum(dim=1).clamp(min=self.eps)             # (B, R)
        ess = torch.where(mass > self.eps, sum_w2.reciprocal(), torch.zeros_like(sum_w2))
        support = (mass / float(ema_full.shape[1])).clamp(min=0.0)           # (B, R)
        return logdet, mass, ess, trace, support

    def _compute_mass_scaled_role_stats(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")
        if soft is None:
            soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)  # (B, N, 3)
        n_roles = soft.shape[-1]
        bad = [idx for idx in self.role_indices if idx < 0 or idx >= n_roles]
        if bad:
            raise ValueError(
                f"neg_logdet_diversity.role_indices out of range for {n_roles} roles: {bad}"
            )

        # Mass-scaled covariance:
        #   C_r = (1/N) Σ_i p_r(i) (z_i - μ_r)(z_i - μ_r)^T
        # Unlike the normalized covariance above, adding low-diversity patches
        # has diminishing logdet return but no separate support reward.
        z = F.normalize(ema_full.float(), dim=-1, eps=self.eps)              # (B, N, D)
        proj = self._get_projection(z.shape[-1], z.device)                  # (S, D)
        u = F.normalize(torch.matmul(z, proj.t()), dim=-1, eps=self.eps)     # (B, N, S)

        role_probs = soft[..., list(self.role_indices)].float()              # (B, N, R)
        mass = role_probs.sum(dim=1)                                         # (B, R)
        support = (mass / float(ema_full.shape[1])).clamp(min=0.0)           # (B, R)

        weights = role_probs / mass.clamp(min=self.eps).unsqueeze(1)         # (B, N, R)
        mean = torch.einsum("bnr,bns->brs", weights, u)                     # (B, R, S)
        raw_second = torch.einsum("bnr,bns,bnt->brst", role_probs, u, u)
        raw_second = raw_second / float(ema_full.shape[1])                  # (B, R, S, S)
        cov = raw_second - support.unsqueeze(-1).unsqueeze(-1) \
              * mean.unsqueeze(-1) * mean.unsqueeze(-2)                    # (B, R, S, S)
        cov = 0.5 * (cov + cov.transpose(-1, -2))

        eye = torch.eye(self.sketch_dim, device=u.device, dtype=u.dtype)
        mat = eye.view(1, 1, self.sketch_dim, self.sketch_dim) + self.alpha * cov
        _, logdet = torch.linalg.slogdet(mat)                                # (B, R)

        trace = cov.diagonal(dim1=-2, dim2=-1).sum(-1)                       # (B, R)
        sum_w2 = weights.square().sum(dim=1).clamp(min=self.eps)             # (B, R)
        ess = torch.where(mass > self.eps, sum_w2.reciprocal(), torch.zeros_like(sum_w2))
        return logdet, mass, ess, trace, support

    def _build_logs(
        self,
        *,
        score: torch.Tensor,
        logdet: torch.Tensor,
        mass: torch.Tensor,
        ess: torch.Tensor,
        trace: torch.Tensor,
        support: torch.Tensor,
        role_score: torch.Tensor | None = None,
    ) -> dict[str, float]:
        prefix = self.log_prefix
        logs = {
            f"{prefix}_diversity": float(score.detach().item()),
        }
        logdet_by_role = logdet.detach().mean(dim=0)
        mass_by_role = mass.detach().mean(dim=0)
        ess_by_role = ess.detach().mean(dim=0)
        trace_by_role = trace.detach().mean(dim=0)
        support_by_role = support.detach().mean(dim=0)
        role_score_by_role = None if role_score is None else role_score.detach().mean(dim=0)
        score_by_role = logdet_by_role if role_score_by_role is None else role_score_by_role
        logs.update({
            "logdet/objective": float(score.detach().item()),
            "logdet/base_mean": float(logdet.detach().mean().item()),
            "logdet/mass_mean": float(mass.detach().mean().item()),
            "logdet/ess_mean": float(ess.detach().mean().item()),
            "logdet/trace_mean": float(trace.detach().mean().item()),
            "logdet/support_mean": float(support.detach().mean().item()),
            "logdet/variant_id": float(self.variant_id),
            "logdet/uses_mass_scaled_cov": float(self.uses_mass_scaled_cov),
            "logdet/uses_support_multiplier": float(self.uses_support_multiplier),
        })
        for j, role_idx in enumerate(self.role_indices):
            logs[f"{prefix}_diversity_role_{role_idx}"] = float(logdet_by_role[j].item())
            logs[f"{prefix}_mass_role_{role_idx}"] = float(mass_by_role[j].item())
            logs[f"{prefix}_ess_role_{role_idx}"] = float(ess_by_role[j].item())
            logs[f"{prefix}_trace_role_{role_idx}"] = float(trace_by_role[j].item())
            logs[f"{prefix}_support_role_{role_idx}"] = float(support_by_role[j].item())
            logs[f"logdet/base_role_{role_idx}"] = float(logdet_by_role[j].item())
            logs[f"logdet/score_role_{role_idx}"] = float(score_by_role[j].item())
            logs[f"logdet/mass_role_{role_idx}"] = float(mass_by_role[j].item())
            logs[f"logdet/ess_role_{role_idx}"] = float(ess_by_role[j].item())
            logs[f"logdet/trace_role_{role_idx}"] = float(trace_by_role[j].item())
            logs[f"logdet/support_role_{role_idx}"] = float(support_by_role[j].item())
            if role_score_by_role is not None:
                logs[f"{prefix}_supported_role_{role_idx}"] = float(role_score_by_role[j].item())
        return logs


class NegLogDetDiversityTerm(_BaseLogDetDiversityTerm):
    name = "neg_logdet_diversity"
    log_prefix = "logdet"
    variant_id = 0

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        logdet, mass, ess, trace, support = self._compute_role_stats(
            p_ctx=p_ctx, p_tgt=p_tgt, p_ign=p_ign, ema_full=ema_full, **kw,
        )
        score = logdet.mean()
        logs = self._build_logs(
            score=score,
            logdet=logdet,
            mass=mass,
            ess=ess,
            trace=trace,
            support=support,
        )

        return -score, logs


class NegSupportLogDetDiversityTerm(_BaseLogDetDiversityTerm):
    name = "neg_support_logdet_diversity"
    log_prefix = "support_logdet"
    variant_id = 1
    uses_support_multiplier = True

    def __init__(self, support_power: float = 0.5, **kw):
        super().__init__(**kw)
        self.support_power = float(support_power)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        logdet, mass, ess, trace, support = self._compute_role_stats(
            p_ctx=p_ctx, p_tgt=p_tgt, p_ign=p_ign, ema_full=ema_full, **kw,
        )
        support_factor = support.clamp(min=self.eps).pow(self.support_power)
        role_score = support_factor * logdet
        score = role_score.mean()
        logs = self._build_logs(
            score=score,
            logdet=logdet,
            mass=mass,
            ess=ess,
            trace=trace,
            support=support,
            role_score=role_score,
        )
        logs["support_logdet_support_power"] = self.support_power

        return -score, logs


class NegSignedSupportLogDetTerm(_BaseLogDetDiversityTerm):
    name = "neg_signed_support_logdet"
    log_prefix = "signed_support_logdet"
    variant_id = 3
    uses_support_multiplier = True

    def __init__(
        self,
        diverse_role_indices: list[int] | tuple[int, ...] = (0, 1),
        compact_role_indices: list[int] | tuple[int, ...] = (2,),
        compact_weight: float = 1.0,
        support_power: float = 0.5,
        **kw,
    ):
        diverse = tuple(int(i) for i in diverse_role_indices)
        compact = tuple(int(i) for i in compact_role_indices)
        if len(set(diverse)) != len(diverse):
            raise ValueError(
                "neg_signed_support_logdet.diverse_role_indices contains duplicates"
            )
        if len(set(compact)) != len(compact):
            raise ValueError(
                "neg_signed_support_logdet.compact_role_indices contains duplicates"
            )
        if len(diverse) == 0 and len(compact) == 0:
            raise ValueError(
                "neg_signed_support_logdet needs at least one diverse or compact role"
            )
        overlap = sorted(set(diverse).intersection(compact))
        if overlap:
            raise ValueError(
                "neg_signed_support_logdet roles cannot be both diverse and compact: "
                f"{overlap}"
            )

        role_indices = diverse + compact
        super().__init__(role_indices=role_indices, **kw)
        self.diverse_role_indices = diverse
        self.compact_role_indices = compact
        self.compact_weight = float(compact_weight)
        if self.compact_weight < 0.0:
            raise ValueError("neg_signed_support_logdet.compact_weight must be >= 0")
        self.support_power = float(support_power)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        logdet, mass, ess, trace, support = self._compute_role_stats(
            p_ctx=p_ctx, p_tgt=p_tgt, p_ign=p_ign, ema_full=ema_full, **kw,
        )
        support_factor = support.clamp(min=self.eps).pow(self.support_power)
        role_score = support_factor * logdet

        role_to_pos = {role: pos for pos, role in enumerate(self.role_indices)}
        diverse_pos = [role_to_pos[r] for r in self.diverse_role_indices]
        compact_pos = [role_to_pos[r] for r in self.compact_role_indices]

        zero = role_score.new_zeros(())
        diverse_score = role_score[..., diverse_pos].mean() if diverse_pos else zero
        compact_score = role_score[..., compact_pos].mean() if compact_pos else zero
        score = diverse_score - self.compact_weight * compact_score

        logs = self._build_logs(
            score=score,
            logdet=logdet,
            mass=mass,
            ess=ess,
            trace=trace,
            support=support,
            role_score=role_score,
        )
        logs.update({
            "signed_support_logdet_objective": float(score.detach().item()),
            "signed_support_logdet_diverse_score": float(diverse_score.detach().item()),
            "signed_support_logdet_compact_score": float(compact_score.detach().item()),
            "signed_support_logdet_compact_weight": self.compact_weight,
            "signed_support_logdet_support_power": self.support_power,
            "signed_support_logdet_n_diverse_roles": float(len(diverse_pos)),
            "signed_support_logdet_n_compact_roles": float(len(compact_pos)),
            "logdet/signed_diverse_score": float(diverse_score.detach().item()),
            "logdet/signed_compact_score": float(compact_score.detach().item()),
            "logdet/uses_signed_roles": 1.0,
        })

        for role, pos in role_to_pos.items():
            signed = 1.0 if role in self.diverse_role_indices else -self.compact_weight
            contrib = signed * role_score[..., pos].mean()
            logs[f"signed_support_logdet_sign_role_{role}"] = float(signed)
            logs[f"signed_support_logdet_contrib_role_{role}"] = float(
                contrib.detach().item()
            )

        return -score, logs


class NegMassLogDetDiversityTerm(_BaseLogDetDiversityTerm):
    name = "neg_mass_logdet_diversity"
    log_prefix = "mass_logdet"
    variant_id = 2
    uses_mass_scaled_cov = True

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        logdet, mass, ess, trace, support = self._compute_mass_scaled_role_stats(
            p_ctx=p_ctx, p_tgt=p_tgt, p_ign=p_ign, ema_full=ema_full, **kw,
        )
        score = logdet.mean()
        logs = self._build_logs(
            score=score,
            logdet=logdet,
            mass=mass,
            ess=ess,
            trace=trace,
            support=support,
        )

        return -score, logs


# ------------------------------------------------------------------
# −centroid distance — ||μ_ctx − μ_tgt||² (minimise → maximise)
#
# Same as neg_surprise but without the within-target variance term.
# See bias-variance decomposition:
#   E_tgt[||z_i − μ_ctx||²] = ||μ_tgt − μ_ctx||² + Var_tgt
# ------------------------------------------------------------------

class NegCentroidDistTerm(MaskerTerm):
    name = "neg_centroid_dist"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        # Context centroid (collapse-safe, blended with image mean)
        image_mean = ema_full.mean(dim=1)                                    # (B, D)
        p_ctx_sum = p_ctx.sum(dim=1, keepdim=True)                           # (B, 1)
        ctx_weighted = (p_ctx.unsqueeze(-1) * ema_full).sum(dim=1)           # (B, D)
        virtual_w = (1.0 - p_ctx_sum).clamp(min=0.0)                        # (B, 1)
        ctx_centroid = (ctx_weighted + virtual_w * image_mean) \
                       / (p_ctx_sum + virtual_w).clamp(min=1e-6)             # (B, D)

        # Target centroid (same blending for symmetry)
        p_tgt_sum = p_tgt.sum(dim=1, keepdim=True)                           # (B, 1)
        tgt_weighted = (p_tgt.unsqueeze(-1) * ema_full).sum(dim=1)           # (B, D)
        virtual_w_tgt = (1.0 - p_tgt_sum).clamp(min=0.0)                    # (B, 1)
        tgt_centroid = (tgt_weighted + virtual_w_tgt * image_mean) \
                       / (p_tgt_sum + virtual_w_tgt).clamp(min=1e-6)         # (B, D)

        dist = (ctx_centroid - tgt_centroid).pow(2).mean(-1).mean()          # scalar

        return -dist, {"centroid_dist_mean": float(dist.detach().item())}


# ------------------------------------------------------------------
# Floor penalty — ReLU(h_floor − H(Y|n))²
# ------------------------------------------------------------------

class FloorPenaltyTerm(MaskerTerm):
    name = "floor_penalty"

    def __init__(self, h_floor: float = 0.1, num_tgt_blocks: int = 1):
        super().__init__()
        self.h_floor = float(h_floor)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2) when N-way, None for 3-way
        if soft is None:
            soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)  # (B, N, 3)
        H_cond = -(soft * (soft + 1e-8).log()).sum(-1).mean()
        penalty = F.relu(self.h_floor - H_cond).pow(2)
        return penalty, {"floor_penalty": float(penalty.detach().item())}


# ------------------------------------------------------------------
# Ignore tax — Σ p_ign · prior_BS
# ------------------------------------------------------------------

class IgnoreTaxTerm(MaskerTerm):
    name = "ignore_tax"

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        # prior BS = squared distance from image mean
        image_mean = ema_full.mean(dim=1, keepdim=True)             # (B, 1, D)
        prior_bs = (ema_full - image_mean).pow(2).mean(-1)          # (B, N)
        tax = (p_ign * prior_bs).sum(-1).mean()                     # scalar
        ign_rate = float(p_ign.detach().mean().item())
        return tax, {"ign_rate": ign_rate}


# ------------------------------------------------------------------
# Context rate — (1/N) Σ p_ctx
# ------------------------------------------------------------------

class ContextRateTerm(MaskerTerm):
    name = "context_rate"

    def __init__(self, num_patches: int = 1):
        super().__init__()
        self.num_patches = int(num_patches)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        R_ctx = p_ctx.mean()  # mean over batch and patches
        return R_ctx, {"R_ctx": float(R_ctx.detach().item())}


# ------------------------------------------------------------------
# Target rate — (1/N) Σ p_tgt
# ------------------------------------------------------------------

class TargetRateTerm(MaskerTerm):
    name = "target_rate"

    def __init__(self, num_patches: int = 1):
        super().__init__()
        self.num_patches = int(num_patches)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        R_tgt = p_tgt.mean()
        return R_tgt, {"R_tgt": float(R_tgt.detach().item())}


# ------------------------------------------------------------------
# N-way cross-target surprise
# ------------------------------------------------------------------

class NWayCrossSurpriseTerm(MaskerTerm):
    """Inter-target surprise: Σ_{k≠l} E_{tgt_k}[‖zᵢ − μ_{tgt_l}‖²].

    Only pushes target blocks apart — no ctx-vs-tgt terms (adversarial).
    """
    name = "nway_cross_surprise"

    def __init__(self, num_tgt_blocks: int = 4):
        super().__init__()
        self.M = int(num_tgt_blocks)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2)
        if soft is None:
            return p_ctx.new_zeros(()), {}

        M = min(self.M, int(kw.get("n_active_tgt", self.M)))
        if M < 2:
            return p_ctx.new_zeros(()), {}
        D = ema_full.shape[-1]
        image_mean = ema_full.mean(dim=1)  # (B, D)

        # --- Vectorized centroids: one bmm instead of M weighted sums ---
        # p_tgts: (B, N, M) → bmm with (B, N, D) → (B, M, D)
        p_tgts = soft[..., 1:M+1]                                           # (B, N, M)
        p_tgts_sum = p_tgts.sum(dim=1)                                      # (B, M)
        weighted = torch.bmm(p_tgts.transpose(1, 2), ema_full)              # (B, M, D)
        virtual_w = (1.0 - p_tgts_sum).clamp(min=0.0).unsqueeze(-1)        # (B, M, 1)
        centroids = (weighted + virtual_w * image_mean.unsqueeze(1)) \
                    / (p_tgts_sum.unsqueeze(-1) + virtual_w).clamp(min=1e-6)  # (B, M, D)

        # --- Vectorized pairwise distances: one bmm instead of M*(M-1) dots ---
        # ‖zᵢ − μ_l‖² = mean_d(zᵢ²) − 2·mean_d(zᵢ·μ_l) + mean_d(μ_l²)
        norm_ema_sq = _mean_sq_norm(ema_full)                               # (B, N)
        all_dots = torch.bmm(ema_full, centroids.transpose(1, 2)) / D      # (B, N, M)
        all_norm_c = _mean_sq_norm(centroids).unsqueeze(1)                  # (B, 1, M)
        dist_sq_all = norm_ema_sq.unsqueeze(-1) - 2.0 * all_dots + all_norm_c  # (B, N, M)

        # S_total = Σ_{k≠l} E_k[dist_sq_l]
        #         = Σ_k E_k[Σ_l dist_sq_l − dist_sq_k]   (subtract self-pair)
        dist_cross = dist_sq_all.sum(-1, keepdim=True) - dist_sq_all        # (B, N, M)
        p_tgts_sum_c = p_tgts_sum.clamp(min=1.0)                           # (B, M)
        S_total = ((p_tgts * dist_cross).sum(1) / p_tgts_sum_c).mean(0).sum()

        return -S_total, {"cross_surprise_mean": float(S_total.detach().item())}


# ------------------------------------------------------------------
# N-way full cross-surprise (targets + context)
# ------------------------------------------------------------------

class NWayFullCrossSurpriseTerm(MaskerTerm):
    """Cross-surprise over all M+1 groups (ctx + M targets, ignoring ign).

    S = S_tgt_tgt + ctx_weight · S_ctx_tgt

    where S_tgt_tgt is the inter-target surprise (same as NWayCrossSurpriseTerm)
    and S_ctx_tgt includes ctx↔tgt_k cross-pairs in both directions.
    ctx_weight controls how much the ctx-vs-target terms contribute.
    """
    name = "nway_full_cross_surprise"

    def __init__(self, num_tgt_blocks: int = 4, ctx_weight: float = 1.0):
        super().__init__()
        self.M = int(num_tgt_blocks)
        self.ctx_weight = float(ctx_weight)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2)
        if soft is None:
            return p_ctx.new_zeros(()), {}

        M = min(self.M, int(kw.get("n_active_tgt", self.M)))
        if M < 1:
            return p_ctx.new_zeros(()), {}
        D = ema_full.shape[-1]
        image_mean = ema_full.mean(dim=1)  # (B, D)

        # --- Vectorized centroids for ctx (idx 0) + M targets (idx 1..M) ---
        p_groups = soft[..., :M+1]                                           # (B, N, M+1)
        p_groups_sum = p_groups.sum(dim=1)                                   # (B, M+1)
        weighted = torch.bmm(p_groups.transpose(1, 2), ema_full)             # (B, M+1, D)
        virtual_w = (1.0 - p_groups_sum).clamp(min=0.0).unsqueeze(-1)       # (B, M+1, 1)
        centroids = (weighted + virtual_w * image_mean.unsqueeze(1)) \
                    / (p_groups_sum.unsqueeze(-1) + virtual_w).clamp(min=1e-6)  # (B, M+1, D)

        # --- Vectorized pairwise distances for all M+1 groups ---
        norm_ema_sq = _mean_sq_norm(ema_full)                                # (B, N)
        all_dots = torch.bmm(ema_full, centroids.transpose(1, 2)) / D       # (B, N, M+1)
        all_norm_c = _mean_sq_norm(centroids).unsqueeze(1)                   # (B, 1, M+1)
        dist_sq_all = norm_ema_sq.unsqueeze(-1) - 2.0 * all_dots + all_norm_c  # (B, N, M+1)

        # --- Inter-target surprise: pairs (k,l) both in {1..M} ---
        p_tgts = p_groups[..., 1:]                                           # (B, N, M)
        p_tgts_sum = p_groups_sum[:, 1:].clamp(min=1.0)                     # (B, M)
        dist_sq_tgts = dist_sq_all[..., 1:]                                  # (B, N, M)
        dist_cross_tgts = dist_sq_tgts.sum(-1, keepdim=True) - dist_sq_tgts  # (B, N, M)
        S_tgt = ((p_tgts * dist_cross_tgts).sum(1) / p_tgts_sum).mean(0).sum()

        # --- Ctx↔target surprise: pairs involving ctx (index 0) ---
        p_0 = p_groups[..., 0]                                               # (B, N)
        p_0_sum = p_groups_sum[:, 0].clamp(min=1.0)                          # (B,)
        # ctx → each tgt: E_ctx[dist(z, c_k)] for k in 1..M
        S_ctx_to_tgt = (p_0.unsqueeze(-1) * dist_sq_all[..., 1:]).sum(1) \
                       / p_0_sum.unsqueeze(-1)                               # (B, M)
        # each tgt → ctx: E_{tgt_k}[dist(z, c_ctx)] for k in 1..M
        S_tgt_to_ctx = (p_tgts * dist_sq_all[..., :1]).sum(1) \
                       / p_tgts_sum                                          # (B, M)
        S_ctx = (S_ctx_to_tgt + S_tgt_to_ctx).mean(0).sum()

        S_total = S_tgt + self.ctx_weight * S_ctx

        return -S_total, {
            "full_cross_surprise_mean": float(S_total.detach().item()),
            "cross_surprise_tgt": float(S_tgt.detach().item()),
            "cross_surprise_ctx": float(S_ctx.detach().item()),
        }


# ------------------------------------------------------------------
# N-way negative marginal entropy
# ------------------------------------------------------------------

class NWayNegHMargTerm(MaskerTerm):
    """−H(Y) over (M+2)-dim marginal distribution."""
    name = "nway_neg_H_marg"

    def __init__(self, num_tgt_blocks: int = 4):
        super().__init__()

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2)
        if soft is None:
            return p_ctx.new_zeros(()), {}

        p_bar = soft.mean(dim=1)  # (B, M+2)
        H_marg = -(p_bar * (p_bar + 1e-8).log()).sum(-1).mean()
        return -H_marg, {"nway_entropy_marginal": float(H_marg.detach().item())}


# ------------------------------------------------------------------
# Role-alive penalty — ReLU(p_min − p_c(n))² per role per patch
#
# Prevents role death: ensures every role maintains at least p_min
# mass on every patch, so gradients flow to all roles and the masker
# can reassign patches as the encoder evolves.
# ------------------------------------------------------------------

class RoleAliveTerm(MaskerTerm):
    name = "role_alive"

    def __init__(self, p_min: float = 0.02, num_tgt_blocks: int = 1):
        super().__init__()
        self.p_min = float(p_min)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")  # (B, N, M+2) when N-way, None for 3-way
        if soft is None:
            soft = torch.stack([p_ctx, p_tgt, p_ign], dim=-1)  # (B, N, 3)
        # Per-role, per-patch: penalise any probability below p_min
        deficit = F.relu(self.p_min - soft)           # (B, N, C)
        penalty = deficit.pow(2).mean()                # scalar
        # Log the fraction of (patch, role) pairs that are below p_min
        dead_frac = float((soft.detach() < self.p_min).float().mean().item())
        return penalty, {
            "role_alive_penalty": float(penalty.detach().item()),
            "role_dead_frac": dead_frac,
        }


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
# KL-to-target marginal — KL(p_bar || q(k))
#
# Generalises nway_neg_H_marg.  The target distribution q(k) allocates
# mass k to context, (1-k)/(M+1) to each other role.  At k = 1/(M+2)
# the loss reduces to -H(p_bar) + const (pure entropy maximisation).
# k is provided by the masker via **kw and follows a warmup+cosine
# schedule across epochs.
# ------------------------------------------------------------------

class NWayKLMargTerm(MaskerTerm):
    """KL(p_bar || q(k)) over (M+2)-dim marginal distribution."""
    name = "nway_kl_marg"

    def __init__(self, num_tgt_blocks: int = 4):
        super().__init__()
        self.M = int(num_tgt_blocks)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")           # (B, N, M+2)
        k = kw.get("k")                 # float or None
        if soft is None or k is None:
            return p_ctx.new_zeros(()), {}

        p_bar = soft.mean(dim=1)        # (B, M+2)

        # Target: ctx gets k, rest get (1-k)/(M+1)
        q = torch.full_like(p_bar, (1.0 - k) / (self.M + 1))
        q[:, 0] = k

        # KL(p_bar || q)
        kl = (p_bar * (p_bar.clamp(min=1e-8).log() - q.clamp(min=1e-8).log())).sum(-1)
        kl_mean = kl.mean()

        # Also log entropy for comparison
        H_marg = -(p_bar * (p_bar + 1e-8).log()).sum(-1).mean()

        return kl_mean, {
            "kl_marg": float(kl_mean.detach().item()),
            "nway_entropy_marginal": float(H_marg.detach().item()),
            "k_schedule": float(k),
        }


# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Progressive KL marginal — KL(q̄ ‖ p̄) / KL(p̄ ‖ q̄) / sum
#
# p̄ is a scheduled uniform distribution over the currently active
# roles {ctx, tgt₁…tgt_n_active, ign}.  Inactive tgt roles receive
# inactive_eps mass so forward KL stays finite.
#
# n_active_tgt is passed per-step via **kw (set by MINWayMasker).
# global_step is passed per-step via **kw for transition blending.
# direction: "forward" = KL(q̄‖p̄), "reverse" = KL(p̄‖q̄), "sum" = both.
#
# Smooth transitions: when n_active_tgt increases, the target linearly
# blends from the current q_eff to the new target over transition_steps
# steps, avoiding the KL discontinuity from hard phase switches.
# ------------------------------------------------------------------

class NWayProgressiveKLTerm(MaskerTerm):
    """Scheduled-target KL for progressive role unlocking with smooth transitions."""
    name = "nway_progressive_kl"

    def __init__(
        self,
        num_tgt_blocks: int = 4,
        direction: str = "forward",
        inactive_eps: float = 1e-4,
        transition_steps: int = 0,
    ):
        super().__init__()
        self.M = int(num_tgt_blocks)
        assert direction in ("forward", "reverse", "sum"), \
            f"direction must be 'forward', 'reverse', or 'sum', got {direction!r}"
        self.direction = str(direction)
        self.inactive_eps = float(inactive_eps)
        # 0 (default): hard switch — no blending.
        # -1: auto — MINWayMasker sets this to the inter-phase gap on first set_step().
        # >0: explicit smooth transition over that many steps.
        self.transition_steps = int(transition_steps)

        # Transition state — persistent so mid-transition survives checkpoint/resume.
        # _transition_start = -1 means "not yet initialized".
        self.register_buffer("_q_from", torch.zeros(self.M + 2), persistent=True)
        self.register_buffer("_q_to", torch.zeros(self.M + 2), persistent=True)
        self.register_buffer(
            "_transition_start", torch.tensor(-1, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "_prev_n_active", torch.tensor(-1, dtype=torch.long), persistent=True
        )
        # n_active at the start of the current transition (for smooth logging)
        self.register_buffer(
            "_n_active_from", torch.tensor(0, dtype=torch.long), persistent=True
        )

    def _build_target(self, n_active: int) -> torch.Tensor:
        """Return (M+2,) uniform target distribution for n_active active tgt roles."""
        M = self.M
        n_inactive = M - n_active
        n_active_roles = n_active + 2  # ctx + n_active tgts + ign
        active_mass = (1.0 - self.inactive_eps * n_inactive) / n_active_roles
        q = torch.full((M + 2,), self.inactive_eps, dtype=torch.float32)
        q[0] = active_mass    # ctx
        q[-1] = active_mass   # ign
        for k in range(n_active):
            q[1 + k] = active_mass
        return q

    def _get_q_eff(self, global_step: int) -> tuple[torch.Tensor, float]:
        """Return (q_eff, alpha) blended target at global_step."""
        t_start = int(self._transition_start.item())
        if t_start < 0 or self.transition_steps <= 0:
            return self._q_to.clone(), 1.0
        alpha = float(min(1.0, max(0.0, (global_step - t_start) / self.transition_steps)))
        q_eff = (1.0 - alpha) * self._q_from + alpha * self._q_to
        return q_eff, alpha

    def reset_parameters(self) -> None:
        self._q_from.zero_()
        self._q_to.zero_()
        self._transition_start.fill_(-1)
        self._prev_n_active.fill_(-1)
        self._n_active_from.fill_(0)

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        soft = kw.get("soft")              # (B, N, M+2)
        n_active = kw.get("n_active_tgt")  # int
        global_step = int(kw.get("global_step", 0))
        if soft is None or n_active is None:
            return p_ctx.new_zeros(()), {}

        n_active = int(n_active)
        prev_n = int(self._prev_n_active.item())

        # --- Detect phase change (or first call) ---
        if prev_n != n_active:
            device = self._q_from.device
            new_q = self._build_target(n_active).to(device)
            if self._transition_start.item() < 0:
                # First call ever: jump directly (no transition)
                self._q_from.copy_(new_q)
                self._q_to.copy_(new_q)
                self._transition_start.fill_(global_step)
                self._n_active_from.fill_(n_active)   # from == to → smooth value = n_active
            else:
                # Phase advanced: blend from current q_eff to new target
                current_q_eff, _ = self._get_q_eff(global_step)
                self._q_from.copy_(current_q_eff.to(device))
                self._q_to.copy_(new_q)
                self._transition_start.fill_(global_step)
                self._n_active_from.fill_(n_active)    # smooth starts at the just-unlocked count
            self._prev_n_active.fill_(n_active)

        p_bar = soft.mean(dim=1)  # (B, M+2)

        # Blended target distribution
        q_eff, alpha = self._get_q_eff(global_step)
        p_target = q_eff.to(p_bar.device).unsqueeze(0).expand_as(p_bar)

        EPS = 1e-8
        log_q = p_bar.clamp(min=EPS).log()
        log_p = p_target.clamp(min=EPS).log()

        # KL(q̄ ‖ p̄) — zero-forcing: penalises mass in inactive roles
        forward_kl = (p_bar * (log_q - log_p)).sum(-1).mean()
        # KL(p̄ ‖ q̄) — zero-avoiding: no penalty for mass in inactive roles
        reverse_kl = (p_target * (log_p - log_q)).sum(-1).mean()

        if self.direction == "forward":
            loss = forward_kl
        elif self.direction == "reverse":
            loss = reverse_kl
        else:  # sum
            loss = forward_kl + reverse_kl

        n_active_smooth = float(self._n_active_from.item()) + alpha

        return loss, {
            "prog_kl/forward": float(forward_kl.detach().item()),
            "prog_kl/reverse": float(reverse_kl.detach().item()),
            "prog_kl/loss": float(loss.detach().item()),
            "prog_kl/n_active_tgt": n_active_smooth,
            "prog_kl/transition_alpha": alpha,
        }


# ------------------------------------------------------------------
# Affinity-coverage novelty (loss_formulation.tex / novelty_proxy.tex)
#
# L = -U_ctx - w_tgt·U_tgt + γ·log(1+α)·C_act
# Coverage scores c_j, t_j come from a sketched RBF affinity A_{ij}
# on EMA embeddings with A_{ii}=0 (load-bearing) and adaptive bandwidth
# σ² from per-image median pair distance.
# ------------------------------------------------------------------

class NegAffinityNoveltyTerm(MaskerTerm):
    name = "neg_affinity_novelty"

    def __init__(
        self,
        alpha: float = 4.0,
        w_tgt: float = 1.0,
        gamma: float | Sequence[float] = 0.5,
        gamma_schedule: str = "constant",
        sketch_dim: int = 64,
        eps: float = 1e-4,
        sigma_rho: float | Sequence[float] = 1.0,
        sigma_rho_schedule: str = "constant",
        sigma_floor: float = 1e-6,
        sketch_seed: int = 0,
        norm_eps: float = 1e-8,
        eff_rank_every_steps: int = 50,
        n_candidates: int = 16,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.w_tgt = float(w_tgt)
        if isinstance(gamma, Sequence) and not isinstance(gamma, (str, bytes)):
            if len(gamma) != 2:
                raise ValueError(
                    "neg_affinity_novelty.gamma must be a scalar or a [start, end] pair"
                )
            gamma_start = float(gamma[0])
            gamma_end = float(gamma[1])
        else:
            gamma_start = float(gamma)
            gamma_end = float(gamma)
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.gamma_schedule = str(gamma_schedule).lower()
        if self.gamma_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                "neg_affinity_novelty.gamma_schedule must be "
                "'constant', 'linear', or 'cosine'"
            )
        self.sketch_dim = int(sketch_dim)
        if self.sketch_dim <= 0:
            raise ValueError("neg_affinity_novelty.sketch_dim must be > 0")
        self.eps = float(eps)
        if isinstance(sigma_rho, Sequence) and not isinstance(sigma_rho, (str, bytes)):
            if len(sigma_rho) != 2:
                raise ValueError(
                    "neg_affinity_novelty.sigma_rho must be a scalar or a [start, end] pair"
                )
            sigma_rho_start = float(sigma_rho[0])
            sigma_rho_end = float(sigma_rho[1])
        else:
            sigma_rho_start = float(sigma_rho)
            sigma_rho_end = float(sigma_rho)
        if sigma_rho_start <= 0.0 or sigma_rho_end <= 0.0:
            raise ValueError("neg_affinity_novelty.sigma_rho endpoints must be > 0")
        self.sigma_rho_start = sigma_rho_start
        self.sigma_rho_end = sigma_rho_end
        self.sigma_rho_schedule = str(sigma_rho_schedule).lower()
        if self.sigma_rho_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                "neg_affinity_novelty.sigma_rho_schedule must be "
                "'constant', 'linear', or 'cosine'"
            )
        self.sigma_floor = float(sigma_floor)
        self.sketch_seed = int(sketch_seed)
        self.norm_eps = float(norm_eps)
        self.eff_rank_every_steps = max(int(eff_rank_every_steps), 1)
        self.n_candidates = max(int(n_candidates), 2)
        self.register_buffer("_sketch_R", torch.empty(0), persistent=False)
        self.register_buffer(
            "_step_counter", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "_eff_rank_mean", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_eff_rank_min", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        # Candidate test (masker-free): does the affinity OBJECTIVE reward
        # per-image masks, or is one static mask optimal for every image?
        # Evaluate K random partitions (shared across the batch, activity matched
        # to the masker's current occupancy) on every image's own affinity A/Z.
        # cand_gap = mean_i[ L(k_global*, i) - min_k L(k, i) ], where k_global* is
        # the single candidate with the lowest mean loss. gap~0 => one mask wins
        # for ~everyone => objective is image-invariant (fix the objective).
        # gap>>0 => different images prefer different masks => per-image structure
        # exists and the masker isn't using it (shortcut). This is immune to
        # masker collapse: candidates are random, not the masker's output.
        self.register_buffer(
            "_cand_gap", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_global_loss", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_perimg_loss", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_argmin_frac", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_nctx", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_ntgt", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        # cand_spread = mean_i[max_k L(k,i) - min_k L(k,i)]: how much the objective
        # differentiates masks AT ALL. spread~0 => objective is mask-insensitive
        # (gap~0 is trivial). spread>>gap => masks differ but ranking is image-
        # invariant (one mask wins for everyone).
        self.register_buffer(
            "_cand_spread", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        # Residual-A variant: same candidate test after removing the position-
        # conditional (stationary) mode from the features. If cand_gap_resid >>
        # cand_gap, per-image structure EXISTS but the stationary mode hid it
        # (=> residualize before building A). If still ~0, even that can't extract
        # it (=> need a cross-image contrastive objective).
        self.register_buffer(
            "_cand_gap_resid", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_argmin_frac_resid", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_global_loss_resid", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_spread_resid", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        # Difficulty variant: replace the COVERAGE cost with a PREDICTABILITY cost
        # (reuse A as a Nadaraya-Watson regressor; targets = patches hard to
        # predict from the candidate's context). Coverage is rank-invariant (no
        # mask x image interaction); predictability should NOT be. Read
        # cand_gap_diff RELATIVE to cand_spread_diff (different units from
        # coverage). gap_diff/spread_diff >> the coverage ratio => functional form
        # restored interaction => build a difficulty-based role term.
        self.register_buffer(
            "_cand_gap_diff", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_argmin_frac_diff", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_global_loss_diff", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_cand_spread_diff", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )

    def _get_sketch(self, dim: int, device: torch.device) -> torch.Tensor:
        if (
            self._sketch_R.numel() == 0
            or self._sketch_R.shape != (self.sketch_dim, dim)
            or self._sketch_R.device != device
        ):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.sketch_seed)
            R = torch.randn(
                self.sketch_dim, dim, generator=gen, dtype=torch.float32
            )
            R = R * (self.sketch_dim ** -0.5)  # entries ~ N(0, 1/S)
            self._sketch_R = R.to(device=device)
        return self._sketch_R

    def _get_sigma_rho(self, epoch: int | None, total_epochs: int | None) -> float:
        if self.sigma_rho_start == self.sigma_rho_end or self.sigma_rho_schedule == "constant":
            return self.sigma_rho_start

        if epoch is None or total_epochs is None or total_epochs <= 1:
            return self.sigma_rho_start

        progress = min(max(float(epoch) / float(total_epochs - 1), 0.0), 1.0)
        if self.sigma_rho_schedule == "linear":
            blend = progress
        else:
            blend = 0.5 * (1.0 - math.cos(math.pi * progress))

        return self.sigma_rho_start + (self.sigma_rho_end - self.sigma_rho_start) * blend

    def _get_gamma(self, epoch: int | None, total_epochs: int | None) -> float:
        if self.gamma_start == self.gamma_end or self.gamma_schedule == "constant":
            return self.gamma_start

        if epoch is None or total_epochs is None or total_epochs <= 1:
            return self.gamma_start

        progress = min(max(float(epoch) / float(total_epochs - 1), 0.0), 1.0)
        if self.gamma_schedule == "linear":
            blend = progress
        else:
            blend = 0.5 * (1.0 - math.cos(math.pi * progress))

        return self.gamma_start + (self.gamma_end - self.gamma_start) * blend

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        del p_ign

        z_hat = F.normalize(ema_full.float(), dim=-1, eps=self.norm_eps)   # (B,N,D)
        B, N, D = z_hat.shape
        device = z_hat.device
        epoch = kw.get("epoch")
        total_epochs = kw.get("total_epochs")
        sigma_rho = self._get_sigma_rho(
            None if epoch is None else int(epoch),
            None if total_epochs is None else int(total_epochs),
        )
        gamma = self._get_gamma(
            None if epoch is None else int(epoch),
            None if total_epochs is None else int(total_epochs),
        )

        R = self._get_sketch(D, device)                                    # (S, D)
        y = torch.einsum("bnd,sd->bns", z_hat, R)                          # (B, N, S)

        y_norm = (y * y).sum(dim=-1)                                       # (B, N)
        gram = torch.einsum("bns,bms->bnm", y, y)                          # (B, N, N)
        d_sq = (
            y_norm.unsqueeze(2) + y_norm.unsqueeze(1) - 2.0 * gram
        ).clamp(min=0.0)

        # Per-image bandwidth: median of off-diagonal pair distances.
        eye_mask = torch.eye(N, dtype=torch.bool, device=device)
        d_for_med = d_sq.masked_fill(eye_mask, float("nan"))
        sigma_sq = torch.nanmedian(d_for_med.flatten(1), dim=-1).values    # (B,)
        sigma_sq = (sigma_rho * sigma_sq).clamp(min=self.sigma_floor)

        A = torch.exp(-d_sq / (2.0 * sigma_sq.view(B, 1, 1)))               # (B, N, N)
        A = A.masked_fill(eye_mask, 0.0)                                    # diagonal-zero

        Z = A.sum(dim=1) + self.eps                                         # (B, N)
        ctx_num = torch.einsum("bi,bij->bj", p_ctx.float(), A)              # (B, N)
        tgt_num = torch.einsum("bi,bij->bj", p_tgt.float(), A)              # (B, N)
        c = ctx_num / Z
        t = tgt_num / Z

        U_ctx = torch.log1p(self.alpha * c).mean()
        ratio = t / (self.eps + c + t)
        U_tgt = torch.log1p(self.alpha * ratio).mean()
        C_act = (p_ctx + p_tgt).mean()

        coupling = gamma * math.log1p(self.alpha)
        loss = -U_ctx - self.w_tgt * U_tgt + coupling * C_act

        # Effective rank via squared-eigenvalue participation ratio:
        # r = (Σ λ²)² / Σ λ⁴ = ‖A‖_F⁴ / ‖A²‖_F²
        # Throttled — A² is an (N×N) matmul per image.
        step = int(self._step_counter.item())
        if step % self.eff_rank_every_steps == 0:
            with torch.no_grad():
                frob_sq = (A * A).sum(dim=(-1, -2))                         # (B,)
                A_sq = torch.bmm(A, A)                                      # (B, N, N)
                A_sq_frob_sq = (A_sq * A_sq).sum(dim=(-1, -2))              # (B,)
                eff_rank = frob_sq.square() / A_sq_frob_sq.clamp(min=1e-12)
                self._eff_rank_mean.fill_(float(eff_rank.mean().item()))
                self._eff_rank_min.fill_(float(eff_rank.amin().item()))

                # --- Candidate test (masker-free per-image structure) --------
                # K random partitions, SHARED across the batch, activity matched
                # to the masker's current mean occupancy. Same counts for all K so
                # the (dropped) C_act term is constant -> cost isolates structure.
                K = self.n_candidates
                n_ctx = int(p_ctx.float().sum(dim=-1).mean().round().item())
                n_tgt = int(p_tgt.float().sum(dim=-1).mean().round().item())
                n_ctx = max(1, min(n_ctx, N - 1))
                n_tgt = max(1, min(n_tgt, N - n_ctx))
                # K random permutations -> first n_ctx are ctx, next n_tgt are tgt.
                order = torch.rand(K, N, device=device).argsort(dim=1)          # (K,N)
                pc_cand = torch.zeros(K, N, device=device)
                pt_cand = torch.zeros(K, N, device=device)
                rows = torch.arange(K, device=device).unsqueeze(1)
                pc_cand[rows, order[:, :n_ctx]] = 1.0
                pt_cand[rows, order[:, n_ctx:n_ctx + n_tgt]] = 1.0

                def _cand_cost(A_, Z_):
                    # cost (K,B): every candidate on every image's own A/Z.
                    Zc = Z_.unsqueeze(0)                                       # (1,B,N)
                    c_ = torch.einsum("kp,bpj->kbj", pc_cand, A_) / Zc
                    t_ = torch.einsum("kp,bpj->kbj", pt_cand, A_) / Zc
                    Uc = torch.log1p(self.alpha * c_).mean(dim=-1)            # (K,B)
                    rt = t_ / (self.eps + c_ + t_)
                    Ut = torch.log1p(self.alpha * rt).mean(dim=-1)            # (K,B)
                    return -Uc - self.w_tgt * Ut                             # (K,B)

                def _cand_metrics(cost):
                    per_best = cost.min(dim=0).values                        # (B,)
                    k_glob = cost.mean(dim=1).argmin()
                    g_cost = cost[k_glob]                                     # (B,)
                    gap = (g_cost - per_best).mean()
                    frac = (cost.argmin(dim=0) == k_glob).float().mean()
                    spread = (cost.amax(dim=0) - per_best).mean()
                    return gap, frac, g_cost.mean(), per_best.mean(), spread

                # Raw affinity: is there per-image structure as-is?
                gap, frac, gloss, ploss, spread = _cand_metrics(_cand_cost(A, Z))
                self._cand_gap.fill_(float(gap.item()))
                self._cand_argmin_frac.fill_(float(frac.item()))
                self._cand_global_loss.fill_(float(gloss.item()))
                self._cand_perimg_loss.fill_(float(ploss.item()))
                self._cand_spread.fill_(float(spread.item()))
                self._cand_nctx.fill_(float(n_ctx))
                self._cand_ntgt.fill_(float(n_tgt))

                # Residual affinity: drop the position-conditional (stationary)
                # mode so A reflects per-image content, then re-test on the SAME
                # candidates. gap_resid >> gap => structure was hidden by the
                # stationary mode (residualize before building A).
                pos_mean = ema_full.mean(dim=0, keepdim=True)                 # (1,N,D)
                z_res = F.normalize(
                    (ema_full - pos_mean).float(), dim=-1, eps=self.norm_eps
                )
                y_r = torch.einsum("bnd,sd->bns", z_res, R)
                yn_r = (y_r * y_r).sum(dim=-1)
                gram_r = torch.einsum("bns,bms->bnm", y_r, y_r)
                dsq_r = (
                    yn_r.unsqueeze(2) + yn_r.unsqueeze(1) - 2.0 * gram_r
                ).clamp(min=0.0)
                sig_r = torch.nanmedian(
                    dsq_r.masked_fill(eye_mask, float("nan")).flatten(1), dim=-1
                ).values
                sig_r = (sigma_rho * sig_r).clamp(min=self.sigma_floor)
                A_r = torch.exp(-dsq_r / (2.0 * sig_r.view(B, 1, 1)))
                A_r = A_r.masked_fill(eye_mask, 0.0)
                Z_r = A_r.sum(dim=1) + self.eps
                gap_r, frac_r, gloss_r, _, spread_r = _cand_metrics(
                    _cand_cost(A_r, Z_r)
                )
                self._cand_gap_resid.fill_(float(gap_r.item()))
                self._cand_argmin_frac_resid.fill_(float(frac_r.item()))
                self._cand_global_loss_resid.fill_(float(gloss_r.item()))
                self._cand_spread_resid.fill_(float(spread_r.item()))

                # Difficulty cost: predict each patch's sketched feature from the
                # candidate's CONTEXT via NW kernel regression (reuse A); targets
                # should be HARD to predict. cost = -mean_{j in tgt} ||y_j - pred||^2
                # (lower cost = harder = better). In sketch space (S<<D) and on a
                # capped image subset to bound the (K,Bd,N,S) intermediate.
                Bd = min(B, 128)
                A_d = A[:Bd]                                                  # (Bd,N,N)
                y_d = y[:Bd]                                                  # (Bd,N,S)
                # numer[k,b,j,s] = sum_i pc[k,i] A[b,i,j] y[b,i,s]
                Gky = pc_cand.unsqueeze(1).unsqueeze(-1) * y_d.unsqueeze(0)   # (K,Bd,N,S)
                numer = torch.einsum("kbis,bij->kbjs", Gky, A_d)             # (K,Bd,N,S)
                denom = torch.einsum("ki,bij->kbj", pc_cand, A_d)            # (K,Bd,N)
                pred = numer / (denom.unsqueeze(-1) + self.eps)              # (K,Bd,N,S)
                diff = ((y_d.unsqueeze(0) - pred) ** 2).sum(dim=-1)          # (K,Bd,N)
                tgt_mass = pt_cand.sum(dim=-1).clamp(min=1.0)                # (K,)
                diff_tgt = torch.einsum("kj,kbj->kb", pt_cand, diff) / tgt_mass.unsqueeze(1)
                cost_diff = -diff_tgt                                        # (K,Bd)
                gap_d, frac_d, gloss_d, _, spread_d = _cand_metrics(cost_diff)
                self._cand_gap_diff.fill_(float(gap_d.item()))
                self._cand_argmin_frac_diff.fill_(float(frac_d.item()))
                self._cand_global_loss_diff.fill_(float(gloss_d.item()))
                self._cand_spread_diff.fill_(float(spread_d.item()))
        self._step_counter.add_(1)

        log_max = math.log1p(self.alpha)
        logs = {
            "affinity/loss": float(loss.detach().item()),
            "affinity/U_ctx": float(U_ctx.detach().item()),
            "affinity/U_tgt": float(U_tgt.detach().item()),
            "affinity/C_act": float(C_act.detach().item()),
            "affinity/U_ctx_norm": float(U_ctx.detach().item()) / max(log_max, 1e-12),
            "affinity/U_tgt_norm": float(U_tgt.detach().item()) / max(log_max, 1e-12),
            "affinity/sigma_sq_mean": float(sigma_sq.detach().mean().item()),
            "affinity/sigma_sq_min": float(sigma_sq.detach().amin().item()),
            "affinity/A_mean_offdiag": float(
                A.detach().sum().item() / max(B * N * (N - 1), 1)
            ),
            "affinity/c_mean": float(c.detach().mean().item()),
            "affinity/t_mean": float(t.detach().mean().item()),
            "affinity/c_max_per_image_mean": float(c.detach().amax(dim=-1).mean().item()),
            "affinity/t_max_per_image_mean": float(t.detach().amax(dim=-1).mean().item()),
            "affinity/active_coverage_mean": float((c + t).detach().mean().item()),
            "affinity/Z_mean": float(Z.detach().mean().item()),
            "affinity/Z_min_mean": float(Z.detach().amin(dim=-1).mean().item()),
            "affinity/eff_rank_pr_mean": float(self._eff_rank_mean.item()),
            "affinity/eff_rank_pr_min": float(self._eff_rank_min.item()),
            "affinity/cand_gap": float(self._cand_gap.item()),
            "affinity/cand_global_loss": float(self._cand_global_loss.item()),
            "affinity/cand_perimg_loss": float(self._cand_perimg_loss.item()),
            "affinity/cand_argmin_frac": float(self._cand_argmin_frac.item()),
            "affinity/cand_spread": float(self._cand_spread.item()),
            "affinity/cand_gap_resid": float(self._cand_gap_resid.item()),
            "affinity/cand_argmin_frac_resid": float(self._cand_argmin_frac_resid.item()),
            "affinity/cand_global_loss_resid": float(self._cand_global_loss_resid.item()),
            "affinity/cand_spread_resid": float(self._cand_spread_resid.item()),
            "affinity/cand_gap_diff": float(self._cand_gap_diff.item()),
            "affinity/cand_argmin_frac_diff": float(self._cand_argmin_frac_diff.item()),
            "affinity/cand_global_loss_diff": float(self._cand_global_loss_diff.item()),
            "affinity/cand_spread_diff": float(self._cand_spread_diff.item()),
            "affinity/cand_nctx": float(self._cand_nctx.item()),
            "affinity/cand_ntgt": float(self._cand_ntgt.item()),
            "affinity/coupling": coupling,
            "affinity/alpha": self.alpha,
            "affinity/w_tgt": self.w_tgt,
            "affinity/gamma": gamma,
            "affinity/gamma_start": self.gamma_start,
            "affinity/gamma_end": self.gamma_end,
            "affinity/gamma_schedule_id": {
                "constant": 0.0,
                "linear": 1.0,
                "cosine": 2.0,
            }[self.gamma_schedule],
            "affinity/sigma_rho": sigma_rho,
            "affinity/sigma_rho_start": self.sigma_rho_start,
            "affinity/sigma_rho_end": self.sigma_rho_end,
            "affinity/sigma_rho_schedule_id": {
                "constant": 0.0,
                "linear": 1.0,
                "cosine": 2.0,
            }[self.sigma_rho_schedule],
        }
        return loss, logs


# ------------------------------------------------------------------
# Hybrid feature-position affinity novelty term
# ------------------------------------------------------------------
# Ã_ij = A_feat_ij × exp(−‖pos_i − pos_j‖² / 2σ_p²)
#
# The spatial factor provides non-zero intra-cluster gradient: coverage from
# a single ctx patch saturates its spatial neighbourhood only, not the whole
# semantic cluster, so the masker must spread ctx across the cluster's spatial
# extent to maximise U_ctx. This prevents the cluster-exhaustion death spiral
# that occurs in NegAffinityNoveltyTerm when the encoder is already trained.
# ------------------------------------------------------------------

class NegHybridAffinityNoveltyTerm(MaskerTerm):
    name = "neg_hybrid_affinity_novelty"

    def __init__(
        self,
        num_patches: int,
        alpha: float = 4.0,
        w_tgt: float = 1.0,
        gamma: float | Sequence[float] = 0.5,
        gamma_schedule: str = "constant",
        sketch_dim: int = 64,
        eps: float = 1e-4,
        sigma_rho: float | Sequence[float] = 1.0,
        sigma_rho_schedule: str = "constant",
        sigma_floor: float = 1e-6,
        sigma_pos: float | Sequence[float] = 2.0,
        sigma_pos_schedule: str = "constant",
        sketch_seed: int = 0,
        norm_eps: float = 1e-8,
        eff_rank_every_steps: int = 50,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.w_tgt = float(w_tgt)
        if isinstance(gamma, Sequence) and not isinstance(gamma, (str, bytes)):
            if len(gamma) != 2:
                raise ValueError(
                    "neg_hybrid_affinity_novelty.gamma must be a scalar or a [start, end] pair"
                )
            gamma_start, gamma_end = float(gamma[0]), float(gamma[1])
        else:
            gamma_start = gamma_end = float(gamma)
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.gamma_schedule = str(gamma_schedule).lower()
        if self.gamma_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                "neg_hybrid_affinity_novelty.gamma_schedule must be "
                "'constant', 'linear', or 'cosine'"
            )
        self.sketch_dim = int(sketch_dim)
        if self.sketch_dim <= 0:
            raise ValueError("neg_hybrid_affinity_novelty.sketch_dim must be > 0")
        self.eps = float(eps)
        if isinstance(sigma_rho, Sequence) and not isinstance(sigma_rho, (str, bytes)):
            if len(sigma_rho) != 2:
                raise ValueError(
                    "neg_hybrid_affinity_novelty.sigma_rho must be a scalar or a [start, end] pair"
                )
            sigma_rho_start, sigma_rho_end = float(sigma_rho[0]), float(sigma_rho[1])
        else:
            sigma_rho_start = sigma_rho_end = float(sigma_rho)
        if sigma_rho_start <= 0.0 or sigma_rho_end <= 0.0:
            raise ValueError("neg_hybrid_affinity_novelty.sigma_rho endpoints must be > 0")
        self.sigma_rho_start = sigma_rho_start
        self.sigma_rho_end = sigma_rho_end
        self.sigma_rho_schedule = str(sigma_rho_schedule).lower()
        if self.sigma_rho_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                "neg_hybrid_affinity_novelty.sigma_rho_schedule must be "
                "'constant', 'linear', or 'cosine'"
            )
        self.sigma_floor = float(sigma_floor)
        if isinstance(sigma_pos, Sequence) and not isinstance(sigma_pos, (str, bytes)):
            if len(sigma_pos) != 2:
                raise ValueError(
                    "neg_hybrid_affinity_novelty.sigma_pos must be a scalar or a [start, end] pair"
                )
            sigma_pos_start, sigma_pos_end = float(sigma_pos[0]), float(sigma_pos[1])
        else:
            sigma_pos_start = sigma_pos_end = float(sigma_pos)
        if sigma_pos_start <= 0.0 or sigma_pos_end <= 0.0:
            raise ValueError("neg_hybrid_affinity_novelty.sigma_pos endpoints must be > 0")
        self.sigma_pos_start = sigma_pos_start
        self.sigma_pos_end = sigma_pos_end
        self.sigma_pos_schedule = str(sigma_pos_schedule).lower()
        if self.sigma_pos_schedule not in ("constant", "linear", "cosine"):
            raise ValueError(
                "neg_hybrid_affinity_novelty.sigma_pos_schedule must be "
                "'constant', 'linear', or 'cosine'"
            )
        self.sketch_seed = int(sketch_seed)
        self.norm_eps = float(norm_eps)
        self.eff_rank_every_steps = max(int(eff_rank_every_steps), 1)
        self.register_buffer("_sketch_R", torch.empty(0), persistent=False)
        self.register_buffer(
            "_step_counter", torch.tensor(0, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "_eff_rank_mean", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )
        self.register_buffer(
            "_eff_rank_min", torch.tensor(0.0, dtype=torch.float32), persistent=True
        )

        # Precompute squared inter-patch distances on the 2D patch grid.
        # A_spat is recomputed each forward at the scheduled sigma_pos value
        # so we only store the fixed geometry here.
        grid_size = int(math.isqrt(int(num_patches)))
        if grid_size * grid_size != int(num_patches):
            raise ValueError(
                f"neg_hybrid_affinity_novelty requires a square patch grid; "
                f"num_patches={num_patches} is not a perfect square"
            )
        rows = torch.arange(grid_size, dtype=torch.float32)
        cols = torch.arange(grid_size, dtype=torch.float32)
        gr, gc = torch.meshgrid(rows, cols, indexing="ij")
        pos = torch.stack([gr.reshape(-1), gc.reshape(-1)], dim=-1)  # (N, 2)
        d_pos_sq = ((pos.unsqueeze(0) - pos.unsqueeze(1)) ** 2).sum(-1)  # (N, N)
        self.register_buffer("_d_pos_sq", d_pos_sq, persistent=False)

    def _get_sketch(self, dim: int, device: torch.device) -> torch.Tensor:
        if (
            self._sketch_R.numel() == 0
            or self._sketch_R.shape != (self.sketch_dim, dim)
            or self._sketch_R.device != device
        ):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.sketch_seed)
            R = torch.randn(
                self.sketch_dim, dim, generator=gen, dtype=torch.float32
            )
            R = R * (self.sketch_dim ** -0.5)
            self._sketch_R = R.to(device=device)
        return self._sketch_R

    def _schedule(
        self,
        start: float,
        end: float,
        kind: str,
        epoch: int | None,
        total_epochs: int | None,
    ) -> float:
        if start == end or kind == "constant":
            return start
        if epoch is None or total_epochs is None or total_epochs <= 1:
            return start
        progress = min(max(float(epoch) / float(total_epochs - 1), 0.0), 1.0)
        blend = progress if kind == "linear" else 0.5 * (1.0 - math.cos(math.pi * progress))
        return start + (end - start) * blend

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        del p_ign

        z_hat = F.normalize(ema_full.float(), dim=-1, eps=self.norm_eps)   # (B, N, D)
        B, N, D = z_hat.shape
        device = z_hat.device
        epoch = kw.get("epoch")
        total_epochs = kw.get("total_epochs")
        ep = None if epoch is None else int(epoch)
        te = None if total_epochs is None else int(total_epochs)

        sigma_rho = self._schedule(
            self.sigma_rho_start, self.sigma_rho_end, self.sigma_rho_schedule, ep, te
        )
        gamma = self._schedule(
            self.gamma_start, self.gamma_end, self.gamma_schedule, ep, te
        )
        sigma_pos = self._schedule(
            self.sigma_pos_start, self.sigma_pos_end, self.sigma_pos_schedule, ep, te
        )

        # Feature affinity via random sketch
        R = self._get_sketch(D, device)                                    # (S, D)
        y = torch.einsum("bnd,sd->bns", z_hat, R)                          # (B, N, S)
        y_norm = (y * y).sum(dim=-1)                                        # (B, N)
        gram = torch.einsum("bns,bms->bnm", y, y)                          # (B, N, N)
        d_sq = (y_norm.unsqueeze(2) + y_norm.unsqueeze(1) - 2.0 * gram).clamp(min=0.0)

        eye_mask = torch.eye(N, dtype=torch.bool, device=device)
        d_for_med = d_sq.masked_fill(eye_mask, float("nan"))
        sigma_sq = torch.nanmedian(d_for_med.flatten(1), dim=-1).values    # (B,)
        sigma_sq = (sigma_rho * sigma_sq).clamp(min=self.sigma_floor)

        A_feat = torch.exp(-d_sq / (2.0 * sigma_sq.view(B, 1, 1)))
        A_feat = A_feat.masked_fill(eye_mask, 0.0)

        # Column-normalised spatial affinity.
        # Each column sums to 1 (off-diagonal) so no patch has a larger
        # total spatial pull just because it sits in the center of the grid.
        A_spat_raw = torch.exp(
            -self._d_pos_sq / (2.0 * sigma_pos ** 2)
        )                                                                   # (N, N)
        col_sum = A_spat_raw.sum(dim=0) - 1.0                              # subtract diagonal=1; (N,)
        A_spat_norm = A_spat_raw / (col_sum.unsqueeze(0) + self.eps)       # (N, N)
        A_spat_norm = A_spat_norm.masked_fill(eye_mask, 0.0)

        # Hybrid: element-wise product of feature and normalised spatial affinities.
        A = A_feat * A_spat_norm                                            # (B, N, N)

        Z = A.sum(dim=1) + self.eps                                         # (B, N)
        ctx_num = torch.einsum("bi,bij->bj", p_ctx.float(), A)
        tgt_num = torch.einsum("bi,bij->bj", p_tgt.float(), A)
        c = ctx_num / Z
        t = tgt_num / Z

        U_ctx = torch.log1p(self.alpha * c).mean()
        ratio = t / (self.eps + c + t)
        U_tgt = torch.log1p(self.alpha * ratio).mean()
        C_act = (p_ctx + p_tgt).mean()

        coupling = gamma * math.log1p(self.alpha)
        loss = -U_ctx - self.w_tgt * U_tgt + coupling * C_act

        step = int(self._step_counter.item())
        if step % self.eff_rank_every_steps == 0:
            with torch.no_grad():
                frob_sq = (A * A).sum(dim=(-1, -2))
                A_sq = torch.bmm(A, A)
                A_sq_frob_sq = (A_sq * A_sq).sum(dim=(-1, -2))
                eff_rank = frob_sq.square() / A_sq_frob_sq.clamp(min=1e-12)
                self._eff_rank_mean.fill_(float(eff_rank.mean().item()))
                self._eff_rank_min.fill_(float(eff_rank.amin().item()))
        self._step_counter.add_(1)

        log_max = math.log1p(self.alpha)
        N_offdiag = N * (N - 1)
        logs = {
            "haff/loss": float(loss.detach().item()),
            "haff/U_ctx": float(U_ctx.detach().item()),
            "haff/U_tgt": float(U_tgt.detach().item()),
            "haff/C_act": float(C_act.detach().item()),
            "haff/U_ctx_norm": float(U_ctx.detach().item()) / max(log_max, 1e-12),
            "haff/U_tgt_norm": float(U_tgt.detach().item()) / max(log_max, 1e-12),
            "haff/sigma_sq_mean": float(sigma_sq.detach().mean().item()),
            "haff/sigma_sq_min": float(sigma_sq.detach().amin().item()),
            "haff/A_mean_offdiag": float(
                A.detach().sum().item() / max(B * N_offdiag, 1)
            ),
            "haff/A_feat_mean_offdiag": float(
                A_feat.detach().sum().item() / max(B * N_offdiag, 1)
            ),
            "haff/A_spat_norm_col_mean": float(
                A_spat_norm.sum().item() / max(N_offdiag, 1)
            ),
            "haff/c_mean": float(c.detach().mean().item()),
            "haff/t_mean": float(t.detach().mean().item()),
            "haff/c_max_per_image_mean": float(c.detach().amax(dim=-1).mean().item()),
            "haff/t_max_per_image_mean": float(t.detach().amax(dim=-1).mean().item()),
            "haff/active_coverage_mean": float((c + t).detach().mean().item()),
            "haff/Z_mean": float(Z.detach().mean().item()),
            "haff/Z_min_mean": float(Z.detach().amin(dim=-1).mean().item()),
            "haff/eff_rank_pr_mean": float(self._eff_rank_mean.item()),
            "haff/eff_rank_pr_min": float(self._eff_rank_min.item()),
            "haff/coupling": coupling,
            "haff/alpha": self.alpha,
            "haff/w_tgt": self.w_tgt,
            "haff/gamma": gamma,
            "haff/sigma_rho": sigma_rho,
            "haff/sigma_pos": sigma_pos,
        }
        return loss, logs


# ------------------------------------------------------------------
# −predictability — role term: targets = patches HARD to predict from the
# (soft) context. Unlike coverage/cos-surprise, this has a real mask×image
# interaction (the candidate diagnostic measured the difficulty landscape as
# ~62× more mask-discriminating and ~94× more per-image than coverage).
# ------------------------------------------------------------------

class NegPredictabilityTerm(MaskerTerm):
    """
    Reuse the cosine-RBF affinity ``A`` (median bandwidth, JL sketch) as a
    Nadaraya–Watson regressor: predict every patch's feature from the soft
    context, ``pred_j = Σ_i p_ctx_i A_ij z_i / Σ_i p_ctx_i A_ij``. The role
    reward makes targets the patches that are *hard* to predict from context:
    ``reward = mean_{j} p_tgt_j · ‖z_j − pred_j‖²`` (loss = −reward).

    ``residualize`` (default on) subtracts each position's batch-mean difficulty
    (detached). This (a) isolates "hard *in this image*" from "universally hard
    at this position" — the per-image signal we want — and (b) self-guards the
    cos-surprise ctx-collapse mode: emptying the context makes ``pred→0`` and the
    difficulty ≈‖z_j‖²≈1 *uniformly*, so the per-position-centered reward → 0
    (shrinking ctx buys nothing). It owns the ctx-vs-tgt axis; pair it with
    affinity-novelty (w_tgt=0) for the active-set / coverage axis.
    """

    name = "neg_predictability"

    def __init__(
        self,
        sketch_dim: int = 64,
        sigma_rho: float = 1.0,
        sigma_floor: float = 1e-6,
        eps: float = 1e-4,
        sketch_seed: int = 0,
        norm_eps: float = 1e-8,
        residualize: bool = True,
    ):
        super().__init__()
        self.sketch_dim = int(sketch_dim)
        if self.sketch_dim <= 0:
            raise ValueError("neg_predictability.sketch_dim must be > 0")
        self.sigma_rho = float(sigma_rho)
        if self.sigma_rho <= 0.0:
            raise ValueError("neg_predictability.sigma_rho must be > 0")
        self.sigma_floor = float(sigma_floor)
        self.eps = float(eps)
        self.sketch_seed = int(sketch_seed)
        self.norm_eps = float(norm_eps)
        self.residualize = bool(residualize)
        self.register_buffer("_sketch_R", torch.empty(0), persistent=False)

    def _get_sketch(self, dim: int, device: torch.device) -> torch.Tensor:
        if (
            self._sketch_R.numel() == 0
            or self._sketch_R.shape != (self.sketch_dim, dim)
            or self._sketch_R.device != device
        ):
            gen = torch.Generator(device="cpu")
            gen.manual_seed(self.sketch_seed)
            R = torch.randn(self.sketch_dim, dim, generator=gen, dtype=torch.float32)
            R = R * (self.sketch_dim ** -0.5)
            self._sketch_R = R.to(device=device)
        return self._sketch_R

    def forward(self, *, p_ctx, p_tgt, p_ign, ema_full, **kw):
        del p_ign

        z = F.normalize(ema_full.float(), dim=-1, eps=self.norm_eps)        # (B,N,D)
        B, N, D = z.shape
        device = z.device

        # Cosine-RBF affinity from the JL sketch (matches affinity-novelty).
        R = self._get_sketch(D, device)                                    # (S,D)
        y = torch.einsum("bnd,sd->bns", z, R)                              # (B,N,S)
        y_norm = (y * y).sum(dim=-1)                                       # (B,N)
        gram = torch.einsum("bns,bms->bnm", y, y)                          # (B,N,N)
        d_sq = (
            y_norm.unsqueeze(2) + y_norm.unsqueeze(1) - 2.0 * gram
        ).clamp(min=0.0)
        eye_mask = torch.eye(N, dtype=torch.bool, device=device)
        sigma_sq = torch.nanmedian(
            d_sq.masked_fill(eye_mask, float("nan")).flatten(1), dim=-1
        ).values                                                           # (B,)
        sigma_sq = (self.sigma_rho * sigma_sq).clamp(min=self.sigma_floor)
        A = torch.exp(-d_sq / (2.0 * sigma_sq.view(B, 1, 1)))
        A = A.masked_fill(eye_mask, 0.0)                                   # (B,N,N)

        # Nadaraya–Watson regression of each patch from the SOFT context.
        pc = p_ctx.float()
        denom = torch.einsum("bi,bij->bj", pc, A)                          # (B,N)
        W = pc.unsqueeze(2) * A                                            # (B,N,N) i,j
        num = torch.einsum("bij,bid->bjd", W, z)                           # (B,N,D)
        pred = num / (denom.unsqueeze(-1) + self.eps)                      # (B,N,D)
        diff = ((z - pred) ** 2).sum(dim=-1)                               # (B,N)

        if self.residualize:
            # Per-position batch-mean baseline (detached control variate):
            # isolates image-specific surprise and neutralizes ctx-collapse.
            diff = diff - diff.mean(dim=0, keepdim=True).detach()

        p_tgt_sum = p_tgt.sum(dim=-1).clamp(min=1.0)                       # (B,)
        reward = ((p_tgt * diff).sum(dim=-1) / p_tgt_sum).mean()
        loss = -reward

        logs = {
            "predict/loss": float(loss.detach().item()),
            "predict/reward": float(reward.detach().item()),
            "predict/diff_mean": float(diff.detach().mean().item()),
            "predict/denom_mean": float(denom.detach().mean().item()),
            "predict/sigma_sq_mean": float(sigma_sq.detach().mean().item()),
            "predict/residualize": 1.0 if self.residualize else 0.0,
        }
        return loss, logs


# ------------------------------------------------------------------

TERM_REGISTRY: dict[str, type[MaskerTerm]] = {
    "H_cond": HCondTerm,
    "neg_H_marg": NegHMargTerm,
    "neg_surprise": NegSurpriseTerm,
    "neg_cos_surprise": NegCosSurpriseTerm,
    "neg_symmetric_cos_surprise": NegSymmetricCosSurpriseTerm,
    "neg_sketched_orthogonal_cos_surprise": NegSketchedOrthogonalCosSurpriseTerm,
    "neg_prompt_contrast": NegPromptContrastTerm,
    "neg_logdet_diversity": NegLogDetDiversityTerm,
    "neg_support_logdet_diversity": NegSupportLogDetDiversityTerm,
    "neg_signed_support_logdet": NegSignedSupportLogDetTerm,
    "neg_mass_logdet_diversity": NegMassLogDetDiversityTerm,
    "neg_centroid_dist": NegCentroidDistTerm,
    "floor_penalty": FloorPenaltyTerm,
    "ignore_tax": IgnoreTaxTerm,
    "context_rate": ContextRateTerm,
    "target_rate": TargetRateTerm,
    "nway_cross_surprise": NWayCrossSurpriseTerm,
    "nway_full_cross_surprise": NWayFullCrossSurpriseTerm,
    "nway_neg_H_marg": NWayNegHMargTerm,
    "nway_kl_marg": NWayKLMargTerm,
    "nway_floor_penalty": FloorPenaltyTerm,
    "role_alive": RoleAliveTerm,
    "nway_progressive_kl": NWayProgressiveKLTerm,
    "neg_affinity_novelty": NegAffinityNoveltyTerm,
    "neg_hybrid_affinity_novelty": NegHybridAffinityNoveltyTerm,
    "neg_predictability": NegPredictabilityTerm,
}
