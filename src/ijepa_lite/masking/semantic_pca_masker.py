from __future__ import annotations

import math
from typing import Optional

import torch

from ijepa_lite.masking.base import LatentMasker, MaskOutput
from ijepa_lite.masking.registry import register


@register("semantic_pca")
class SemanticPCAMasker(LatentMasker):
    """
    Algorithmic semantic masker driven by per-image PCA of EMA patch tokens.

    The masker has no learnable parameters. It uses the latent-masker path only
    because it needs full-image encoder tokens, which CPU-side collate maskers
    cannot access.
    """

    owns_loss: bool = False
    needs_full_tokens: bool = True

    def __init__(
        self,
        num_patches: int,
        target_ratio: float,
        context_ratio: float,
        pca_dim: int = 3,
        split_mode: str = "band",
        context_mode: str = "complement",
        normalize_tokens: bool = True,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.ntgt = max(1, int(round(self.num_patches * float(target_ratio))))
        self.nctx = max(1, int(round(self.num_patches * float(context_ratio))))
        self.pca_dim = max(1, int(pca_dim))
        self.normalize_tokens = bool(normalize_tokens)
        self.eps = float(eps)

        if self.ntgt >= self.num_patches:
            raise ValueError(
                "SemanticPCAMasker requires at least one context candidate. "
                f"Got ntgt={self.ntgt}, num_patches={self.num_patches}."
            )

        if split_mode not in {"band", "side"}:
            raise ValueError(
                f"split_mode must be 'band' or 'side', got {split_mode!r}."
            )
        self.split_mode = str(split_mode)

        if context_mode not in {"complement", "balanced"}:
            raise ValueError(
                "context_mode must be 'complement' or 'balanced', "
                f"got {context_mode!r}."
            )
        self.context_mode = str(context_mode)

    @torch.no_grad()
    def forward(
        self,
        tokens: torch.Tensor,
        ema_full: Optional[torch.Tensor] = None,
    ) -> MaskOutput:
        if ema_full is None:
            raise ValueError("SemanticPCAMasker requires ema_full tokens.")

        B, N, _D = ema_full.shape
        if N != self.num_patches:
            raise ValueError(
                f"Expected {self.num_patches} patches, got {N}. "
                "Check model.image_size and model.patch_size."
            )

        ctx_rows: list[torch.Tensor] = []
        tgt_rows: list[torch.Tensor] = []
        explained_rows: list[torch.Tensor] = []

        for b in range(B):
            scores, explained = self._pca_scores(ema_full[b])
            semantic_score = self._project_scores(scores)
            order = semantic_score.argsort()

            tgt_idx = self._select_target(order)
            ctx_idx = self._select_context(order, tgt_idx)

            tgt_rows.append(tgt_idx)
            ctx_rows.append(ctx_idx)
            explained_rows.append(explained)

        target_idx = torch.stack(tgt_rows, dim=0)
        context_idx = torch.stack(ctx_rows, dim=0)
        explained_var = torch.stack(explained_rows, dim=0)

        return MaskOutput(
            context_idx=context_idx,
            target_idx=target_idx,
            aux={
                "pca_dim": float(self.pca_dim),
                "pca_explained_var_mean": explained_var.mean().detach(),
                "pca_explained_var_top1": explained_var[:, 0].mean().detach(),
                "target_ratio_actual": float(target_idx.shape[1] / N),
                "context_ratio_actual": float(context_idx.shape[1] / N),
            },
        )

    def _pca_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # CUDA eigensolvers do not support bf16. AMP can autocast the Gram/cov
        # matmul back to bf16, so keep the whole PCA block explicitly in fp32.
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.detach().to(dtype=torch.float32)
            x = x - x.mean(dim=0, keepdim=True)
            if self.normalize_tokens:
                x = x / x.norm(dim=-1, keepdim=True).clamp_min(self.eps)

            N, D = x.shape
            k = min(self.pca_dim, N - 1, D)
            if k <= 0:
                zeros = x.new_zeros(N, 1)
                explained = x.new_zeros(1)
                return zeros, explained

            if N <= D:
                gram = x @ x.T
                evals, evecs = torch.linalg.eigh(gram)
                top_vals = evals[-k:].flip(0).clamp_min(0.0)
                top_vecs = evecs[:, -k:].flip(1)
                scores = top_vecs * top_vals.sqrt().unsqueeze(0)
            else:
                cov = x.T @ x
                evals, evecs = torch.linalg.eigh(cov)
                top_vals = evals[-k:].flip(0).clamp_min(0.0)
                top_vecs = evecs[:, -k:].flip(1)
                scores = x @ top_vecs

            total_var = evals.clamp_min(0.0).sum().clamp_min(self.eps)
            explained = top_vals / total_var
            return scores, explained

    def _project_scores(self, scores: torch.Tensor) -> torch.Tensor:
        dim = scores.shape[1]
        if dim == 1:
            direction = torch.empty((), device=scores.device).bernoulli_()
            sign = direction.mul(2.0).sub(1.0)
            return scores[:, 0] * sign

        direction = torch.randn(dim, device=scores.device, dtype=scores.dtype)
        direction = direction / direction.norm().clamp_min(self.eps)
        return scores @ direction

    def _select_target(self, order: torch.Tensor) -> torch.Tensor:
        N = order.numel()
        ntgt = min(self.ntgt, N - 1)

        if self.split_mode == "side":
            use_high_side = bool(torch.empty((), device=order.device).bernoulli_().item())
            if use_high_side:
                return order[-ntgt:]
            return order[:ntgt]

        max_start = N - ntgt
        start = int(torch.randint(max_start + 1, (), device=order.device).item())
        return order[start : start + ntgt]

    def _select_context(self, order: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        N = order.numel()
        is_tgt = torch.zeros(N, dtype=torch.bool, device=order.device)
        is_tgt[tgt_idx] = True

        available = order[~is_tgt[order]]
        nctx = min(self.nctx, available.numel())
        if self.context_mode == "complement" or nctx == available.numel():
            return available[:nctx]

        # Balanced keeps context semantically broad by taking evenly spaced
        # points along the same PCA ordering, without introducing per-token
        # salt-and-pepper target sampling.
        positions = torch.linspace(
            0,
            available.numel() - 1,
            steps=nctx,
            device=available.device,
        ).round().long()
        return available[positions]
