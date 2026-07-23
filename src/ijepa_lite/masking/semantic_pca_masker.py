from __future__ import annotations

from typing import Optional

import torch

from ijepa_lite.masking.base import LatentMasker, MaskOutput, MaskPartition
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
        pca_dim: int = 1,
        power_iterations: int = 4,
        split_mode: str = "band",
        context_mode: str = "complement",
        normalize_tokens: bool = True,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.ntgt = max(1, int(round(self.num_patches * float(target_ratio))))
        self.nctx = max(1, int(round(self.num_patches * float(context_ratio))))
        self.requested_pca_dim = max(1, int(pca_dim))
        self.pca_dim = 1
        self.power_iterations = max(1, int(power_iterations))
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

        semantic_score, explained_var = self._batched_pc1_scores(ema_full)
        order = semantic_score.argsort(dim=1)
        target_idx = self._select_target_batched(order)
        context_idx = self._select_context_batched(order, target_idx)

        return MaskOutput(
            partition=MaskPartition(
                context_idx=context_idx,
                target_idx=target_idx,
            ),
            diagnostics={
                "pca_dim": float(self.pca_dim),
                "pca_requested_dim": float(self.requested_pca_dim),
                "pca_power_iterations": float(self.power_iterations),
                "pca_explained_var_mean": explained_var.mean().detach(),
                "pca_explained_var_top1": explained_var.mean().detach(),
                "target_ratio_actual": float(target_idx.shape[1] / N),
                "context_ratio_actual": float(context_idx.shape[1] / N),
            },
        )

    def _batched_pc1_scores(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Approximate PC1 with batched power iteration using only matrix-vector
        # products. No eigensolver is used, so AMP/bf16 is allowed here.
        x = x.detach()
        x = x - x.mean(dim=1, keepdim=True)
        if self.normalize_tokens:
            x = x / x.norm(dim=-1, keepdim=True).clamp_min(self.eps)

        B, _N, D = x.shape
        direction = torch.randn(B, D, 1, device=x.device, dtype=x.dtype)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(self.eps)

        for _ in range(self.power_iterations):
            scores = torch.bmm(x, direction)
            direction = torch.bmm(x.transpose(1, 2), scores)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(self.eps)

        scores = torch.bmm(x, direction).squeeze(-1)
        sign = torch.empty(B, 1, device=x.device, dtype=x.dtype).bernoulli_()
        scores = scores * sign.mul(2.0).sub(1.0)

        pc1_var = scores.float().square().sum(dim=1)
        total_var = x.float().square().sum(dim=(1, 2)).clamp_min(self.eps)
        explained = pc1_var / total_var
        return scores, explained

    def _select_target_batched(self, order: torch.Tensor) -> torch.Tensor:
        B, N = order.shape
        ntgt = min(self.ntgt, N - 1)

        if self.split_mode == "side":
            low = order[:, :ntgt]
            high = order[:, -ntgt:]
            use_high = torch.empty(B, 1, device=order.device).bernoulli_().bool()
            return torch.where(use_high, high, low)

        max_start = N - ntgt
        starts = torch.randint(max_start + 1, (B, 1), device=order.device)
        offsets = torch.arange(ntgt, device=order.device).unsqueeze(0)
        positions = starts + offsets
        return order.gather(1, positions)

    def _select_context_batched(
        self,
        order: torch.Tensor,
        tgt_idx: torch.Tensor,
    ) -> torch.Tensor:
        B, N = order.shape
        is_tgt = torch.zeros(B, N, dtype=torch.bool, device=order.device)
        is_tgt.scatter_(1, tgt_idx, True)

        available_mask = ~is_tgt.gather(1, order)
        available = order[available_mask].view(B, N - tgt_idx.shape[1])
        nctx = min(self.nctx, available.shape[1])
        if self.context_mode == "complement" or nctx == available.shape[1]:
            return available[:, :nctx]

        # Balanced keeps context semantically broad by taking evenly spaced
        # points along the same PCA ordering, without introducing per-token
        # salt-and-pepper target sampling.
        positions = torch.linspace(
            0,
            available.shape[1] - 1,
            steps=nctx,
            device=available.device,
        ).round().long()
        return available.gather(1, positions.unsqueeze(0).expand(B, -1))
