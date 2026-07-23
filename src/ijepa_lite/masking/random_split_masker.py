from __future__ import annotations

import math

import torch

from ijepa_lite.masking.base import LatentMasker, MaskOutput, MaskPartition
from ijepa_lite.masking.registry import register


class _BaseRandomSplitMasker(LatentMasker):
    owns_loss: bool = False
    needs_full_tokens: bool = False

    def __init__(self, num_patches: int, upper_half_only: bool = False) -> None:
        super().__init__()
        self.num_patches = int(num_patches)
        self.upper_half_only = bool(upper_half_only)

        if self.num_patches < 2:
            raise ValueError("Random split maskers require at least two patches.")

        grid = int(math.isqrt(self.num_patches))
        if grid * grid != self.num_patches:
            raise ValueError(
                "Random split maskers require a square patch grid. "
                f"Got num_patches={self.num_patches}."
            )
        self.grid_size = grid

        if self.upper_half_only:
            if grid < 2:
                raise ValueError("upper_half_only requires grid_size >= 2.")
            rows = torch.arange(grid).repeat_interleave(grid)
            self.register_buffer(
                "_candidate_idx",
                torch.nonzero(rows < (grid // 2), as_tuple=False).flatten().long(),
                persistent=False,
            )
        else:
            self.register_buffer(
                "_candidate_idx",
                torch.arange(self.num_patches).long(),
                persistent=False,
            )

        n_candidates = int(self._candidate_idx.numel())
        self.nctx = n_candidates // 2
        self.ntgt = n_candidates - self.nctx
        if self.nctx <= 0 or self.ntgt <= 0:
            raise ValueError(
                "Random split maskers need non-empty context and target sets. "
                f"Got nctx={self.nctx}, ntgt={self.ntgt}."
            )

    @torch.no_grad()
    def forward(self, tokens: torch.Tensor, ema_full: torch.Tensor | None = None) -> MaskOutput:
        B = int(tokens.shape[0])
        candidates = self._candidate_idx.to(device=tokens.device)
        n_candidates = int(candidates.numel())

        scores = torch.rand(B, n_candidates, device=tokens.device)
        order = scores.argsort(dim=1)
        shuffled = candidates[order]

        context_idx = shuffled[:, : self.nctx]
        target_idx = shuffled[:, self.nctx :]

        return MaskOutput(
            partition=MaskPartition(
                context_idx=context_idx,
                target_idx=target_idx,
            ),
            diagnostics={
                "random_split_upper_half_only": float(self.upper_half_only),
                "random_split_candidates": float(n_candidates),
                "target_ratio_actual": float(self.ntgt / self.num_patches),
                "context_ratio_actual": float(self.nctx / self.num_patches),
            },
        )


@register("random_split_50")
class RandomSplit50Masker(_BaseRandomSplitMasker):
    """Randomly split all patches into 50% context and 50% target."""

    def __init__(self, num_patches: int) -> None:
        super().__init__(num_patches=num_patches, upper_half_only=False)


@register("upper_half_random_split_50")
class UpperHalfRandomSplit50Masker(_BaseRandomSplitMasker):
    """Randomly split only upper-half patches; lower-half patches are ignored."""

    def __init__(self, num_patches: int) -> None:
        super().__init__(num_patches=num_patches, upper_half_only=True)
