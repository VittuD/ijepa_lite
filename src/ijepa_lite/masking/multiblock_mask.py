# FILE: src/ijepa_lite/masking/multiblock_mask.py
"""
Multi-block mask generator faithful to the original I-JEPA design.

Key design choices (matching facebook/ijepa):
  - Target block (h, w) is sampled **once per batch** — all images share the
    same block dimensions, only the placement varies.
  - Each target block IS the full rectangle (no subsampling within it).
  - K = h * w varies across batches but is fixed within a batch → (B, M, K)
    is always a regular tensor.
  - Context block is sampled per-image, excluding target patches.
  - When allow_overlap=False and the `acceptable_regions` constraint removes
    patches from a block, blocks are truncated to min_keep across the batch
    so all images share the same K (mirrors the original collation logic).
"""
from __future__ import annotations

import math
import random

import torch

from ijepa_lite.masking.base import CollateMasker, MaskOutput


class MultiBlockMaskGenerator(CollateMasker):
    """
    CPU-side multi-block mask generator intended to run in DataLoader workers.

    Returns:
        MaskOutput with:
          context_idx: LongTensor (B, Nctx)
          target_idx:  LongTensor (B, M, K)   — K varies across calls
          context_soft / target_soft: None
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        target_ratio: float,
        context_ratio: float,
        num_target_blocks: int,
        # Target block sampling.
        tgt_min_scale: float,
        tgt_max_scale: float,
        tgt_min_aspect: float,
        tgt_max_aspect: float,
        # Context block sampling.
        ctx_min_scale: float = 0.85,
        ctx_max_scale: float = 1.00,
        ctx_min_aspect: float = 0.75,
        ctx_max_aspect: float = 1.50,
        allow_overlap: bool = False,
        min_keep: int = 10,
        max_resample_tries: int = 20,
    ):
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid = self.image_size // self.patch_size
        self.num_patches = self.grid * self.grid

        self.num_target_blocks = int(num_target_blocks)

        self.tgt_min_scale = float(tgt_min_scale)
        self.tgt_max_scale = float(tgt_max_scale)
        self.tgt_min_aspect = float(tgt_min_aspect)
        self.tgt_max_aspect = float(tgt_max_aspect)

        self.ctx_min_scale = float(ctx_min_scale)
        self.ctx_max_scale = float(ctx_max_scale)
        self.ctx_min_aspect = float(ctx_min_aspect)
        self.ctx_max_aspect = float(ctx_max_aspect)

        self.allow_overlap = bool(allow_overlap)
        self.min_keep = max(1, int(min_keep))
        self.max_resample_tries = int(max_resample_tries)

        self.nctx = max(1, int(round(self.num_patches * float(context_ratio))))

    # ------------------------------------------------------------------
    # Rectangle sampling helpers
    # ------------------------------------------------------------------

    def _sample_rect_size(
        self,
        min_scale: float,
        max_scale: float,
        min_aspect: float,
        max_aspect: float,
    ) -> tuple[int, int]:
        """Sample (h, w) for a rectangle in patch-grid coordinates."""
        g = self.grid
        n = self.num_patches

        scale = random.uniform(min_scale, max_scale)
        area = max(1, int(round(scale * n)))

        log_aspect = random.uniform(math.log(min_aspect), math.log(max_aspect))
        aspect = math.exp(log_aspect)

        h = max(1, min(g, int(round(math.sqrt(area * aspect)))))
        w = max(1, min(g, int(round(math.sqrt(area / aspect)))))
        return h, w

    def _sample_rect_indices(self, h: int, w: int) -> list[int]:
        """Return flat patch indices for a randomly placed h-by-w rectangle (sorted)."""
        g = self.grid
        top = random.randint(0, max(0, g - h))
        left = random.randint(0, max(0, g - w))
        return sorted((top + r) * g + (left + c) for r in range(h) for c in range(w))

    # ------------------------------------------------------------------
    # Per-image sampling (given pre-sampled block sizes)
    # ------------------------------------------------------------------

    def _sample_one(
        self,
        tgt_sizes: list[tuple[int, int]],
    ) -> tuple[list[list[int]], list[int]]:
        """
        Generate target blocks and context for one image.

        Parameters
        ----------
        tgt_sizes : list of (h, w) per target block — shared across batch.

        Returns
        -------
        tgt_blocks : list of M lists (variable length per block)
        ctx        : list of length ≤ Nctx
        """
        n = self.num_patches

        occupied: set[int] = set()
        tgt_blocks: list[list[int]] = []

        # 1) Target blocks — use the full rectangle.
        for h, w in tgt_sizes:
            for _attempt in range(self.max_resample_tries + 1):
                rect = self._sample_rect_indices(h, w)

                if self.allow_overlap:
                    picked = rect
                    break

                picked = [i for i in rect if i not in occupied]
                if picked:
                    break
            else:
                # Fallback: use whatever non-occupied indices we got
                if not picked:
                    remaining = sorted(set(range(n)) - occupied)
                    picked = remaining[:max(1, h * w)]

            tgt_blocks.append(picked)

            if not self.allow_overlap:
                occupied.update(picked)

        # 2) Context rectangle minus targets.
        ctx_candidates: list[int] = []

        for _attempt in range(self.max_resample_tries + 1):
            h, w = self._sample_rect_size(
                self.ctx_min_scale,
                self.ctx_max_scale,
                self.ctx_min_aspect,
                self.ctx_max_aspect,
            )
            rect = self._sample_rect_indices(h, w)

            if self.allow_overlap:
                ctx_candidates = rect
            else:
                ctx_candidates = [i for i in rect if i not in occupied]

            if ctx_candidates:
                break
        else:
            if not ctx_candidates:
                ctx_candidates = sorted(set(range(n)) - occupied)

        nctx = min(self.nctx, len(ctx_candidates))
        if nctx < len(ctx_candidates):
            ctx = sorted(random.sample(ctx_candidates, nctx))
        else:
            ctx = ctx_candidates

        return tgt_blocks, ctx

    # ------------------------------------------------------------------
    # Batch entry point — called by IJEPACollate
    # ------------------------------------------------------------------

    def __call__(self, batch_size: int) -> MaskOutput:
        """
        Generate masks for a batch.

        Block sizes are sampled once (shared across all images in the batch),
        matching the original I-JEPA design. K = h*w per block varies across
        calls but is fixed within a batch.

        When allow_overlap=False, the acceptable-region constraint can reduce
        block sizes differently per image. We truncate to min_keep per block
        (same as original I-JEPA collation) so the tensor stays regular.
        """
        B = int(batch_size)
        M = self.num_target_blocks

        # Sample ONE block size for ALL M target blocks (original I-JEPA design).
        h, w = self._sample_rect_size(
            self.tgt_min_scale,
            self.tgt_max_scale,
            self.tgt_min_aspect,
            self.tgt_max_aspect,
        )
        tgt_sizes = [(h, w)] * M

        all_tgt: list[list[list[int]]] = []
        all_ctx: list[list[int]] = []

        for _ in range(B):
            tgt_blocks, ctx = self._sample_one(tgt_sizes)
            all_tgt.append(tgt_blocks)
            all_ctx.append(ctx)

        # Truncate to a single K across ALL blocks and images so the
        # (B, M, K) tensor is regular.  Floor at self.min_keep (original
        # I-JEPA uses min_keep=10).
        K = min(
            len(all_tgt[b][m]) for b in range(B) for m in range(M)
        )
        K = max(K, self.min_keep)
        for b in range(B):
            for m in range(M):
                block = all_tgt[b][m]
                if len(block) > K:
                    all_tgt[b][m] = block[:K]
                elif len(block) < K:
                    # Pad with random non-occupied patches to reach min_keep
                    have = set(block)
                    pool = [i for i in range(self.num_patches) if i not in have]
                    random.shuffle(pool)
                    block = block + pool[:K - len(block)]
                    all_tgt[b][m] = block[:K]

        # Truncate context to min across batch.
        min_ctx = min(len(all_ctx[b]) for b in range(B))
        for b in range(B):
            all_ctx[b] = all_ctx[b][:min_ctx]

        tgt_tensor = torch.tensor(all_tgt, dtype=torch.long)   # (B, M, K)
        ctx_tensor = torch.tensor(all_ctx, dtype=torch.long)    # (B, Nctx)

        return MaskOutput(
            context_idx=ctx_tensor,
            target_idx=tgt_tensor,
        )
