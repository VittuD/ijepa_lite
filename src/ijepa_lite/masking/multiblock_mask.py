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
        unclaimed: str = "ignore",
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

        if unclaimed not in ("ignore", "context", "target"):
            raise ValueError(f"unclaimed must be 'ignore', 'context', or 'target', got {unclaimed!r}")
        self.unclaimed = unclaimed

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

    # ------------------------------------------------------------------
    # Acceptable-regions block placement (matches original I-JEPA)
    # ------------------------------------------------------------------

    def _sample_block_mask(
        self,
        h: int,
        w: int,
        acceptable_regions: list[list[list[int]]],
    ) -> tuple[list[int], list[list[int]]]:
        """
        Place an h-by-w rectangle on the grid, masked by acceptable_regions.

        Each entry in *acceptable_regions* is a 2D grid (list-of-lists, g×g)
        with 1 = available, 0 = claimed. The rectangle is AND-ed with all of
        them so it only keeps cells that are free under every prior block.

        Returns
        -------
        indices : sorted flat patch indices for the block
        mask_complement : g×g grid that is 1 everywhere EXCEPT the placed
                          rectangle (to be appended to acceptable_regions)
        """
        g = self.grid
        tries = 0

        while True:
            # Sample a random placement.
            top = random.randint(0, max(0, g - h))
            left = random.randint(0, max(0, g - w))

            # Build the rectangle mask (g×g), then AND with acceptable_regions.
            mask = [[0] * g for _ in range(g)]
            for r in range(top, min(top + h, g)):
                for c in range(left, min(left + w, g)):
                    mask[r][c] = 1

            # Apply acceptable_regions (drop entries from the tail for relaxation).
            n_regions = max(0, len(acceptable_regions) - tries)
            for region in acceptable_regions[:n_regions]:
                for r in range(g):
                    for c in range(g):
                        mask[r][c] &= region[r][c]

            # Flatten → nonzero → sorted indices.
            indices = sorted(
                r * g + c for r in range(g) for c in range(g) if mask[r][c]
            )

            if len(indices) >= self.min_keep:
                break

            # Resample placement up to max_resample_tries, then relax.
            tries += 1
            if tries > self.max_resample_tries + len(acceptable_regions):
                # Fully relaxed and still failing — take whatever we got.
                if not indices:
                    # Extreme fallback: just use the raw rectangle.
                    indices = sorted(
                        (top + dr) * g + (left + dc)
                        for dr in range(h)
                        for dc in range(w)
                    )
                break

        # Build complement: 1 everywhere except the original rectangle.
        complement = [[1] * g for _ in range(g)]
        for r in range(top, min(top + h, g)):
            for c in range(left, min(left + w, g)):
                complement[r][c] = 0

        return indices, complement

    # ------------------------------------------------------------------
    # Per-image sampling (given pre-sampled block sizes)
    # ------------------------------------------------------------------

    def _sample_one(
        self,
        tgt_sizes: list[tuple[int, int]],
    ) -> tuple[list[list[int]], list[int]]:
        """
        Generate target blocks and context for one image.

        Uses acceptable_regions to ensure blocks land in open space.
        Each placed block adds its complement to the acceptable_regions
        list, so subsequent blocks avoid it.

        Parameters
        ----------
        tgt_sizes : list of (h, w) per target block — shared across batch.

        Returns
        -------
        tgt_blocks : list of M lists (variable length per block)
        ctx        : list of length ≤ Nctx
        """
        acceptable_regions: list[list[list[int]]] = []
        tgt_blocks: list[list[int]] = []

        if self.allow_overlap:
            # No acceptable_regions needed — just place freely.
            for h, w in tgt_sizes:
                g = self.grid
                top = random.randint(0, max(0, g - h))
                left = random.randint(0, max(0, g - w))
                indices = sorted(
                    (top + r) * g + (left + c)
                    for r in range(h)
                    for c in range(w)
                )
                tgt_blocks.append(indices)

            # Context.
            ctx_h, ctx_w = self._sample_rect_size(
                self.ctx_min_scale,
                self.ctx_max_scale,
                self.ctx_min_aspect,
                self.ctx_max_aspect,
            )
            top = random.randint(0, max(0, g - ctx_h))
            left = random.randint(0, max(0, g - ctx_w))
            ctx = sorted(
                (top + r) * g + (left + c)
                for r in range(ctx_h)
                for c in range(ctx_w)
            )
            nctx = min(self.nctx, len(ctx))
            if nctx < len(ctx):
                ctx = sorted(random.sample(ctx, nctx))
            return tgt_blocks, ctx

        # --- Non-overlapping path: use acceptable_regions ---

        # 1) Target blocks.
        for h, w in tgt_sizes:
            indices, complement = self._sample_block_mask(
                h, w, acceptable_regions,
            )
            tgt_blocks.append(indices)
            acceptable_regions.append(complement)

        # 2) Context block — same mechanism, naturally excludes targets.
        ctx_h, ctx_w = self._sample_rect_size(
            self.ctx_min_scale,
            self.ctx_max_scale,
            self.ctx_min_aspect,
            self.ctx_max_aspect,
        )
        ctx, _ = self._sample_block_mask(
            ctx_h, ctx_w, acceptable_regions,
        )
        nctx = min(self.nctx, len(ctx))
        if nctx < len(ctx):
            ctx = sorted(random.sample(ctx, nctx))

        # Redistribute unclaimed patches if requested.
        if self.unclaimed != "ignore":
            claimed: set[int] = set(ctx)
            for block in tgt_blocks:
                claimed.update(block)
            unclaimed = sorted(set(range(self.num_patches)) - claimed)

            if self.unclaimed == "context":
                ctx = sorted(ctx + unclaimed)
            elif self.unclaimed == "target":
                for i, idx in enumerate(unclaimed):
                    tgt_blocks[i % len(tgt_blocks)].append(idx)
                for block in tgt_blocks:
                    block.sort()

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
