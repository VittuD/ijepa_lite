from __future__ import annotations

import random

import torch


class BlockMaskGenerator:
    """
    CPU-side single-block mask generator, called from the DataLoader collate function.

    Produces fixed-count context_idx and target_idx per sample using only
    Python/CPU logic — no GPU tensor ops, no sync stalls.

    Returns:
        context_idx: (B, Nctx)  long
        target_idx:  (B, Ntgt)  long
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        target_ratio: float,
        context_ratio: float,
        min_block_area_ratio: float = 0.10,
        max_block_area_ratio: float = 0.35,
    ):
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid = self.image_size // self.patch_size
        self.num_patches = self.grid * self.grid

        self.target_ratio = float(target_ratio)
        self.context_ratio = float(context_ratio)

        self.ntgt = max(1, int(round(self.num_patches * self.target_ratio)))
        self.nctx = max(1, int(round(self.num_patches * self.context_ratio)))

        self.min_block_area_ratio = float(min_block_area_ratio)
        self.max_block_area_ratio = float(max_block_area_ratio)

    # ------------------------------------------------------------------
    # Per-sample helpers — pure Python / CPU
    # ------------------------------------------------------------------

    def _sample_block_indices(self) -> list[int]:
        """Sample a block of approximately self.ntgt patch indices."""
        g = self.grid
        area_ratio = random.uniform(
            self.min_block_area_ratio, self.max_block_area_ratio
        )
        block_area = max(self.ntgt, int(round(area_ratio * self.num_patches)))

        h = random.randint(1, g)
        w = max(1, min(g, int(round(block_area / h))))
        h = max(1, min(g, int(round(block_area / w))))

        top = random.randint(0, max(0, g - h))
        left = random.randint(0, max(0, g - w))

        idx = [(top + r) * g + (left + c) for r in range(h) for c in range(w)]

        if len(idx) >= self.ntgt:
            return random.sample(idx, self.ntgt)
        else:
            # block too small — fill from outside
            block_set = set(idx)
            pool = [i for i in range(self.num_patches) if i not in block_set]
            missing = self.ntgt - len(idx)
            idx += random.sample(pool, min(missing, len(pool)))
            # if pool exhausted (very small grid), sample with replacement
            while len(idx) < self.ntgt:
                idx.append(random.randint(0, self.num_patches - 1))
            return idx

    def _sample_one(self) -> tuple[list[int], list[int]]:
        """Return (ctx_indices, tgt_indices) for one image."""
        tgt = self._sample_block_indices()
        tgt_set = set(tgt)

        pool = [i for i in range(self.num_patches) if i not in tgt_set]
        if len(pool) >= self.nctx:
            ctx = random.sample(pool, self.nctx)
        else:
            # edge case: context quota exceeds available non-target patches
            ctx = pool[:]
            while len(ctx) < self.nctx:
                ctx.append(random.randint(0, self.num_patches - 1))

        return ctx, tgt

    # ------------------------------------------------------------------
    # Batch entry point — called by IJEPACollate
    # ------------------------------------------------------------------

    def __call__(self, batch_size: int) -> dict:
        """
        Args:
            batch_size: number of samples in the batch.

        Returns dict with:
            "context_idx": LongTensor (B, Nctx)
            "target_idx":  LongTensor (B, Ntgt)
        """
        ctx_list, tgt_list = [], []
        for _ in range(batch_size):
            ctx, tgt = self._sample_one()
            ctx_list.append(ctx)
            tgt_list.append(tgt)

        return {
            "context_idx": torch.tensor(ctx_list, dtype=torch.long),  # (B, Nctx)
            "target_idx": torch.tensor(tgt_list, dtype=torch.long),  # (B, Ntgt)
        }
