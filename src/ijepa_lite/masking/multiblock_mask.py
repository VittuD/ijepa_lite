from __future__ import annotations

import math
import random

import torch


class MultiBlockMaskGenerator:
    """
    CPU-side multi-block mask generator intended to run in DataLoader workers.

    Behavior (per sample):
      - Sample M target rectangles (optionally non-overlapping).
      - Sample one context rectangle, then remove target patches when overlap is disallowed.
      - Return fixed-shape tensors for batching:
          context_idx: (B, Nctx)
          target_idx:  (B, M, K)

    Notes on shapes:
      - Nctx is derived from context_ratio, but may be clipped when many patches are
        occupied by targets (when allow_overlap=False).
      - K is derived from target_ratio / M and is fixed per instance.
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
        max_resample_tries: int = 20,
    ):
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.grid = self.image_size // self.patch_size
        self.num_patches = self.grid * self.grid

        self.num_target_blocks = int(num_target_blocks)
        m = self.num_target_blocks

        self.tgt_min_scale = float(tgt_min_scale)
        self.tgt_max_scale = float(tgt_max_scale)
        self.tgt_min_aspect = float(tgt_min_aspect)
        self.tgt_max_aspect = float(tgt_max_aspect)

        self.ctx_min_scale = float(ctx_min_scale)
        self.ctx_max_scale = float(ctx_max_scale)
        self.ctx_min_aspect = float(ctx_min_aspect)
        self.ctx_max_aspect = float(ctx_max_aspect)

        self.allow_overlap = bool(allow_overlap)
        self.max_resample_tries = int(max_resample_tries)

        self.k_per_block = max(
            1, int(round(self.num_patches * float(target_ratio) / max(1, m)))
        )
        self.nctx = max(1, int(round(self.num_patches * float(context_ratio))))

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

    def _sample_rect_indices(self, h: int, w: int) -> set[int]:
        """Return the set of flat patch indices for a randomly placed h-by-w rectangle."""
        g = self.grid
        top = random.randint(0, max(0, g - h))
        left = random.randint(0, max(0, g - w))
        return {(top + r) * g + (left + c) for r in range(h) for c in range(w)}

    def _sample_one(self) -> tuple[list[list[int]], list[int]]:
        """
        Generate target blocks and context indices for one image.

        Returns:
            tgt_blocks: list of M lists, each length K
            ctx:        list of length Nctx (possibly clipped)
        """
        n = self.num_patches
        m = self.num_target_blocks
        k = self.k_per_block

        occupied: set[int] = set()
        tgt_blocks: list[list[int]] = []

        # 1) Target blocks.
        for _ in range(m):
            candidates: set[int] = set()

            for attempt in range(self.max_resample_tries + 1):
                h, w = self._sample_rect_size(
                    self.tgt_min_scale,
                    self.tgt_max_scale,
                    self.tgt_min_aspect,
                    self.tgt_max_aspect,
                )
                rect = self._sample_rect_indices(h, w)

                if self.allow_overlap:
                    candidates = rect
                    break

                candidates = rect - occupied
                if candidates:
                    break

                # Last attempt: accept an empty candidate set and rely on fallback sampling.
                if attempt == self.max_resample_tries:
                    candidates = set()

            picked = _sample_k(candidates, k, fallback_pool=_complement(occupied, n))
            tgt_blocks.append(picked)

            if not self.allow_overlap:
                occupied.update(picked)

        # 2) Context rectangle minus targets (when overlap is disallowed).
        ctx_candidates: set[int] = set()

        for attempt in range(self.max_resample_tries + 1):
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
                ctx_candidates = rect - occupied

            if ctx_candidates:
                break

            if attempt == self.max_resample_tries:
                ctx_candidates = set()

        # If the sampled rectangle produced no usable context indices, fall back to any non-occupied patch.
        if not ctx_candidates:
            ctx_candidates = _complement(occupied, n)

        max_ctx = n - len(occupied)
        nctx = min(self.nctx, max_ctx)
        ctx = _sample_k(ctx_candidates, nctx, fallback_pool=_complement(occupied, n))

        return tgt_blocks, ctx

    def __call__(self, batch_size: int) -> dict:
        """
        Generate masks for a batch.

        Returns:
            context_idx: LongTensor (B, Nctx)
            target_idx:  LongTensor (B, M, K)
        """
        ctx_list: list[list[int]] = []
        tgt_list: list[list[list[int]]] = []

        for _ in range(int(batch_size)):
            tgt_blocks, ctx = self._sample_one()
            ctx_list.append(ctx)
            tgt_list.append(tgt_blocks)

        return {
            "context_idx": torch.tensor(ctx_list, dtype=torch.long),
            "target_idx": torch.tensor(tgt_list, dtype=torch.long),
        }


def _complement(occupied: set[int], n: int) -> set[int]:
    """Return the set {0, ..., n-1} excluding occupied."""
    return set(range(n)) - occupied


def _sample_k(pool: set[int], k: int, fallback_pool: set[int]) -> list[int]:
    """
    Sample exactly k unique indices.

    Strategy:
      - If pool has >= k elements, sample from pool.
      - Otherwise, take all of pool and fill the remainder from fallback_pool.
      - If still short, sample with replacement from the union to reach length k.
    """
    pool_list = list(pool)
    if len(pool_list) >= k:
        return random.sample(pool_list, k)

    chosen = pool_list[:]
    needed = k - len(chosen)

    remaining = list(fallback_pool - pool)
    if len(remaining) >= needed:
        chosen += random.sample(remaining, needed)
        return chosen

    chosen += remaining
    universe = list(pool | fallback_pool)
    while len(chosen) < k:
        chosen.append(random.choice(universe))
    return chosen[:k]
