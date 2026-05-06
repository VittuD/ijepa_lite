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
    max_total_hard : Optional cap on nctx + ntgt predictor tokens.
    hard_assignment: "topk", "argmax", "gumbel", "random_region_growth",
                     or "random_region_growth_multiblock"
                     for hard reconstruction masks.
    gumbel_tau     : Temperature for Gumbel hard assignment.
    rrg_keep_percent: Percentage of argmax winners to keep per role when using
                      random_region_growth.
    rrg_num_target_blocks: Number of target blocks when using multiblock
                           random-region-growth hard assignment.
    warmup_use_vanilla_multiblock: When True and epoch < warmup_epochs, use
                      the vanilla random MultiBlockMaskGenerator for hard masks
                      and fall back to pure reconstruction loss, regardless of
                      the configured post-warmup hard_assignment mode.
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
        max_total_hard: int = 0,
        hard_assignment: str = "topk",
        gumbel_tau: float = 1.0,
        rrg_keep_percent: int = 50,
        rrg_num_target_blocks: int = 4,
        rrg_keep_k_per_block: int = 0,
        warmup_use_vanilla_multiblock: bool = False,
        warmup_epochs: int = 0,
        total_epochs: int = 1000,
        pos_embed_kind: str = "learned",
        image_size: int = 96,
        patch_size: int = 8,
        target_ratio: float = 0.25,
        context_ratio: float = 0.75,
        num_target_blocks: int = 4,
        allow_overlap: bool = False,
        min_keep: int = 10,
        ctx_scale: list[float] | tuple[float, float] = (0.85, 1.00),
        ctx_aspect: list[float] | tuple[float, float] = (0.75, 1.50),
        tgt_scale: list[float] | tuple[float, float] = (0.15, 0.20),
        tgt_aspect: list[float] | tuple[float, float] = (0.75, 1.50),
        # Unused — kept for build.py kwarg filtering
        base_kind: str = "smooth_l1",
        normalize: bool = False,
    ) -> None:
        super().__init__()

        self.num_patches = int(num_patches)
        self.ntgt_min = max(1, int(ntgt_min))
        self.nctx_min = max(1, int(nctx_min))
        self.max_total_hard = int(max_total_hard)
        if hard_assignment not in (
            "topk",
            "argmax",
            "gumbel",
            "random_region_growth",
            "random_region_growth_multiblock",
        ):
            raise ValueError(
                "hard_assignment must be 'topk', 'argmax', 'gumbel', "
                "'random_region_growth', or 'random_region_growth_multiblock', "
                f"got {hard_assignment!r}"
            )
        if not (1 <= int(rrg_keep_percent) <= 100):
            raise ValueError(
                "rrg_keep_percent must be in [1, 100], "
                f"got {rrg_keep_percent!r}"
            )
        if int(rrg_keep_k_per_block) < 0:
            raise ValueError(
                "rrg_keep_k_per_block must be >= 0, "
                f"got {rrg_keep_k_per_block!r}"
            )
        self.rrg_num_target_blocks = max(1, int(rrg_num_target_blocks))
        if self.rrg_num_target_blocks >= self.num_patches:
            raise ValueError(
                "rrg_num_target_blocks must be smaller than num_patches, "
                f"got {self.rrg_num_target_blocks} for num_patches={self.num_patches}."
            )
        if (
            hard_assignment in ("random_region_growth", "random_region_growth_multiblock")
            and self.max_total_hard > 0
        ):
            raise ValueError(
                f"hard_assignment={hard_assignment!r} does not support "
                "max_total_hard > 0."
            )
        self.hard_assignment = str(hard_assignment)
        self.gumbel_tau = float(gumbel_tau)
        self.rrg_keep_percent = int(rrg_keep_percent)
        self.rrg_keep_k_per_block = int(rrg_keep_k_per_block)
        self.warmup_use_vanilla_multiblock = bool(warmup_use_vanilla_multiblock)
        self.warmup_epochs = int(warmup_epochs)
        self.total_epochs = int(total_epochs)

        # _progress in [0, 1]; initialised to 1.0 so unit tests use full range.
        self.register_buffer("_progress", torch.tensor(1.0), persistent=False)

        d = predictor_dim

        from ijepa_lite.models.pos_embed import build_pos_embed_2d

        grid_size = int(math.isqrt(num_patches))
        if grid_size * grid_size != self.num_patches:
            raise ValueError(
                f"num_patches={self.num_patches} is not a perfect square; "
                "RandomRegionGrowth relies on 2D patch-grid coordinates."
            )

        rows = torch.arange(grid_size, dtype=torch.long)
        cols = torch.arange(grid_size, dtype=torch.long)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        coords = torch.stack([grid_r.reshape(-1), grid_c.reshape(-1)], dim=-1)
        self.register_buffer("_patch_coords", coords, persistent=False)

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

        self._warmup_mask_generator = None
        if self.warmup_use_vanilla_multiblock:
            from ijepa_lite.masking.multiblock_mask import MultiBlockMaskGenerator

            ctx_scale = list(ctx_scale)
            ctx_aspect = list(ctx_aspect)
            tgt_scale = list(tgt_scale)
            tgt_aspect = list(tgt_aspect)
            self._warmup_mask_generator = MultiBlockMaskGenerator(
                image_size=int(image_size),
                patch_size=int(patch_size),
                target_ratio=float(target_ratio),
                context_ratio=float(context_ratio),
                num_target_blocks=self.rrg_num_target_blocks,
                tgt_min_scale=float(tgt_scale[0]),
                tgt_max_scale=float(tgt_scale[1]),
                tgt_min_aspect=float(tgt_aspect[0]),
                tgt_max_aspect=float(tgt_aspect[1]),
                ctx_min_scale=float(ctx_scale[0]),
                ctx_max_scale=float(ctx_scale[1]),
                ctx_min_aspect=float(ctx_aspect[0]),
                ctx_max_aspect=float(ctx_aspect[1]),
                allow_overlap=bool(allow_overlap),
                min_keep=int(min_keep),
            )

        # ----------------------------------------------------------------
        # Initialisation
        # ----------------------------------------------------------------
        nn.init.trunc_normal_(self.selection_token, std=0.02)
        nn.init.trunc_normal_(self.proj_score.weight, std=0.02)
        nn.init.zeros_(self.proj_score.bias)

    def _sample_hard_winners(
        self,
        logits: torch.Tensor,
        soft: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        if mode == "gumbel":
            g = -torch.empty_like(logits).exponential_().log()
            return (logits / self.gumbel_tau + g).argmax(dim=-1)
        if mode == "argmax":
            return soft.argmax(dim=-1)
        raise ValueError(f"Unsupported winner sampling mode={mode!r}")

    def _warmup_random_multiblock_active(self, epoch: Optional[int]) -> bool:
        return (
            self.warmup_use_vanilla_multiblock
            and self._warmup_mask_generator is not None
            and epoch is not None
            and int(epoch) < self.warmup_epochs
        )

    def _build_warmup_multiblock_masks(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
        mask_output = self._warmup_mask_generator(batch_size=batch_size)
        ctx_idx = mask_output.context_idx.to(device)
        tgt_idx = mask_output.target_idx.to(device)
        b = int(batch_size)
        n = self.num_patches
        m = int(tgt_idx.shape[1])
        k = int(tgt_idx.shape[2])

        ctx_mask = torch.zeros((b, n), device=device, dtype=torch.float32)
        tgt_mask = torch.zeros((b, n), device=device, dtype=torch.float32)
        ctx_mask.scatter_(1, ctx_idx, 1.0)
        tgt_flat = tgt_idx.reshape(b, -1)
        tgt_mask.scatter_(1, tgt_flat, 1.0)
        p_ctx = ctx_mask
        p_tgt = tgt_mask.clamp(max=1.0)
        p_ign = (1.0 - p_ctx - p_tgt).clamp(min=0.0)
        target_block_counts = torch.full(
            (m,), k, device=device, dtype=torch.long
        )
        aux_counts = {
            "warmup_random_multiblock_active": 1.0,
            "target_block_counts": target_block_counts,
            "hard_sampled_nctx": float(ctx_idx.shape[1]),
            "hard_sampled_ntgt": float(m * k),
            "hard_sampled_nign": float(self.num_patches - ctx_idx.shape[1] - m * k),
        }
        return ctx_idx, tgt_idx, {
            "context_soft": p_ctx,
            "target_soft": p_tgt,
            "p_ign": p_ign,
            "aux_counts": aux_counts,
        }

    def _ensure_role_min_counts(
        self,
        winners: torch.Tensor,
        soft: torch.Tensor,
        min_counts: dict[int, int],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Ensure each sample has the requested minimum winner count per role."""
        winners = winners.clone()
        fallback_counts = {role: 0 for role in min_counts}

        for b in range(winners.shape[0]):
            reserved: set[int] = set()
            while True:
                role_counts = {
                    role: int((winners[b] == role).sum().item())
                    for role in min_counts
                }
                missing_roles = [
                    role for role, min_count in min_counts.items()
                    if role_counts[role] < min_count
                ]
                if not missing_roles:
                    break

                role = missing_roles[0]
                order = soft[b, :, role].argsort(descending=True)
                chosen_idx = None
                for idx in order.tolist():
                    idx = int(idx)
                    if idx in reserved:
                        continue
                    current_role = int(winners[b, idx].item())
                    if (
                        current_role in role_counts
                        and current_role != role
                        and role_counts[current_role] <= min_counts[current_role]
                    ):
                        continue
                    chosen_idx = idx
                    break
                if chosen_idx is None:
                    raise RuntimeError(
                        "Could not assign fallback winner without violating "
                        f"required role minima in sample {b}."
                    )
                winners[b, chosen_idx] = role
                reserved.add(chosen_idx)
                fallback_counts[role] += 1

        aux_counts = {}
        role_names = {0: "ctx", 1: "tgt"}
        for role, count in fallback_counts.items():
            aux_counts[f"rrg_fallback_{role_names.get(role, role)}"] = float(count)
        return winners, aux_counts

    def _ensure_required_winner_roles(
        self,
        winners: torch.Tensor,
        soft: torch.Tensor,
        required_roles: tuple[int, ...],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Ensure each sample has at least one winner for every required role.

        RandomRegionGrowth operates on an exclusive hard winner partition, but
        plain argmax can collapse an entire sample to a subset of roles early in
        training. For missing required roles, minimally repair the winner map by
        reassigning the highest-probability available patch to that role.
        """
        return self._ensure_role_min_counts(
            winners,
            soft,
            {role: 1 for role in required_roles},
        )

    def _allocate_hard_counts(self, raw_nctx: float, raw_ntgt: float) -> tuple[int, int]:
        """Allocate hard ctx/tgt widths under the optional predictor-token cap."""
        nctx = max(self.nctx_min, int(round(float(raw_nctx))))
        ntgt = max(self.ntgt_min, int(round(float(raw_ntgt))))
        nctx = min(nctx, self.num_patches - self.ntgt_min)
        ntgt = min(ntgt, self.num_patches - nctx)

        if self.max_total_hard <= 0:
            return nctx, ntgt

        min_total = self.nctx_min + self.ntgt_min
        budget = min(self.num_patches, max(min_total, self.max_total_hard))
        if nctx + ntgt <= budget:
            return nctx, ntgt

        extra_budget = budget - min_total
        if extra_budget <= 0:
            return self.nctx_min, self.ntgt_min

        ctx_extra_signal = max(0.0, float(raw_nctx) - float(self.nctx_min))
        tgt_extra_signal = max(0.0, float(raw_ntgt) - float(self.ntgt_min))
        signal_sum = ctx_extra_signal + tgt_extra_signal
        if signal_sum <= 0.0:
            ctx_extra = extra_budget // 2
        else:
            ctx_extra = int(round(extra_budget * ctx_extra_signal / signal_sum))
        ctx_extra = max(0, min(extra_budget, ctx_extra))
        tgt_extra = extra_budget - ctx_extra
        return self.nctx_min + ctx_extra, self.ntgt_min + tgt_extra

    def _rectangularize_winner_masks(
        self,
        winners: torch.Tensor,
        p_ctx: torch.Tensor,
        p_tgt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        tgt_winners = winners == 1
        # Pack a rectangular hard mask without letting the max-count sample
        # dictate the whole batch width. Winners are ranked before fillers;
        # fillers use soft probabilities to avoid arbitrary zero-score ties.
        tgt_counts = tgt_winners.float().sum(dim=-1)
        tgt_scores = tgt_winners.float() + p_tgt * (~tgt_winners).float()
        raw_ntgt = float(tgt_counts.float().mean().item())
        ctx_winners = winners == 0
        ctx_counts = ctx_winners.float().sum(dim=-1)
        ctx_scores = ctx_winners.float() + p_ctx * (~ctx_winners).float()
        raw_nctx = float(ctx_counts.float().mean().item())
        nctx, ntgt = self._allocate_hard_counts(raw_nctx, raw_ntgt)
        # topk avoids sorting all patches while still selecting winners
        # before non-winner fillers.
        _, tgt_idx = torch.topk(tgt_scores, ntgt, dim=-1, sorted=False)

        ctx_scores = ctx_scores.scatter(1, tgt_idx, -torch.inf)
        _, ctx_idx = torch.topk(ctx_scores, nctx, dim=-1, sorted=False)

        aux_counts = {
            "hard_sampled_nctx": float(ctx_counts.float().mean().detach().item()),
            "hard_sampled_ntgt": float(tgt_counts.float().mean().detach().item()),
            "hard_sampled_nign": float(
                (winners == 2).float().sum(dim=-1).mean().detach().item()
            ),
        }
        return ctx_idx, tgt_idx, aux_counts

    def _grow_random_region(
        self,
        winners: torch.Tensor,
        role_idx: int,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        keep_fraction = float(self.rrg_keep_percent) / 100.0
        exact_indices: list[torch.Tensor] = []
        exact_counts = torch.empty(
            winners.shape[0], device=winners.device, dtype=torch.long
        )
        seed_indices = torch.empty(
            winners.shape[0], device=winners.device, dtype=torch.long
        )
        patch_coords = self._patch_coords.to(device=winners.device)

        for b in range(winners.shape[0]):
            role_indices = (winners[b] == role_idx).nonzero(as_tuple=False).flatten()
            count = int(role_indices.numel())
            if count <= 0:
                role_name = "ctx" if role_idx == 0 else "tgt"
                raise RuntimeError(
                    "hard_assignment='random_region_growth' requires at least one "
                    f"{role_name} argmax winner per sample; sample {b} had none."
                )
            keep_count = max(1, int(math.ceil(count * keep_fraction)))
            seed_offset = int(
                torch.randint(count, (1,), device=winners.device).item()
            )
            seed_idx = role_indices[seed_offset]
            seed_indices[b] = seed_idx
            seed_coord = patch_coords[seed_idx]
            role_coords = patch_coords[role_indices]
            distances = (role_coords - seed_coord).abs().sum(dim=-1)
            tie_break = distances * self.num_patches + role_indices
            order = tie_break.argsort(dim=0)
            exact = role_indices[order[:keep_count]]
            exact_indices.append(exact)
            exact_counts[b] = keep_count

        return exact_indices, exact_counts, seed_indices

    def _sample_rrg_target_seeds(
        self,
        role_indices: torch.Tensor,
    ) -> torch.Tensor:
        count = int(role_indices.numel())
        if count >= self.rrg_num_target_blocks:
            order = torch.randperm(count, device=role_indices.device)
            return role_indices[order[:self.rrg_num_target_blocks]]
        seed_offsets = torch.randint(
            count,
            (self.rrg_num_target_blocks,),
            device=role_indices.device,
        )
        return role_indices[seed_offsets]

    def _partition_target_support(
        self,
        role_indices: torch.Tensor,
        seed_indices: torch.Tensor,
    ) -> list[torch.Tensor]:
        patch_coords = self._patch_coords.to(device=role_indices.device)
        role_coords = patch_coords[role_indices]
        seed_coords = patch_coords[seed_indices]
        distances = (role_coords[:, None, :] - seed_coords[None, :, :]).abs().sum(dim=-1)
        tie_break = distances * self.num_patches + seed_indices.unsqueeze(0)
        assign = tie_break.argmin(dim=1)

        blocks: list[torch.Tensor] = []
        for k in range(seed_indices.numel()):
            block_indices = role_indices[assign == k]
            block_distances = distances[assign == k, k]
            order = (block_distances * self.num_patches + block_indices).argsort(dim=0)
            blocks.append(block_indices[order])
        return blocks

    def _allocate_rrg_block_keep_counts(
        self,
        block_sizes: list[int],
        keep_total: int,
    ) -> list[int]:
        num_blocks = len(block_sizes)
        total_size = sum(block_sizes)
        if num_blocks <= 0 or total_size <= 0:
            raise RuntimeError("RRG multiblock allocation requires non-empty blocks.")

        keep_total = max(num_blocks, min(total_size, int(keep_total)))
        counts = [1 for _ in block_sizes]
        remaining = keep_total - num_blocks
        if remaining <= 0:
            return counts

        capacities = [max(0, size - 1) for size in block_sizes]
        capacity_sum = sum(capacities)
        if capacity_sum <= 0:
            return counts

        raw_quota = [remaining * cap / capacity_sum for cap in capacities]
        extra = [min(cap, int(math.floor(quota))) for cap, quota in zip(capacities, raw_quota)]
        counts = [base + add for base, add in zip(counts, extra)]
        remaining -= sum(extra)

        order = sorted(
            range(num_blocks),
            key=lambda i: (raw_quota[i] - extra[i], capacities[i], block_sizes[i]),
            reverse=True,
        )
        for i in order:
            if remaining <= 0:
                break
            if counts[i] < block_sizes[i]:
                counts[i] += 1
                remaining -= 1
        if remaining > 0:
            for i in order:
                while remaining > 0 and counts[i] < block_sizes[i]:
                    counts[i] += 1
                    remaining -= 1
                if remaining <= 0:
                    break
        return counts

    def _truncate_multiblock_target_sets(
        self,
        exact_blocks: list[list[torch.Tensor]],
    ) -> torch.Tensor:
        final_ntgt = min(
            int(block.numel()) for sample_blocks in exact_blocks for block in sample_blocks
        )
        if final_ntgt <= 0:
            raise RuntimeError(
                "RandomRegionGrowthMultiblock produced an empty executed target block "
                "after batch-min truncation."
            )
        return torch.stack(
            [
                torch.stack([block[:final_ntgt] for block in sample_blocks], dim=0)
                for sample_blocks in exact_blocks
            ],
            dim=0,
        )

    def _random_region_growth_multiblock_indices(
        self,
        logits: torch.Tensor,
        soft: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        winners = self._sample_hard_winners(logits, soft, mode="argmax")
        winners, fallback_counts = self._ensure_role_min_counts(
            winners,
            soft,
            {0: 1, 1: 1},
        )
        exact_ctx, exact_ctx_counts, ctx_seed_idx = self._grow_random_region(
            winners, role_idx=0
        )

        keep_fraction = float(self.rrg_keep_percent) / 100.0
        patch_coords_device = winners.device
        patch_coords = self._patch_coords.to(device=patch_coords_device)
        target_blocks: list[list[torch.Tensor]] = []
        target_seed_idx = torch.empty(
            winners.shape[0], self.rrg_num_target_blocks, device=patch_coords_device, dtype=torch.long
        )
        semantic_counts = torch.empty(
            winners.shape[0], self.rrg_num_target_blocks, device=patch_coords_device, dtype=torch.long
        )
        semantic_unique_counts = torch.empty(
            winners.shape[0], device=patch_coords_device, dtype=torch.long
        )
        exec_unique_counts = torch.empty(
            winners.shape[0], device=patch_coords_device, dtype=torch.long
        )
        semantic_keep_per_block = torch.empty(
            winners.shape[0], device=patch_coords_device, dtype=torch.long
        )

        for b in range(winners.shape[0]):
            total_count = int((winners[b] == 1).sum().item())
            if self.rrg_keep_k_per_block > 0:
                semantic_keep_per_block[b] = max(
                    1,
                    min(total_count, self.rrg_keep_k_per_block),
                )
            else:
                semantic_keep_per_block[b] = max(
                    1,
                    int(math.ceil(total_count * keep_fraction)),
                )

        final_ntgt_per_block = int(semantic_keep_per_block.min().item())
        if final_ntgt_per_block <= 0:
            raise RuntimeError(
                "RandomRegionGrowthMultiblock produced an empty executed target block "
                "after batch-min truncation."
            )

        for b in range(winners.shape[0]):
            role_indices = (winners[b] == 1).nonzero(as_tuple=False).flatten()
            total_count = int(role_indices.numel())
            if total_count <= 0:
                raise RuntimeError(
                    "hard_assignment='random_region_growth_multiblock' requires at "
                    f"least one tgt argmax winner per sample; sample {b} had none."
                )
            keep_count = int(semantic_keep_per_block[b].item())
            seeds = self._sample_rrg_target_seeds(role_indices)
            role_coords = patch_coords[role_indices]
            blocks_semantic: list[torch.Tensor] = []
            blocks_exec: list[torch.Tensor] = []
            for seed_idx in seeds:
                seed_coord = patch_coords[seed_idx]
                distances = (role_coords - seed_coord).abs().sum(dim=-1)
                tie_break = distances * self.num_patches + role_indices
                order = tie_break.argsort(dim=0)
                block_semantic = role_indices[order[:keep_count]]
                blocks_semantic.append(block_semantic)
                blocks_exec.append(block_semantic[:final_ntgt_per_block])

            target_blocks.append(blocks_exec)
            target_seed_idx[b] = seeds
            semantic_counts[b].fill_(keep_count)
            semantic_unique_counts[b] = torch.unique(
                torch.cat(blocks_semantic, dim=0)
            ).numel()
            exec_unique_counts[b] = torch.unique(
                torch.cat(blocks_exec, dim=0)
            ).numel()

        ctx_idx = torch.stack(
            [x[: min(int(y.numel()) for y in exact_ctx)] for x in exact_ctx], dim=0
        )
        tgt_idx = torch.stack(
            [torch.stack(sample_blocks, dim=0) for sample_blocks in target_blocks],
            dim=0,
        )
        final_nctx = ctx_idx.shape[1]
        target_block_counts = torch.full(
            (self.rrg_num_target_blocks,),
            final_ntgt_per_block,
            device=patch_coords_device,
            dtype=torch.long,
        )
        semantic_counts_mean = semantic_counts.float().mean(dim=0)
        semantic_total_mean = semantic_counts.float().sum(dim=1).mean().item()
        exec_total = self.rrg_num_target_blocks * final_ntgt_per_block

        aux_counts = {
            "rrg_keep_percent": float(self.rrg_keep_percent),
            "rrg_keep_k_per_block": float(self.rrg_keep_k_per_block),
            "rrg_num_target_blocks": float(self.rrg_num_target_blocks),
            "rrg_ctx_seed_idx": ctx_seed_idx,
            "rrg_tgt_seed_idx_blocks": target_seed_idx,
            "rrg_semantic_nctx": float(exact_ctx_counts.float().mean().item()),
            "rrg_semantic_ntgt": float(semantic_total_mean),
            "rrg_semantic_ntgt_total": float(semantic_total_mean),
            "rrg_semantic_ntgt_unique": float(semantic_unique_counts.float().mean().item()),
            "rrg_semantic_nign": float(
                self.num_patches
                - exact_ctx_counts.float().mean().item()
                - semantic_total_mean
            ),
            "rrg_exec_nctx": float(final_nctx),
            "rrg_exec_ntgt": float(exec_total),
            "rrg_exec_ntgt_total": float(exec_total),
            "rrg_exec_ntgt_unique": float(exec_unique_counts.float().mean().item()),
            "rrg_exec_nign": float(self.num_patches - final_nctx - exec_total),
            "hard_sampled_nctx": float(final_nctx),
            "hard_sampled_ntgt": float(exec_total),
            "hard_sampled_nign": float(self.num_patches - final_nctx - exec_total),
            "target_block_counts": target_block_counts,
            **fallback_counts,
        }
        for k in range(self.rrg_num_target_blocks):
            aux_counts[f"rrg_semantic_ntgt_block_{k}"] = float(semantic_counts_mean[k].item())
            aux_counts[f"rrg_exec_ntgt_block_{k}"] = float(final_ntgt_per_block)
        return ctx_idx, tgt_idx, aux_counts

    def _truncate_exact_sets(
        self,
        exact_ctx: list[torch.Tensor],
        exact_tgt: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        final_nctx = min(int(x.numel()) for x in exact_ctx)
        final_ntgt = min(int(x.numel()) for x in exact_tgt)
        if final_nctx <= 0 or final_ntgt <= 0:
            raise RuntimeError(
                "RandomRegionGrowth produced an empty executed context or target set "
                "after batch-min truncation."
            )
        ctx_idx = torch.stack([x[:final_nctx] for x in exact_ctx], dim=0)
        tgt_idx = torch.stack([x[:final_ntgt] for x in exact_tgt], dim=0)
        return ctx_idx, tgt_idx

    def _random_region_growth_indices(
        self,
        logits: torch.Tensor,
        soft: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        winners = self._sample_hard_winners(logits, soft, mode="argmax")
        winners, fallback_counts = self._ensure_required_winner_roles(
            winners, soft, required_roles=(0, 1)
        )
        exact_ctx, exact_ctx_counts, ctx_seed_idx = self._grow_random_region(
            winners, role_idx=0
        )
        exact_tgt, exact_tgt_counts, tgt_seed_idx = self._grow_random_region(
            winners, role_idx=1
        )
        ctx_idx, tgt_idx = self._truncate_exact_sets(exact_ctx, exact_tgt)

        aux_counts = {
            "rrg_keep_percent": float(self.rrg_keep_percent),
            "rrg_ctx_seed_idx": ctx_seed_idx,
            "rrg_tgt_seed_idx": tgt_seed_idx,
            "rrg_semantic_nctx": float(exact_ctx_counts.float().mean().item()),
            "rrg_semantic_ntgt": float(exact_tgt_counts.float().mean().item()),
            "rrg_semantic_nign": float(
                self.num_patches
                - exact_ctx_counts.float().mean().item()
                - exact_tgt_counts.float().mean().item()
            ),
            "rrg_exec_nctx": float(ctx_idx.shape[1]),
            "rrg_exec_ntgt": float(tgt_idx.shape[1]),
            "rrg_exec_nign": float(self.num_patches - ctx_idx.shape[1] - tgt_idx.shape[1]),
            "hard_sampled_nctx": float(ctx_idx.shape[1]),
            "hard_sampled_ntgt": float(tgt_idx.shape[1]),
            "hard_sampled_nign": float(self.num_patches - ctx_idx.shape[1] - tgt_idx.shape[1]),
            **fallback_counts,
        }
        return ctx_idx, tgt_idx, aux_counts

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
        epoch: Optional[int] = None,
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
        if self._warmup_random_multiblock_active(epoch):
            ctx_idx, tgt_idx, warmup_payload = self._build_warmup_multiblock_masks(
                batch_size=B,
                device=tokens.device,
            )
            p_ctx = warmup_payload["context_soft"]
            p_tgt = warmup_payload["target_soft"]
            p_ign = warmup_payload["p_ign"]
            aux_counts = warmup_payload["aux_counts"]
        elif self.hard_assignment == "random_region_growth":
            ctx_idx, tgt_idx, aux_counts = self._random_region_growth_indices(
                logits, soft
            )
        elif self.hard_assignment == "random_region_growth_multiblock":
            ctx_idx, tgt_idx, aux_counts = self._random_region_growth_multiblock_indices(
                logits, soft
            )
        elif self.hard_assignment in ("argmax", "gumbel"):
            winners = self._sample_hard_winners(logits, soft, mode=self.hard_assignment)
            ctx_idx, tgt_idx, aux_counts = self._rectangularize_winner_masks(
                winners, p_ctx, p_tgt
            )
        else:
            raw_ntgt = float(p_tgt.sum(dim=-1).mean().item())
            raw_nctx = float(p_ctx.sum(dim=-1).mean().item())
            nctx, ntgt = self._allocate_hard_counts(raw_nctx, raw_ntgt)

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
                "warmup_grad_anchor": logits,
                "ema_full": ema_full,
                "epoch": epoch,
                "total_epochs": self.total_epochs,
                "hard_assignment": self.hard_assignment,
                "max_total_hard": self.max_total_hard,
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

        if bool(mask_output.aux.get("warmup_random_multiblock_active", 0.0)):
            warmup_grad_anchor = mask_output.aux.get("warmup_grad_anchor")
            if warmup_grad_anchor is None:
                return reconstruction_loss
            # Keep the learned masker branch in the autograd graph during
            # vanilla-mask warmup so DDP does not flag its parameters as unused,
            # while still applying exactly zero update to that branch.
            return reconstruction_loss + 0.0 * warmup_grad_anchor.sum()

        weights  = mask_output.aux["weights"]
        ema_full = mask_output.aux.get("ema_full")
        epoch = mask_output.aux.get("epoch")
        total_epochs = mask_output.aux.get("total_epochs")

        if ema_full is None:
            # Unit test fallback — compute only entropy terms
            device = p_ctx.device
            ema_full = torch.zeros(
                p_ctx.shape[0], p_ctx.shape[1], 1, device=device,
            )

        extra_kw = {}
        if epoch is not None:
            extra_kw["epoch"] = int(epoch)
        if total_epochs is not None:
            extra_kw["total_epochs"] = int(total_epochs)
        total, logs = self.composite_loss(
            weights=weights,
            p_ctx=p_ctx,
            p_tgt=p_tgt,
            p_ign=p_ign,
            ema_full=ema_full,
            **extra_kw,
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
        max_total_hard: int = 0,
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
        self.max_total_hard = int(max_total_hard)
        self.nctx_min = max(1, int(nctx_min))
        if hard_assignment not in ("topk", "argmax", "gumbel"):
            raise ValueError(
                "hard_assignment must be 'topk', 'argmax', or 'gumbel' for "
                f"mi_nway, got {hard_assignment!r}"
            )
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
        counts : (M,) LongTensor when a target or total-hard cap is active,
                 otherwise None.
        """
        target_budget = self._target_budget_limit()
        if target_budget <= 0:
            return None

        device = signal.device
        counts = torch.full(
            (self.M,), self.ntgt_min_per_block, device=device, dtype=torch.long
        )
        min_total = int(counts.sum().item())
        total_budget = max(min_total, target_budget)
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

    def _target_budget_limit(self) -> int:
        """Target-token budget implied by max_total_tgt and max_total_hard."""
        budgets = []
        if self.max_total_tgt > 0:
            budgets.append(self.max_total_tgt)
        if self.max_total_hard > 0:
            min_target_total = self.M * self.ntgt_min_per_block
            budgets.append(max(min_target_total, self.max_total_hard - self.nctx_min))
        return min(budgets) if budgets else 0

    def _context_budget_limit(self, target_total: int) -> int:
        """Context-token cap implied by max_total_hard after target allocation."""
        limit = self.num_patches
        if self.max_total_hard > 0:
            limit = min(limit, max(self.nctx_min, self.max_total_hard - int(target_total)))
        return limit

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
            target_total = int(target_block_counts.sum().item())

            # --- Vectorized context indices ---
            ctx_scores = (winners == 0).float()  # (B, N)
            ctx_counts = ctx_scores.sum(-1)       # (B,)
            needs_ctx_fallback = ctx_counts < self.nctx_min
            if needs_ctx_fallback.any():
                # For fallback samples: use p_ctx with target positions zeroed out
                p_ctx_fb = p_ctx[needs_ctx_fallback].clone()              # (n_fb, N)
                p_ctx_fb.scatter_(1, tgt_flat[needs_ctx_fallback], 0.0)
                ctx_scores[needs_ctx_fallback] = p_ctx_fb

            ctx_cap = self._context_budget_limit(target_total)
            nctx = max(self.nctx_min, min(K_max_cap, ctx_cap, int(ctx_counts.max().item())))
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
            target_total = int(target_block_counts.sum().item())

            # Context: topk on p_ctx after zeroing all target positions
            ctx_cap = self._context_budget_limit(target_total)
            nctx = max(self.nctx_min, min(ctx_cap, int(round(p_ctx.sum(dim=-1).mean().item()))))
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
            "max_total_hard": self.max_total_hard,
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
