# FILE: src/ijepa_lite/masking/compressor.py
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class TokenCompressor(nn.Module):
    """
    Compresses EMA encoder patch tokens before feeding to a LatentMasker.

    Sits between: target_encoder output (B, N, D)  →  compressed (B, M, D)

    Modes
    -----
    "full"
        Identity pass-through.  All N patch tokens are forwarded unchanged.
        M = N.  No learned parameters.

    "cls"
        Returns only the CLS token produced by the ViT encoder.
        Requires ViTTokens.forward(return_cls=True) — set
        TokenCompressor.needs_cls to True to trigger this in IJEPAModel.
        The CLS token is the ViT's own global summary representation, computed
        over the full unmasked image.
        M = 1.  No learned parameters.

    "mean_pool"
        Global average pooling over all N patch tokens.
        Differentiable, cheap, no learned parameters.
        M = 1.

    "spatial_stride"
        Subsample the patch grid: keep 1 of every (stride × stride) patches.
        Preserves spatial structure at lower resolution.
        Requires a square patch grid (N = grid²).
        M = ceil(grid / stride)².  No learned parameters.
        Configured via: stride (int, default 2).

    "attention_pool"
        M learned query vectors cross-attend over N patch tokens via MHA.
        Flexible M, learns to pool task-relevant information.
        M = num_queries (default 4).  Learned parameters: query embeddings + MHA.
        Configured via: num_queries (int), num_heads (int).

    Args
    ----
    mode        : compression mode (see above)
    dim         : token embedding dimension — must match encoder embed_dim
    stride      : spatial subsampling factor (spatial_stride only, default 2)
    num_queries : number of output tokens   (attention_pool only, default 4)
    num_heads   : attention heads           (attention_pool only, default 8)
    """

    VALID_MODES = frozenset({"full", "cls", "mean_pool", "spatial_stride", "attention_pool"})

    def __init__(
        self,
        mode: str,
        dim: int,
        stride: int = 2,
        num_queries: int = 4,
        num_heads: int = 8,
    ) -> None:
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"TokenCompressor mode={mode!r} is not valid. "
                f"Choose from: {sorted(self.VALID_MODES)}"
            )

        self.mode = mode
        self.dim = int(dim)
        self.stride = int(stride)
        self.num_queries = int(num_queries)

        # Learned parameters — only for attention_pool
        if mode == "attention_pool":
            self.queries = nn.Parameter(torch.zeros(1, num_queries, dim))
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                batch_first=True,
            )
            nn.init.trunc_normal_(self.queries, std=0.02)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def needs_cls(self) -> bool:
        """True iff this compressor requires the CLS token from the encoder.

        When True, IJEPAModel will call ViTTokens.forward(return_cls=True) and
        pass the resulting CLS tensor as cls_token to this compressor's forward.
        """
        return self.mode == "cls"

    def output_len(self, num_patches: int) -> int:
        """Return the number of output tokens M for a given input N = num_patches."""
        if self.mode == "full":
            return num_patches
        if self.mode in ("cls", "mean_pool"):
            return 1
        if self.mode == "spatial_stride":
            grid = int(math.isqrt(num_patches))
            return math.ceil(grid / self.stride) ** 2
        if self.mode == "attention_pool":
            return self.num_queries
        raise AssertionError(f"unreachable: mode={self.mode!r}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        patch_tokens: torch.Tensor,                   # (B, N, D)
        cls_token: Optional[torch.Tensor] = None,     # (B, D), required for mode="cls"
    ) -> torch.Tensor:
        """
        Args:
            patch_tokens : (B, N, D) patch token output from the EMA encoder.
            cls_token    : (B, D) CLS token from the EMA encoder.
                           Only required (and used) when mode="cls".

        Returns:
            (B, M, D) compressed token sequence.
        """
        if self.mode == "full":
            return patch_tokens

        if self.mode == "cls":
            if cls_token is None:
                raise ValueError(
                    "TokenCompressor(mode='cls') requires cls_token to be provided. "
                    "Ensure IJEPAModel calls ViTTokens.forward(return_cls=True)."
                )
            return cls_token.unsqueeze(1)  # (B, 1, D)

        if self.mode == "mean_pool":
            return patch_tokens.mean(dim=1, keepdim=True)  # (B, 1, D)

        if self.mode == "spatial_stride":
            return self._spatial_stride(patch_tokens)

        if self.mode == "attention_pool":
            return self._attention_pool(patch_tokens)

        raise AssertionError(f"unreachable: mode={self.mode!r}")

    # ------------------------------------------------------------------
    # Mode-specific helpers
    # ------------------------------------------------------------------

    def _spatial_stride(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Subsample a square patch grid by self.stride along both spatial axes."""
        B, N, D = patch_tokens.shape
        grid = int(math.isqrt(N))
        if grid * grid != N:
            raise ValueError(
                f"TokenCompressor(mode='spatial_stride') requires a square patch grid. "
                f"Got N={N} which is not a perfect square (isqrt={grid}, {grid}²={grid**2})."
            )
        # Reshape to (B, grid, grid, D), subsample, flatten back
        tokens_2d = patch_tokens.view(B, grid, grid, D)
        tokens_sub = tokens_2d[:, :: self.stride, :: self.stride, :]  # (B, G', G', D)
        G_prime = tokens_sub.shape[1]
        return tokens_sub.reshape(B, G_prime * G_prime, D)  # (B, M, D)

    def _attention_pool(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """M learned query vectors cross-attend over N patch tokens."""
        B = patch_tokens.shape[0]
        queries = self.queries.expand(B, -1, -1)  # (B, M, D)
        out, _ = self.cross_attn(
            query=queries,
            key=patch_tokens,
            value=patch_tokens,
        )
        return out  # (B, M, D)