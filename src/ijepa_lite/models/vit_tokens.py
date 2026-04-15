# FILE: src/ijepa_lite/models/vit_tokens.py
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16
from torchvision.models.vision_transformer import VisionTransformer


def _build_vit_small_16(
    image_size: int, patch_size: int, embed_dim: int, depth: int, num_heads: int
) -> VisionTransformer:
    return VisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=depth,
        num_heads=num_heads,
        hidden_dim=embed_dim,
        mlp_dim=embed_dim * 4,
        num_classes=1000,  # irrelevant; head is removed below
    )


def _remove_classifier_head(vit: nn.Module) -> None:
    """
    We never use classification logits in this repo (token-level pretraining),
    so remove classifier params to avoid unused-parameter issues in DDP.
    """
    if hasattr(vit, "heads"):
        setattr(vit, "heads", nn.Identity())
    if hasattr(vit, "classifier"):
        setattr(vit, "classifier", nn.Identity())
    if hasattr(vit, "fc"):
        setattr(vit, "fc", nn.Identity())


def _build_src_key_padding_mask(
    valid: Optional[torch.Tensor],
    *,
    prepend_cls: bool = False,
) -> Optional[torch.Tensor]:
    """
    Convert a per-token validity mask into PyTorch's src_key_padding_mask
    convention, where True means "ignore this position".
    """
    if valid is None:
        return None

    valid = valid.to(dtype=torch.bool)
    if prepend_cls:
        cls_valid = torch.ones(
            valid.shape[0], 1, device=valid.device, dtype=torch.bool
        )
        valid = torch.cat([cls_valid, valid], dim=1)

    src_key_padding_mask = ~valid
    if not bool(src_key_padding_mask.any().item()):
        return None
    return src_key_padding_mask


def _run_vit_layers_with_padding_mask(
    vit: VisionTransformer,
    x: torch.Tensor,
    src_key_padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Torchvision's encoder stack is stored as an nn.Sequential, so the stock
    `vit.encoder.layers(x)` call cannot thread a padding mask. When no mask is
    needed we keep the exact legacy path; otherwise we replay the EncoderBlock
    forward manually with `key_padding_mask`.
    """
    if src_key_padding_mask is None:
        return vit.encoder.layers(x)

    for block in vit.encoder.layers:
        required = ("ln_1", "self_attention", "dropout", "ln_2", "mlp")
        if not all(hasattr(block, name) for name in required):
            raise RuntimeError(
                "Unsupported torchvision EncoderBlock layout for padding-mask forward."
            )

        residual = x
        y = block.ln_1(x)
        y, _ = block.self_attention(
            y,
            y,
            y,
            need_weights=False,
            key_padding_mask=src_key_padding_mask,
        )
        y = block.dropout(y)
        x = residual + y

        y = block.ln_2(x)
        y = block.mlp(y)
        x = x + y

    return x


class ViTTokens(nn.Module):
    """
    Wrap torchvision ViT to return patch tokens (B, K, D), with optional
    masking applied *before* the transformer blocks.

    The forward pass is split into three explicit stages matching i-JEPA:

      Stage 1: patchify + pos embed (all N patches, no masking yet):
        images → patch_embed → add pos_embed → (B, N, D)

      Stage 2: optional token selection (done HERE, before self-attention):
        if keep_idx is given:  (B, N, D) → gather → (B, K, D)
        This is the key i-JEPA trick: the transformer blocks never see the
        target patches, so the context encoder is genuinely blind to them.
        Positional information is already encoded before the subset, so no
        information is lost about *where* each kept token sits.

      Stage 3: transformer blocks + LN:
        (B, 1+K, D) → blocks → LN → strip CLS → (B, K, D)

    When keep_idx=None (default) the encoder sees all N patches, which is
    the correct behaviour for:
      - the target encoder (always unmasked, full image)
      - linear probe evaluation (want all patch tokens for mean-pooling)

    CLS token
    ---------
    By default the CLS token is stripped from the output (legacy behaviour).
    Pass return_cls=True to additionally receive the CLS token as a separate
    tensor.  This is used by TokenCompressor(mode='cls') to feed the ViT's
    global summary representation to a LatentMasker.

    Signature change is fully backward compatible: return_cls=False is the
    default and the return type is identical to before when False.
    """

    def __init__(self, vit: VisionTransformer) -> None:
        super().__init__()
        self.vit = vit

    def forward(
        self,
        x: torch.Tensor,
        keep_idx: Optional[torch.Tensor] = None,
        keep_valid: Optional[torch.Tensor] = None,
        return_cls: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x          : (B, C, H, W) input images.
            keep_idx   : (B, K) long tensor of patch indices to keep, or None.
                         When provided, only those K patches enter the transformer.
                         When None, all N patches are processed (target encoder /
                         linear probe path).
            keep_valid : optional boolean mask matching keep_idx (or the full patch
                         sequence when keep_idx=None). True marks real tokens;
                         False marks padded slots that should be ignored by
                         self-attention. When omitted, behaviour is unchanged.
            return_cls : when False (default) return patch tokens only — (B, K, D).
                         when True return (cls_token, patch_tokens):
                           cls_token    : (B, D)   — ViT global summary token
                           patch_tokens : (B, K, D) — same as the default return

        Returns:
            return_cls=False : (B, K, D) patch token embeddings
            return_cls=True  : Tuple[(B, D), (B, K, D)] — (cls_token, patch_tokens)
        """
        if not hasattr(self.vit, "_process_input"):
            raise RuntimeError("Unsupported torchvision VisionTransformer version.")

        b = x.shape[0]

        # ------------------------------------------------------------------
        # Stage 1: patchify + positional embedding (full image, all N tokens)
        # ------------------------------------------------------------------
        x = self.vit._process_input(x)  # (B, N, D)
        cls = self.vit.class_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, x], dim=1)  # (B, 1+N, D)
        x = x + self.vit.encoder.pos_embedding  # broadcast add
        x = self.vit.encoder.dropout(x)

        # ------------------------------------------------------------------
        # Stage 2: optional masking: select context patches BEFORE blocks
        #
        # We keep the CLS token (position 0) so the transformer still has a
        # global summary token; only the patch sequence is subsetted.
        # ------------------------------------------------------------------
        if keep_idx is not None:
            patch_tokens = x[:, 1:]  # (B, N, D)
            d = patch_tokens.shape[-1]
            # gather the K kept patches; keep_idx is (B, K)
            patch_tokens = patch_tokens.gather(
                1, keep_idx.unsqueeze(-1).expand(-1, -1, d)
            )  # (B, K, D)
            x = torch.cat([x[:, :1], patch_tokens], dim=1)  # (B, 1+K, D)
            if keep_valid is not None and keep_valid.shape != keep_idx.shape:
                raise ValueError(
                    f"keep_valid shape {tuple(keep_valid.shape)} does not match "
                    f"keep_idx shape {tuple(keep_idx.shape)}."
                )
        elif keep_valid is not None:
            expected_shape = (b, x.shape[1] - 1)
            if tuple(keep_valid.shape) != expected_shape:
                raise ValueError(
                    f"keep_valid shape {tuple(keep_valid.shape)} does not match "
                    f"the full patch-token shape {expected_shape}."
                )

        src_key_padding_mask = _build_src_key_padding_mask(
            keep_valid, prepend_cls=True
        )

        # ------------------------------------------------------------------
        # Stage 3: transformer blocks + layer norm
        # The sequence length seen here is:
        #   1+K  (masked context encoder path)
        #   1+N  (full target encoder / linear probe path)
        # ------------------------------------------------------------------
        x = _run_vit_layers_with_padding_mask(
            self.vit, x, src_key_padding_mask=src_key_padding_mask
        )
        x = self.vit.encoder.ln(x)

        patch_tokens = x[:, 1:]  # (B, K, D) or (B, N, D) — drop CLS from sequence

        if return_cls:
            cls_token = x[:, 0]  # (B, D)
            return cls_token, patch_tokens

        return patch_tokens


def build_torchvision_vit_tokens(cfg) -> ViTTokens:
    arch = str(cfg.arch)
    image_size = int(cfg.image_size)
    patch_size = int(cfg.patch_size)
    embed_dim = int(cfg.embed_dim)
    depth = int(cfg.depth)
    num_heads = int(cfg.num_heads)

    if arch == "vit_base_16":
        if image_size == 224 and patch_size == 16:
            vit = vit_b_16(weights=None)
        else:
            vit = VisionTransformer(
                image_size=image_size,
                patch_size=patch_size,
                num_layers=depth,
                num_heads=num_heads,
                hidden_dim=embed_dim,
                mlp_dim=embed_dim * 4,
                num_classes=1000,
            )
    elif arch == "vit_large_16":
        if image_size == 224 and patch_size == 16:
            vit = vit_l_16(weights=None)
        else:
            vit = VisionTransformer(
                image_size=image_size,
                patch_size=patch_size,
                num_layers=depth,
                num_heads=num_heads,
                hidden_dim=embed_dim,
                mlp_dim=embed_dim * 4,
                num_classes=1000,
            )
    elif arch == "vit_small_16":
        vit = _build_vit_small_16(image_size, patch_size, embed_dim, depth, num_heads)
    else:
        raise ValueError(f"Unknown arch={arch}")

    if bool(getattr(cfg, "remove_head", True)):
        _remove_classifier_head(vit)

    # Replace torchvision's learned pos_embedding if sincos is requested
    pos_kind = str(getattr(cfg, "pos_embed_kind", "learned"))
    if pos_kind != "learned":
        from ijepa_lite.models.pos_embed import build_pos_embed_2d_with_cls
        grid_size = image_size // patch_size
        new_pe = build_pos_embed_2d_with_cls(pos_kind, grid_size, embed_dim)
        vit.encoder.pos_embedding = new_pe

    return ViTTokens(vit)
