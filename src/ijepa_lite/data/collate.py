# FILE: src/ijepa_lite/data/collate.py
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from ijepa_lite.masking.base import CollateMasker


class IJEPACollate:
    """
    Collate function for i-JEPA pre-training.

    Mask generation is performed here for deterministic (CollateMasker) strategies.
    When a LatentMasker is used, no masker is passed and the collate stays dumb —
    it only stacks images.  Masking then happens inside IJEPAModel.forward on GPU,
    conditioned on EMA encoder outputs.

    The masker argument is typed as Optional[CollateMasker].  Passing a LatentMasker
    here is a type error by design: LatentMaskers are nn.Modules that need GPU access
    and cannot run in DataLoader worker processes.

    Args:
        masker : a CollateMasker instance (BlockMaskGenerator or MultiBlockMaskGenerator),
                 or None when using a LatentMasker (collate stays dumb).

    Batch dict keys produced:
        "images"       : FloatTensor (B, C, H, W)           always
        "context_idx"  : LongTensor  (B, Nctx)              when masker is not None
        "target_idx"   : LongTensor  (B, Ntgt) or (B, M, K) when masker is not None
    """

    def __init__(self, masker: Optional[CollateMasker] = None) -> None:
        self.masker = masker

    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        imgs = [item[0] if isinstance(item, (tuple, list)) else item for item in batch]
        images = torch.stack(imgs, dim=0)  # (B, C, H, W)

        out: Dict[str, torch.Tensor] = {"images": images}

        if self.masker is not None:
            mask_output = self.masker(batch_size=len(imgs))
            # Serialise MaskOutput to plain tensors for the DataLoader batch dict.
            # Soft scores are None for all CollateMaskers — not added to batch.
            out["context_idx"] = mask_output.context_idx  # (B, Nctx)
            out["target_idx"] = mask_output.target_idx    # (B, Ntgt) or (B, M, K)

        return out


class SupervisedCollate:
    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        imgs, labels = [], []
        for item in batch:
            imgs.append(item[0])
            labels.append(item[1])
        images = torch.stack(imgs, dim=0)
        if labels and isinstance(labels[0], torch.Tensor):
            y = torch.stack(labels, dim=0)
        else:
            y = torch.as_tensor(labels)
            if y.ndim <= 1:
                y = y.long()
        return {"images": images, "labels": y}


class SegmentationCollate:
    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        imgs, masks = [], []
        for item in batch:
            imgs.append(item[0])
            masks.append(item[1])
        images = torch.stack(imgs, dim=0)
        seg_masks = torch.stack(masks, dim=0).long()
        return {"images": images, "masks": seg_masks}


class DetectionCollate:
    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        imgs, targets = [], []
        for image, target in batch:
            imgs.append(image)
            targets.append(target)
        images = torch.stack(imgs, dim=0)
        return {"images": images, "targets": targets}
