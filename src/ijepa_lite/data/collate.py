from __future__ import annotations

from typing import Any, Dict, List

import torch


class IJEPACollate:
    """
    Collate function for i-JEPA pre-training.

    Mask generation is performed here — in the DataLoader worker processes on
    CPU so it is fully overlapped with GPU computation and never causes a
    CPU<->GPU synchronisation stall.

    If no masker is provided the batch dict will contain only "images";
    IJEPAModel.forward will then call the mask generator on the device
    (legacy / debug path, not recommended for training).

    Args:
        masker: a BlockMaskGenerator or MultiBlockMaskGenerator instance
                whose __call__(batch_size: int) -> dict method returns
                {"context_idx": LongTensor, "target_idx": LongTensor}
                on CPU.
    """

    def __init__(self, masker=None):
        self.masker = masker

    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        imgs = []
        for item in batch:
            imgs.append(item[0] if isinstance(item, (tuple, list)) else item)

        images = torch.stack(imgs, dim=0)  # (B, C, H, W)
        out: Dict[str, torch.Tensor] = {"images": images}

        if self.masker is not None:
            masks = self.masker(batch_size=len(imgs))
            # masks keys: "context_idx" (B, Nctx), "target_idx" (B, Ntgt) or (B, M, K)
            out.update(masks)

        return out


class SupervisedCollate:
    def __call__(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        imgs, labels = [], []
        for item in batch:
            imgs.append(item[0])
            labels.append(item[1])
        images = torch.stack(imgs, dim=0)
        y = torch.tensor(labels, dtype=torch.long)
        return {"images": images, "labels": y}
