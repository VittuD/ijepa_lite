from __future__ import annotations

from typing import Optional

from datasets import load_dataset
from torch.utils.data import ConcatDataset, Dataset
from torchvision import datasets as tv_datasets

from ijepa_lite.utils.dist import barrier, is_rank0


class HFImageNet128(Dataset):
    """
    Thin wrapper around the HF ImageNet-1k-128x128 dataset.
    """

    def __init__(self, split: str, transform=None, cache_dir: Optional[str] = None):

        self.ds = load_dataset(
            "benjamin-paine/imagenet-1k-128x128", split=split, cache_dir=cache_dir
        )
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds[idx]
        img = x["image"]
        y = int(x["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def _maybe_download_dataset(name: str, root: str, split: str, transform) -> None:
    """
    Ensure dataset files exist on disk (rank0 only).
    """
    if name == "cifar10":
        tv_datasets.CIFAR10(
            root=root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )
        return

    if name == "cifar100":
        tv_datasets.CIFAR100(
            root=root,
            train=(split == "train"),
            download=True,
            transform=transform,
        )
        return

    if name == "stl10":
        if split == "train+unlabeled":
            # STL10 ships as a single archive; calling download on either split is enough,
            # but we touch both to keep intent explicit.
            tv_datasets.STL10(
                root=root, split="train", download=True, transform=transform
            )
            tv_datasets.STL10(
                root=root, split="unlabeled", download=True, transform=transform
            )
            return
        tv_datasets.STL10(root=root, split=split, download=True, transform=transform)
        return

    # imagenet: no download path (user should provide directory)
    if name == "imagenet":
        return

    if name == "imagenet_128":
        # HF dataset downloads on demand in __init__
        return

    raise ValueError(f"Unknown dataset name={name}")


def build_dataset(cfg, split: str, transform):
    name = str(cfg.name)
    root = str(cfg.root)
    download = bool(getattr(cfg, "download", True))

    specs = {
        "cifar10": (
            ("train", "test"),
            lambda r, s, t: tv_datasets.CIFAR10(
                root=r, train=(s == "train"), download=False, transform=t
            ),
        ),
        "cifar100": (
            ("train", "test"),
            lambda r, s, t: tv_datasets.CIFAR100(
                root=r, train=(s == "train"), download=False, transform=t
            ),
        ),
        "imagenet": (
            ("train", "val"),
            lambda r, s, t: tv_datasets.ImageFolder(
                root=f"{r}/{'train' if s == 'train' else 'val'}", transform=t
            ),
        ),
        "stl10": (
            ("train", "test", "unlabeled", "train+unlabeled"),
            lambda r, s, t: (
                ConcatDataset(
                    [
                        tv_datasets.STL10(
                            root=r, split="train", download=False, transform=t
                        ),
                        tv_datasets.STL10(
                            root=r, split="unlabeled", download=False, transform=t
                        ),
                    ]
                )
                if s == "train+unlabeled"
                else tv_datasets.STL10(root=r, split=s, download=False, transform=t)
            ),
        ),
        "imagenet_128": (
            ("train", "validation", "test"),
            lambda r, s, t: HFImageNet128(split=s, transform=t, cache_dir=r),
        ),
    }

    if name not in specs:
        raise ValueError(f"Unknown dataset name={name}")

    valid_splits, builder = specs[name]
    if split not in valid_splits:
        raise ValueError(
            f"Unknown split='{split}' for {name}. Expected: {'|'.join(valid_splits)}."
        )

    if download and is_rank0():
        _maybe_download_dataset(name=name, root=root, split=split, transform=transform)

    barrier()
    return builder(root, split, transform)
