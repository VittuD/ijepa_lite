from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk
from PIL import Image
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


class HFClassificationDataset(Dataset):
    """
    Thin wrapper for Hugging Face image classification datasets.

    Supports either:
    - loading directly from a Hub dataset id, or
    - loading from a local dataset-repo snapshot copied into the shared root.
    """

    def __init__(
        self,
        split: str,
        transform=None,
        repo_id: Optional[str] = None,
        local_dir: Optional[str] = None,
        saved_dir: Optional[str] = None,
        subset: Optional[str] = None,
        image_field: str = "image",
        label_field: str = "label",
        cache_dir: Optional[str] = None,
    ) -> None:
        saved_path = Path(saved_dir) if saved_dir is not None else None
        local_path = Path(local_dir) if local_dir is not None else None
        self.ds = None
        if saved_path is not None and saved_path.exists():
            try:
                ds_obj = load_from_disk(str(saved_path))
            except (FileNotFoundError, ValueError) as exc:
                print(
                    f"[HFClassificationDataset] ignoring invalid saved_dir='{saved_path}': {exc}"
                )
            else:
                if hasattr(ds_obj, "keys"):
                    if split not in ds_obj:
                        raise ValueError(
                            f"Saved HF dataset at '{saved_path}' has no split='{split}'."
                        )
                    self.ds = ds_obj[split]
                else:
                    self.ds = ds_obj

        if self.ds is None:
            source = (
                str(local_path)
                if local_path is not None and local_path.exists()
                else repo_id
            )
            if source is None:
                raise ValueError(
                    "HFClassificationDataset requires an existing saved_dir/local_dir or a repo_id."
                )

            kwargs = {}
            if subset is not None:
                kwargs["name"] = subset
            if cache_dir is not None:
                kwargs["cache_dir"] = cache_dir

            self.ds = load_dataset(source, split=split, **kwargs)
        self.transform = transform
        self.image_field = image_field
        self.label_field = label_field

        label_feature = self.ds.features[label_field]
        self.classes = list(getattr(label_feature, "names", []))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample[self.image_field]
        target = int(sample[self.label_field])
        if hasattr(img, "convert"):
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class SUN397Split(Dataset):
    """
    Deterministic train/val/test split wrapper around torchvision SUN397.

    torchvision exposes the full SUN397 dataset but not a built-in probe split,
    so we keep loading/parsing in torchvision and only add a reproducible split
    partition here.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        split_seed: int = 0,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split='{split}' for sun397. Expected: train|val|test."
            )
        if train_ratio <= 0.0 or val_ratio < 0.0 or (train_ratio + val_ratio) >= 1.0:
            raise ValueError(
                "sun397 requires 0 < train_ratio, 0 <= val_ratio, "
                "and train_ratio + val_ratio < 1."
            )

        base = tv_datasets.SUN397(root=root, download=False)
        pairs = sorted(
            zip(base._image_files, base._labels),
            key=lambda pair: str(pair[0]),
        )

        indices = list(range(len(pairs)))
        random.Random(split_seed).shuffle(indices)

        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train : n_train + n_val]
        else:
            selected = indices[n_train + n_val :]

        self.samples = [pairs[i] for i in selected]
        self.transform = transform
        self.loader = base.loader
        self.classes = list(base.classes)
        self.class_to_idx = dict(base.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(target)


class FairFaceDataset(Dataset):
    """
    CSV-backed FairFace loader for downstream eval.

    FairFace is distributed as image folders plus label CSVs rather than a
    torchvision dataset class, so we implement the thin wrapper here.
    """

    _CANONICAL_CLASSES = {
        "gender": ["Male", "Female"],
        "race": [
            "White",
            "Black",
            "Latino_Hispanic",
            "East Asian",
            "Southeast Asian",
            "Indian",
            "Middle Eastern",
        ],
        "age": [
            "0-2",
            "3-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70+",
        ],
    }

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        target_attr: str = "race",
        image_dirname: str = "fairface-img-margin025-trainval",
        train_csv: str = "fairface_label_train.csv",
        val_csv: str = "fairface_label_val.csv",
        csv_dir: Optional[str] = None,
        image_root: Optional[str] = None,
    ) -> None:
        target_attr = str(target_attr)
        if split not in ("train", "val"):
            raise ValueError(
                f"Unknown split='{split}' for fairface. Expected: train|val."
            )
        if target_attr not in self._CANONICAL_CLASSES:
            raise ValueError(
                "fairface target_attr must be one of: "
                + ", ".join(sorted(self._CANONICAL_CLASSES))
            )

        root_path = Path(root)
        csv_root = Path(csv_dir) if csv_dir is not None else root_path
        images_root = (
            Path(image_root) if image_root is not None else root_path / image_dirname
        )
        csv_name = train_csv if split == "train" else val_csv
        csv_path = csv_root / csv_name

        self.transform = transform
        self.target_attr = target_attr
        self.classes = list(self._CANONICAL_CLASSES[target_attr])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.samples = []

        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rel_path = row.get("file")
                if not rel_path:
                    raise ValueError(
                        f"fairface csv '{csv_path}' is missing the required 'file' column."
                    )
                raw_label = row.get(target_attr)
                if raw_label is None:
                    raise ValueError(
                        f"fairface csv '{csv_path}' is missing the required "
                        f"'{target_attr}' column."
                    )
                label = self._canonicalize_label(target_attr, raw_label)
                self.samples.append(
                    (images_root / rel_path, self.class_to_idx[label])
                )

    @staticmethod
    def _canonicalize_label(target_attr: str, raw_label: str) -> str:
        canonical = {
            label.lower().replace("_", " ").strip(): label
            for label in FairFaceDataset._CANONICAL_CLASSES[target_attr]
        }
        key = raw_label.lower().replace("_", " ").strip()
        if key not in canonical:
            raise ValueError(
                f"Unknown fairface {target_attr} label '{raw_label}'. "
                f"Expected one of: {FairFaceDataset._CANONICAL_CLASSES[target_attr]}"
            )
        return canonical[key]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class TissueMNISTDataset(Dataset):
    """
    Thin wrapper around MedMNIST TissueMNIST using the MedMNIST+ 128px source.

    The downstream probe stack expects ImageNet-style RGB-normalized inputs, so
    we request RGB conversion from MedMNIST even though the source data are
    grayscale.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        size: int = 128,
        download: bool = False,
        as_rgb: bool = True,
        mmap_mode: Optional[str] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split='{split}' for tissuemnist. Expected: train|val|test."
            )

        from medmnist import TissueMNIST

        self.ds = TissueMNIST(
            split=split,
            transform=transform,
            download=download,
            as_rgb=as_rgb,
            root=root,
            size=size,
            mmap_mode=mmap_mode,
        )
        self.transform = transform
        self.classes = list(class_names or [])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        return img, int(target.reshape(-1)[0])


class OrganMNISTDataset(Dataset):
    """
    Thin wrapper around MedMNIST OrganAMNIST using the MedMNIST+ 128px source.

    We interpret the repo-level `organmnist` downstream task as the axial-view
    OrganAMNIST variant and request RGB conversion to stay compatible with the
    shared ImageNet-style probe transforms.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        size: int = 128,
        download: bool = False,
        as_rgb: bool = True,
        mmap_mode: Optional[str] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split='{split}' for organmnist. Expected: train|val|test."
            )

        from medmnist import OrganAMNIST

        self.ds = OrganAMNIST(
            split=split,
            transform=transform,
            download=download,
            as_rgb=as_rgb,
            root=root,
            size=size,
            mmap_mode=mmap_mode,
        )
        self.transform = transform
        self.classes = list(class_names or [])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        return img, int(target.reshape(-1)[0])


class ChestMNISTDataset(Dataset):
    """
    Thin wrapper around MedMNIST ChestMNIST using the MedMNIST+ 128px source.

    ChestMNIST is a 14-label binary multilabel task. We return the full target
    vector as float32 so downstream probes can train with BCE-with-logits.
    For self-supervised pretraining, split="all" concatenates train/val/test;
    the pretrain collate keeps only the image tensor and ignores the target.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        size: int = 128,
        download: bool = False,
        as_rgb: bool = True,
        mmap_mode: Optional[str] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        if split not in ("train", "val", "test", "all"):
            raise ValueError(
                f"Unknown split='{split}' for chestmnist. Expected: train|val|test|all."
            )

        from medmnist import ChestMNIST

        kwargs = dict(
            transform=transform,
            download=download,
            as_rgb=as_rgb,
            root=root,
            size=size,
            mmap_mode=mmap_mode,
        )
        if split == "all":
            self.ds = ConcatDataset(
                ChestMNIST(split=part, **kwargs) for part in ("train", "val", "test")
            )
        else:
            self.ds = ChestMNIST(split=split, **kwargs)
        self.transform = transform
        self.classes = list(class_names or [])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.target_type = "multilabel"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        return img, torch.as_tensor(target.reshape(-1), dtype=torch.float32)


class PneumoniaMNISTDataset(Dataset):
    """
    Thin wrapper around MedMNIST PneumoniaMNIST using the MedMNIST+ 224px source.

    PneumoniaMNIST is a binary-class pediatric chest X-ray task. We request RGB
    conversion to stay compatible with the shared ImageNet-style probe transforms.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        size: int = 224,
        download: bool = False,
        as_rgb: bool = True,
        mmap_mode: Optional[str] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split='{split}' for pneumoniamnist. Expected: train|val|test."
            )

        from medmnist import PneumoniaMNIST

        self.ds = PneumoniaMNIST(
            split=split,
            transform=transform,
            download=download,
            as_rgb=as_rgb,
            root=root,
            size=size,
            mmap_mode=mmap_mode,
        )
        self.transform = transform
        self.classes = list(class_names or [])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        return img, int(target.reshape(-1)[0])


class SIIMACRPneumothoraxSegmentation(Dataset):
    """
    SIIM-ACR pneumothorax segmentation using the Kaggle PNG image/mask layout.

    Expected default layout under ``root``:
      siim-acr-pneumothorax/
        png_images/*.png
        png_masks/*.png

    When present, ``stage_1_train_images.csv`` and ``stage_1_test_images.csv``
    define the split via their ``new_filename`` column. If CSVs are absent, we
    fall back to filename markers and then to a deterministic train/val split.
    """

    _EXTS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root: str,
        split: str,
        transforms=None,
        data_dir: str = "siim-acr-pneumothorax",
        image_dir: str = "png_images",
        mask_dir: str = "png_masks",
        val_ratio: float = 0.2,
        split_seed: int = 0,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split='{split}' for siimacr_pneumothorax. "
                "Expected: train|val|test."
            )
        self.transforms = transforms

        base = Path(root) / data_dir
        if not base.exists():
            base = Path(root)
        self.image_dir = base / image_dir
        self.mask_dir = base / mask_dir
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"SIIM image_dir not found: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"SIIM mask_dir not found: {self.mask_dir}")

        mask_by_name = {
            path.name: path
            for path in self.mask_dir.rglob("*")
            if path.suffix.lower() in self._EXTS
        }
        pairs = []
        for image_path in sorted(self.image_dir.rglob("*")):
            if image_path.suffix.lower() not in self._EXTS:
                continue
            mask_path = mask_by_name.get(image_path.name)
            if mask_path is not None:
                pairs.append((image_path, mask_path))

        if not pairs:
            raise RuntimeError(
                f"No SIIM image/mask pairs found in {self.image_dir} and {self.mask_dir}."
            )

        pair_by_name = {image_path.name: (image_path, mask_path) for image_path, mask_path in pairs}
        train_pairs = self._pairs_from_csv(base, "stage_1_train_images.csv", pair_by_name)
        test_pairs = self._pairs_from_csv(base, "stage_1_test_images.csv", pair_by_name)

        if train_pairs and test_pairs:
            self.pairs = train_pairs if split == "train" else test_pairs
        else:
            train_pairs = [
                pair for pair in pairs if "_train_" in pair[0].name.lower()
            ]
            test_pairs = [
                pair for pair in pairs if "_test_" in pair[0].name.lower()
            ]
            if train_pairs and test_pairs:
                self.pairs = train_pairs if split == "train" else test_pairs
            else:
                rng = random.Random(int(split_seed))
                shuffled = list(pairs)
                rng.shuffle(shuffled)
                n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
                val_pairs = sorted(shuffled[:n_val])
                train_pairs = sorted(shuffled[n_val:])
                self.pairs = train_pairs if split == "train" else val_pairs

        if not self.pairs:
            raise RuntimeError(f"No SIIM pairs available for split='{split}'.")

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _pairs_from_csv(
        base: Path,
        filename: str,
        pair_by_name: dict[str, tuple[Path, Path]],
    ) -> list[tuple[Path, Path]]:
        csv_path = base / filename
        if not csv_path.is_file():
            matches = sorted(base.rglob(filename))
            if not matches:
                return []
            csv_path = matches[0]
        out = []
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("new_filename")
                if name is None:
                    continue
                pair = pair_by_name.get(name)
                if pair is not None:
                    out.append(pair)
        return out

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        mask = mask.point(lambda px: 1 if px > 0 else 0, mode="L")
        if self.transforms is not None:
            return self.transforms(image, mask)
        return image, mask


class RSNAPneumoniaDetectionDataset(Dataset):
    """
    RSNA Pneumonia Detection Challenge wrapper using pre-converted images.

    Expected default layout under ``root``:
      rsna-pneumonia-detection/
        stage_2_train_labels.csv
        png_images/<patientId>.png

    The CSV can contain multiple positive rows per patient. Negative rows have
    ``Target=0`` and no boxes. Boxes are returned as absolute xyxy pixel
    coordinates and should be normalized by the detection transform.
    """

    _EXTS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        root: str,
        split: str,
        transforms=None,
        data_dir: str = "rsna-pneumonia-detection",
        image_dir: str = "png_images",
        annotations_csv: str = "stage_2_train_labels.csv",
        val_ratio: float = 0.2,
        split_seed: int = 0,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split='{split}' for rsna_pneumonia_detection. "
                "Expected: train|val|test."
            )
        self.transforms = transforms

        base = Path(root) / data_dir
        if not base.exists():
            base = Path(root)
        self.image_dir = base / image_dir
        self.annotations_path = base / annotations_csv
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"RSNA image_dir not found: {self.image_dir}")
        if not self.annotations_path.is_file():
            matches = sorted(base.rglob(annotations_csv))
            if not matches:
                raise FileNotFoundError(
                    f"RSNA annotations_csv not found: {self.annotations_path}"
                )
            self.annotations_path = matches[0]

        image_by_id = {
            path.stem: path
            for path in self.image_dir.rglob("*")
            if path.suffix.lower() in self._EXTS
        }
        if not image_by_id:
            raise RuntimeError(f"No RSNA images found under {self.image_dir}.")

        boxes_by_id: dict[str, list[list[float]]] = {}
        seen_ids: set[str] = set()
        with self.annotations_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                patient_id = str(row.get("patientId", "")).strip()
                if not patient_id:
                    continue
                seen_ids.add(patient_id)
                boxes_by_id.setdefault(patient_id, [])
                target = int(float(row.get("Target") or 0))
                if target <= 0:
                    continue
                x = float(row.get("x") or 0.0)
                y = float(row.get("y") or 0.0)
                w = float(row.get("width") or 0.0)
                h = float(row.get("height") or 0.0)
                if w <= 0.0 or h <= 0.0:
                    continue
                boxes_by_id[patient_id].append([x, y, x + w, y + h])

        patient_ids = sorted(pid for pid in seen_ids if pid in image_by_id)
        missing = sorted(pid for pid in seen_ids if pid not in image_by_id)
        if missing:
            print(
                "[RSNAPneumoniaDetectionDataset] ignoring "
                f"{len(missing)} annotated patients without converted images."
            )
        if not patient_ids:
            raise RuntimeError(
                "No RSNA annotated patients with matching converted images found."
            )

        rng = random.Random(int(split_seed))
        shuffled = list(patient_ids)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
        val_ids = set(shuffled[:n_val])
        if split == "train":
            selected = [pid for pid in patient_ids if pid not in val_ids]
        else:
            selected = [pid for pid in patient_ids if pid in val_ids]

        if not selected:
            raise RuntimeError(f"No RSNA samples available for split='{split}'.")

        self.samples = [
            (pid, image_by_id[pid], boxes_by_id.get(pid, [])) for pid in selected
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        patient_id, image_path, boxes = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.ones((len(boxes),), dtype=torch.long),
            "image_id": patient_id,
            "orig_size": torch.as_tensor([image.height, image.width], dtype=torch.long),
        }
        if self.transforms is not None:
            return self.transforms(image, target)
        return image, target


class CLEVRCountDataset(Dataset):
    """
    Downstream classification wrapper for torchvision CLEVRClassification.

    torchvision returns the raw object count per scene. For downstream
    cross-entropy probes we remap the configured count values to contiguous
    zero-based class ids.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        download: bool = False,
        class_values: Optional[list[int]] = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split='{split}' for clevr_count. Expected: train|val|test."
            )

        base = tv_datasets.CLEVRClassification(
            root=root,
            split=split,
            transform=transform,
            download=download,
        )

        inferred_values = sorted({int(x) for x in base._labels if x is not None})
        values = list(class_values) if class_values is not None else inferred_values
        if not values:
            raise ValueError("clevr_count needs non-empty class_values or non-empty labels.")

        self.ds = base
        self.transform = transform
        self.class_values = [int(v) for v in values]
        self.classes = [str(v) for v in self.class_values]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self._value_to_idx = {value: idx for idx, value in enumerate(self.class_values)}

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img, target = self.ds[idx]
        if target is None:
            raise ValueError("clevr_count test split does not provide labels for downstream probing.")
        target = int(target)
        if target not in self._value_to_idx:
            raise ValueError(
                f"CLEVR count target {target} not present in configured class_values={self.class_values}."
            )
        return img, self._value_to_idx[target]


def _build_dataset_local(cfg, split: str, transform):
    name = str(cfg.name).lower()
    root = str(cfg.root)

    if name == "cifar10":
        if split not in ("train", "test"):
            raise ValueError(
                f"Unknown split='{split}' for cifar10. Expected: train|test."
            )
        return tv_datasets.CIFAR10(
            root=root,
            train=(split == "train"),
            download=False,
            transform=transform,
        )

    if name == "cifar100":
        if split not in ("train", "test"):
            raise ValueError(
                f"Unknown split='{split}' for cifar100. Expected: train|test."
            )
        return tv_datasets.CIFAR100(
            root=root,
            train=(split == "train"),
            download=False,
            transform=transform,
        )

    if name == "imagenet":
        if split not in ("train", "val"):
            raise ValueError(
                f"Unknown split='{split}' for imagenet. Expected: train|val."
            )
        leaf = "train" if split == "train" else "val"
        return tv_datasets.ImageFolder(root=f"{root}/{leaf}", transform=transform)

    if name == "stl10":
        if split not in ("train", "test", "unlabeled", "train+unlabeled"):
            raise ValueError(
                "Unknown split='{split}' for stl10. Expected: "
                "train|test|unlabeled|train+unlabeled.".format(split=split)
            )
        if split == "train+unlabeled":
            return ConcatDataset(
                [
                    tv_datasets.STL10(
                        root=root,
                        split="train",
                        download=False,
                        transform=transform,
                    ),
                    tv_datasets.STL10(
                        root=root,
                        split="unlabeled",
                        download=False,
                        transform=transform,
                    ),
                ]
            )
        return tv_datasets.STL10(
            root=root,
            split=split,
            download=False,
            transform=transform,
        )

    if name == "imagenet_128":
        if split not in ("train", "validation", "test"):
            raise ValueError(
                "Unknown split='{split}' for imagenet_128. Expected: "
                "train|validation|test.".format(split=split)
            )
        return HFImageNet128(split=split, transform=transform, cache_dir=root)

    if name == "food101":
        if split not in ("train", "test"):
            raise ValueError(
                f"Unknown split='{split}' for food101. Expected: train|test."
            )
        return tv_datasets.Food101(
            root=root,
            split=split,
            download=False,
            transform=transform,
        )

    if name == "dtd":
        if split not in ("train", "val", "test"):
            raise ValueError(
                f"Unknown split='{split}' for dtd. Expected: train|val|test."
            )
        return tv_datasets.DTD(
            root=root,
            split=split,
            partition=int(getattr(cfg, "partition", 1)),
            download=False,
            transform=transform,
        )

    if name == "sun397":
        backend = str(getattr(cfg, "backend", "hf")).lower()
        if backend == "hf":
            return HFClassificationDataset(
                split=split,
                transform=transform,
                repo_id=str(getattr(cfg, "hf_repo_id", "tanganke/sun397")),
                saved_dir=str(getattr(cfg, "hf_saved_dir", f"{root}/sun397_hf_saved")),
                local_dir=str(getattr(cfg, "hf_local_dir", f"{root}/sun397_hf")),
                subset=getattr(cfg, "hf_subset", None),
                image_field=str(getattr(cfg, "hf_image_field", "image")),
                label_field=str(getattr(cfg, "hf_label_field", "label")),
                cache_dir=root,
            )
        if backend != "torchvision":
            raise ValueError(
                f"Unknown sun397 backend='{backend}'. Expected: hf|torchvision."
            )
        return SUN397Split(
            root=root,
            split=split,
            transform=transform,
            train_ratio=float(getattr(cfg, "train_ratio", 0.8)),
            val_ratio=float(getattr(cfg, "val_ratio", 0.1)),
            split_seed=int(getattr(cfg, "split_seed", 0)),
        )

    if name == "minc2500":
        backend = str(getattr(cfg, "backend", "hf")).lower()
        if backend != "hf":
            raise ValueError(
                f"Unknown minc2500 backend='{backend}'. Expected: hf."
            )
        return HFClassificationDataset(
            split=split,
            transform=transform,
            repo_id=str(getattr(cfg, "hf_repo_id", "mcimpoi/minc-2500_split_1")),
            saved_dir=str(getattr(cfg, "hf_saved_dir", f"{root}/minc-2500-hf")),
            local_dir=getattr(cfg, "hf_local_dir", None),
            subset=getattr(cfg, "hf_subset", None),
            image_field=str(getattr(cfg, "hf_image_field", "image")),
            label_field=str(getattr(cfg, "hf_label_field", "label")),
            cache_dir=root,
        )

    if name == "fairface":
        backend = str(getattr(cfg, "backend", "hf")).lower()
        if backend == "hf":
            return HFClassificationDataset(
                split=split,
                transform=transform,
                repo_id=str(getattr(cfg, "hf_repo_id", "HuggingFaceM4/FairFace")),
                saved_dir=str(getattr(cfg, "hf_saved_dir", f"{root}/fairface_hf_saved")),
                local_dir=str(getattr(cfg, "hf_local_dir", f"{root}/fairface_hf")),
                subset=str(getattr(cfg, "hf_subset", "0.25")),
                image_field=str(getattr(cfg, "hf_image_field", "image")),
                label_field=str(getattr(cfg, "target_attr", "race")),
                cache_dir=root,
            )
        if backend != "local_csv":
            raise ValueError(
                f"Unknown fairface backend='{backend}'. Expected: hf|local_csv."
            )
        return FairFaceDataset(
            root=root,
            split=split,
            transform=transform,
            target_attr=str(getattr(cfg, "target_attr", "race")),
            image_dirname=str(
                getattr(cfg, "image_dirname", "fairface-img-margin025-trainval")
            ),
            train_csv=str(getattr(cfg, "train_csv", "fairface_label_train.csv")),
            val_csv=str(getattr(cfg, "val_csv", "fairface_label_val.csv")),
            csv_dir=getattr(cfg, "csv_dir", None),
            image_root=getattr(cfg, "image_root", None),
        )

    if name == "tissuemnist":
        return TissueMNISTDataset(
            root=root,
            split=split,
            transform=transform,
            size=int(getattr(cfg, "size", 128)),
            download=bool(getattr(cfg, "download", False)),
            as_rgb=bool(getattr(cfg, "as_rgb", True)),
            mmap_mode=getattr(cfg, "mmap_mode", None),
            class_names=list(getattr(cfg, "class_names", [])),
        )

    if name == "organmnist":
        return OrganMNISTDataset(
            root=root,
            split=split,
            transform=transform,
            size=int(getattr(cfg, "size", 128)),
            download=bool(getattr(cfg, "download", False)),
            as_rgb=bool(getattr(cfg, "as_rgb", True)),
            mmap_mode=getattr(cfg, "mmap_mode", None),
            class_names=list(getattr(cfg, "class_names", [])),
        )

    if name == "chestmnist":
        return ChestMNISTDataset(
            root=root,
            split=split,
            transform=transform,
            size=int(getattr(cfg, "size", 128)),
            download=bool(getattr(cfg, "download", False)),
            as_rgb=bool(getattr(cfg, "as_rgb", True)),
            mmap_mode=getattr(cfg, "mmap_mode", None),
            class_names=list(getattr(cfg, "class_names", [])),
        )

    if name == "pneumoniamnist":
        return PneumoniaMNISTDataset(
            root=root,
            split=split,
            transform=transform,
            size=int(getattr(cfg, "size", 224)),
            download=bool(getattr(cfg, "download", False)),
            as_rgb=bool(getattr(cfg, "as_rgb", True)),
            mmap_mode=getattr(cfg, "mmap_mode", None),
            class_names=list(getattr(cfg, "class_names", [])),
        )

    if name == "clevr_count":
        class_values = getattr(cfg, "class_values", None)
        return CLEVRCountDataset(
            root=root,
            split=split,
            transform=transform,
            download=bool(getattr(cfg, "download", False)),
            class_values=list(class_values) if class_values is not None else None,
        )

    raise ValueError(f"Unknown dataset name={name}")


def _maybe_download_dataset(cfg, split: str, transform) -> None:
    """
    Ensure dataset files exist on disk (rank0 only).
    """
    name = str(cfg.name).lower()
    root = str(cfg.root)

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

    if name == "food101":
        tv_datasets.Food101(
            root=root,
            split=split,
            download=True,
            transform=transform,
        )
        return

    if name == "vocseg":
        tv_datasets.VOCSegmentation(
            root=root,
            year=str(getattr(cfg, "year", "2012")),
            image_set=str(split),
            download=True,
        )
        return

    if name == "dtd":
        tv_datasets.DTD(
            root=root,
            split=split,
            partition=int(getattr(cfg, "partition", 1)),
            download=True,
            transform=transform,
        )
        return

    if name == "sun397":
        if str(getattr(cfg, "backend", "hf")).lower() == "hf":
            return
        tv_datasets.SUN397(root=root, download=True)
        return

    if name == "fairface":
        # No built-in download path; user should provide the extracted files.
        return

    if name == "tissuemnist":
        # MedMNIST file should already be present under the shared dataset root.
        return

    if name == "organmnist":
        # MedMNIST file should already be present under the shared dataset root.
        return

    if name == "chestmnist":
        # MedMNIST file should already be present under the shared dataset root.
        return

    if name == "pneumoniamnist":
        # MedMNIST file should already be present under the shared dataset root.
        return

    if name == "clevr_count":
        tv_datasets.CLEVRClassification(
            root=root,
            split=split,
            transform=transform,
            download=True,
        )
        return

    raise ValueError(f"Unknown dataset name={name}")


def build_dataset(cfg, split: str, transform):
    name = str(cfg.name).lower()
    download = bool(getattr(cfg, "download", True))

    if download and is_rank0():
        _maybe_download_dataset(cfg=cfg, split=split, transform=transform)

    barrier()
    return _build_dataset_local(cfg=cfg, split=split, transform=transform)


def build_segmentation_dataset(cfg, split: str, transforms):
    name = str(cfg.name).lower()
    root = str(cfg.root)
    download = bool(getattr(cfg, "download", True))

    if name == "siimacr_pneumothorax":
        if download:
            raise ValueError(
                "siimacr_pneumothorax cannot be downloaded automatically. "
                "Download/extract the Kaggle PNG dataset under data.root first."
            )
        barrier()
        return SIIMACRPneumothoraxSegmentation(
            root=root,
            split=split,
            transforms=transforms,
            data_dir=str(getattr(cfg, "data_dir", "siim-acr-pneumothorax")),
            image_dir=str(getattr(cfg, "image_dir", "png_images")),
            mask_dir=str(getattr(cfg, "mask_dir", "png_masks")),
            val_ratio=float(getattr(cfg, "val_ratio", 0.2)),
            split_seed=int(getattr(cfg, "split_seed", 0)),
        )

    if name != "vocseg":
        raise ValueError(
            f"Unknown segmentation dataset name={name}. Add it to build_segmentation_dataset()."
        )

    if split not in ("train", "trainval", "val"):
        raise ValueError(
            f"Unknown split='{split}' for vocseg. Expected: train|trainval|val."
        )

    if download and is_rank0():
        _maybe_download_dataset(cfg=cfg, split=split, transform=None)

    barrier()
    return tv_datasets.VOCSegmentation(
        root=root,
        year=str(getattr(cfg, "year", "2012")),
        image_set=str(split),
        download=False,
        transforms=transforms,
    )


def build_detection_dataset(cfg, split: str, transforms):
    name = str(cfg.name).lower()
    root = str(cfg.root)
    download = bool(getattr(cfg, "download", True))

    if name == "rsna_pneumonia_detection":
        if download:
            raise ValueError(
                "rsna_pneumonia_detection cannot be downloaded automatically. "
                "Pre-convert RSNA DICOMs to PNG/JPG under data.root first."
            )
        barrier()
        return RSNAPneumoniaDetectionDataset(
            root=root,
            split=split,
            transforms=transforms,
            data_dir=str(getattr(cfg, "data_dir", "rsna-pneumonia-detection")),
            image_dir=str(getattr(cfg, "image_dir", "png_images")),
            annotations_csv=str(
                getattr(cfg, "annotations_csv", "stage_2_train_labels.csv")
            ),
            val_ratio=float(getattr(cfg, "val_ratio", 0.2)),
            split_seed=int(getattr(cfg, "split_seed", 0)),
        )

    raise ValueError(
        f"Unknown detection dataset name={name}. Add it to build_detection_dataset()."
    )
