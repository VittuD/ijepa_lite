from __future__ import annotations

import random

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def build_pretrain_transform(cfg):
    image_size = int(cfg.model.image_size)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
            # transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def build_linear_probe_transforms(cfg):
    image_size = int(cfg.model.image_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_tfm = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    val_tfm = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tfm, val_tfm


class SegmentationPairTransform:
    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = int(image_size)
        self.train = bool(train)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, image, mask):
        if self.train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image, scale=(0.3, 1.0), ratio=(3 / 4, 4 / 3)
            )
            image = TF.resized_crop(
                image,
                i,
                j,
                h,
                w,
                size=[self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resized_crop(
                mask,
                i,
                j,
                h,
                w,
                size=[self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            image = TF.resize(
                image,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            )
            mask = TF.resize(
                mask,
                [self.image_size, self.image_size],
                interpolation=InterpolationMode.NEAREST,
            )

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask


def build_segmentation_transforms(cfg):
    image_size = int(cfg.model.image_size)
    return (
        SegmentationPairTransform(image_size=image_size, train=True),
        SegmentationPairTransform(image_size=image_size, train=False),
    )


class DetectionPairTransform:
    def __init__(self, image_size: int, train: bool) -> None:
        self.image_size = int(image_size)
        self.train = bool(train)
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def __call__(self, image, target):
        width, height = image.size
        boxes = target["boxes"].clone().float()

        image = TF.resize(
            image,
            [self.image_size, self.image_size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )

        if boxes.numel() > 0:
            scale_x = float(self.image_size) / float(width)
            scale_y = float(self.image_size) / float(height)
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        if self.train and random.random() < 0.5:
            image = TF.hflip(image)
            if boxes.numel() > 0:
                x0 = boxes[:, 0].clone()
                x1 = boxes[:, 2].clone()
                boxes[:, 0] = float(self.image_size) - x1
                boxes[:, 2] = float(self.image_size) - x0

        if boxes.numel() > 0:
            boxes[:, 0::2].clamp_(0.0, float(self.image_size))
            boxes[:, 1::2].clamp_(0.0, float(self.image_size))
            boxes = boxes / float(self.image_size)

        out = dict(target)
        out["boxes"] = boxes
        out["labels"] = target["labels"].long()
        out["orig_size"] = torch.as_tensor([height, width], dtype=torch.long)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, out


def build_detection_transforms(cfg):
    image_size = int(cfg.model.image_size)
    return (
        DetectionPairTransform(image_size=image_size, train=True),
        DetectionPairTransform(image_size=image_size, train=False),
    )
