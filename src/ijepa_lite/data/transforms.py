from __future__ import annotations

from torchvision import transforms


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
