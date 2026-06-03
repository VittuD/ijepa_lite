def infer_num_classes(cfg, ds_train=None) -> int:
    """
    Hardcoded mapping for known datasets, with optional override.

    Priority:
      1) cfg.task.num_classes (explicit override)
      2) dataset metadata (if available)
      3) hardcoded mapping by cfg.data.name
    """
    # 1) Explicit override
    if getattr(cfg.task, "num_classes", None) is not None:
        return int(cfg.task.num_classes)

    # 2) Use dataset metadata when it exists (nice for downstream custom datasets)
    if ds_train is not None:
        if hasattr(ds_train, "classes"):
            return len(ds_train.classes)
        if hasattr(ds_train, "class_to_idx"):
            return len(ds_train.class_to_idx)

    # 3) Hardcoded mapping
    name = str(cfg.data.name).lower()
    mapping = {
        "cifar10": 10,
        "cifar100": 100,
        "stl10": 10,
        "food101": 101,
        "dtd": 47,
        "minc2500": 23,
        "sun397": 397,
        "clevr_count": 8,
        "organmnist": 11,
        "tissuemnist": 8,
        "chestmnist": 14,
        "pneumoniamnist": 2,
        "vocseg": 21,
        "imagenet100": 100,
        "imagenet": 1000,  # imagenet1k on disk (ImageFolder)
        "imagenet_128": 1000,  # alias for the HF 128x128 version
    }
    if name in mapping:
        return mapping[name]

    if name == "fairface":
        target_attr = str(getattr(cfg.data, "target_attr", "race")).lower()
        fairface_mapping = {
            "race": 7,
            "gender": 2,
            "age": 9,
        }
        if target_attr in fairface_mapping:
            return fairface_mapping[target_attr]

    raise ValueError(
        f"Unknown dataset '{name}'. Add it to infer_num_classes() or set task.num_classes."
    )
