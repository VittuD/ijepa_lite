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
        "imagenet100": 100,
        "imagenet": 1000,  # imagenet1k on disk (ImageFolder)
        "imagenet_128": 1000,  # alias for the HF 128x128 version
    }
    if name in mapping:
        return mapping[name]

    raise ValueError(
        f"Unknown dataset '{name}'. Add it to infer_num_classes() or set task.num_classes."
    )
