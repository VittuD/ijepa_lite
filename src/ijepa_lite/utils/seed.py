from __future__ import annotations

import os
import random

import numpy as np
import torch

from ijepa_lite.utils.dist import get_rank


def set_seed(seed: int) -> None:
    # rank offset so each process has different RNG streams
    seed = int(seed) + 1000 * int(get_rank())

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    # deterministic per-worker seeding derived from torch initial seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
