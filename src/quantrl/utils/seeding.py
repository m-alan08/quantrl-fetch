from __future__ import annotations

import os
import random

import numpy as np
import torch

CUBLAS_WORKSPACE_SETTING = ":4096:8"


def configure_determinism(deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", CUBLAS_WORKSPACE_SETTING)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_everything(seed: int, deterministic: bool = True) -> np.random.Generator:
    configure_determinism(deterministic)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)
