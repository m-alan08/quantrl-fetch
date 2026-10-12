from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_torch: bool = True) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic_torch:
        torch.use_deterministic_algorithms(True)
    return np.random.default_rng(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
