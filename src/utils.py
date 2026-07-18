from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Union

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds used by Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            pass


def ensure_directories(*paths: Union[str, Path]) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
