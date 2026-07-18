from __future__ import annotations

from typing import Dict

import numpy as np
from lifelines.utils import concordance_index


def c_index(survival: np.ndarray, risk: np.ndarray) -> float:
    """Compute Harrell's C-index; higher model output denotes higher risk."""
    survival = np.asarray(survival)
    risk = np.asarray(risk).reshape(-1)
    if survival.ndim != 2 or survival.shape[1] != 2:
        raise ValueError("survival must have shape [N, 2].")
    if len(survival) != len(risk):
        raise ValueError("survival and risk must contain the same number of samples.")
    return float(concordance_index(survival[:, 0], -risk, survival[:, 1]))


def bootstrap_c_index(
    survival: np.ndarray,
    risk: np.ndarray,
    seed: int = 42,
    iterations: int = 1000,
) -> Dict[str, float]:
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")
    point_estimate = c_index(survival, risk)
    generator = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        indices = generator.integers(0, len(survival), len(survival))
        sampled_survival = survival[indices]
        if sampled_survival[:, 1].sum() == 0:
            continue
        try:
            estimates.append(c_index(sampled_survival, risk[indices]))
        except ZeroDivisionError:
            continue
    if not estimates:
        raise RuntimeError("No valid bootstrap replicate could be computed.")
    return {
        "c_index": point_estimate,
        "ci_lower": float(np.percentile(estimates, 2.5)),
        "ci_upper": float(np.percentile(estimates, 97.5)),
        "valid_bootstrap_iterations": int(len(estimates)),
    }
