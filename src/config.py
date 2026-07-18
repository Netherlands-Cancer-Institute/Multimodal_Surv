from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import yaml


PROJECT_RELATIVE_PATHS = {
    "internal_metadata",
    "radiologic_dir",
    "checkpoint_dir",
    "output_dir",
}


def _resolve_path(value: str, project_root: Path) -> str:
    path = Path(value).expanduser()
    return str(path if path.is_absolute() else (project_root / path).resolve())


def _validate_config(config: Dict[str, Any]) -> None:
    required_sections = {"seed", "device", "paths", "data", "model", "training"}
    missing = required_sections.difference(config)
    if missing:
        raise KeyError(f"Missing configuration section(s): {sorted(missing)}")

    data = config["data"]
    fractions = [float(data[key]) for key in ("train_fraction", "val_fraction", "test_fraction")]
    if any(value <= 0 or value >= 1 for value in fractions):
        raise ValueError("Train, validation, and test fractions must each be between 0 and 1.")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError(f"Data split fractions must sum to 1.0, received {sum(fractions):.8f}.")
    if str(data["endpoint"]).lower() not in {"os", "dfs"}:
        raise ValueError("data.endpoint must be either 'os' or 'dfs'.")
    if int(data["batch_size"]) < 2:
        raise ValueError("Training batch size must be at least 2 for the Cox loss and BatchNorm layers.")
    if int(data["max_report_length"]) <= 0:
        raise ValueError("data.max_report_length must be positive.")

    model = config["model"]
    if str(model.get("image_backbone", "")).lower() != "resnet18":
        raise ValueError("Only model.image_backbone=resnet18 is implemented in this release.")
    if int(model["clinical_feature_dim"]) != 25 * 12:
        raise ValueError("model.clinical_feature_dim must be 300 (25 x 12).")
    if int(model["mutation_feature_dim"]) != 30:
        raise ValueError("model.mutation_feature_dim must be 30.")
    if int(model["therapy_feature_dim"]) != 8:
        raise ValueError("model.therapy_feature_dim must be 8.")


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load, validate, and resolve paths from a YAML configuration file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file does not contain a YAML mapping: {config_path}")

    _validate_config(config)
    project_root = config_path.parent.parent
    path_cfg = config["paths"]

    for key in PROJECT_RELATIVE_PATHS:
        path_cfg[key] = _resolve_path(str(path_cfg[key]), project_root)

    for group in ("external_metadata", "image_dirs"):
        for cohort, value in path_cfg[group].items():
            path_cfg[group][cohort] = _resolve_path(str(value), project_root)

    return config
