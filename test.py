from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from transformers import RobertaTokenizerFast

from src.config import load_config
from src.dataset import SurvivalDataset, load_records, make_dataloader, split_records
from src.metrics import bootstrap_c_index
from src.models import VLINC
from src.utils import ensure_directories, set_global_seed


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def main(config_path: str, cohort: str) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_global_seed(seed)
    requested_device = str(cfg["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    if cohort not in cfg["paths"]["image_dirs"]:
        raise KeyError(f"No image directory is configured for cohort '{cohort}'.")
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["paths"]["radiologic_dir"])

    if cohort == "internal":
        all_records = load_records(cfg["paths"]["internal_metadata"])
        records = split_records(all_records, cfg["data"], seed)[2]
    else:
        metadata_map = cfg["paths"]["external_metadata"]
        if cohort not in metadata_map:
            raise KeyError(f"No external metadata file is configured for cohort '{cohort}'.")
        records = load_records(metadata_map[cohort])

    dataset = SurvivalDataset(
        records,
        cfg["paths"]["image_dirs"][cohort],
        tokenizer,
        cfg["data"],
        cohort,
    )
    loader = make_dataloader(
        dataset,
        cfg["data"]["evaluation_batch_size"],
        cfg["data"]["num_workers"],
        shuffle=False,
        seed=seed,
    )

    model = VLINC(cfg["model"], cfg["paths"]["radiologic_dir"]).to(device)
    checkpoint_path = Path(cfg["paths"]["checkpoint_dir"]) / str(cfg["training"]["checkpoint_name"]).format(endpoint=cfg["data"]["endpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}")
    saved_endpoint = str(checkpoint.get("config", {}).get("data", {}).get("endpoint", "")).lower()
    requested_endpoint = str(cfg["data"]["endpoint"]).lower()
    if saved_endpoint and saved_endpoint != requested_endpoint:
        raise ValueError(
            f"Checkpoint endpoint is '{saved_endpoint}', but evaluation requested '{requested_endpoint}'."
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    outcomes, risks, identifiers = [], [], []
    with torch.no_grad():
        for batch in loader:
            identifiers.extend(batch["identifier"])
            tensor_batch = move_batch(batch, device)
            risks.append(model(tensor_batch).cpu().numpy())
            outcomes.append(batch["survival"].numpy())
    if not risks:
        raise RuntimeError("Evaluation loader produced no batches.")

    outcomes_array = np.concatenate(outcomes)
    risks_array = np.concatenate(risks).reshape(-1)
    metrics = bootstrap_c_index(
        outcomes_array,
        risks_array,
        seed=seed,
        iterations=int(cfg["evaluation"]["bootstrap_iterations"]),
    )
    ensure_directories(cfg["paths"]["output_dir"])
    output = {
        "cohort": cohort,
        "endpoint": cfg["data"]["endpoint"],
        "checkpoint": str(checkpoint_path.name),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_validation_c_index": checkpoint.get("val_c_index"),
        "metrics": metrics,
        "predictions": [
            {
                "identifier": identifier,
                "time": float(outcome[0]),
                "event": int(outcome[1]),
                "risk": float(risk),
            }
            for identifier, outcome, risk in zip(identifiers, outcomes_array, risks_array)
        ],
    }
    output_path = Path(cfg["paths"]["output_dir"]) / f"{cohort}_{cfg['data']['endpoint']}_results.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
    print(json.dumps(metrics, indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained V-LINC checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cohort", choices=["internal", "ispy1", "duke"], default="internal")
    args = parser.parse_args()
    main(args.config, args.cohort)
