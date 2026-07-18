from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import RobertaTokenizerFast

from src.config import load_config
from src.dataset import SurvivalDataset, load_records, make_dataloader, split_records
from src.losses import cox_ph_loss
from src.metrics import c_index
from src.models import VLINC
from src.utils import ensure_directories, set_global_seed


def move_batch(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate(model: VLINC, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    risks, outcomes = [], []
    total_loss, total_events = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            risk = model(batch)
            event_count = int(batch["survival"][:, 1].sum().item())
            if event_count > 0:
                total_loss += float(cox_ph_loss(batch["survival"], risk)) * event_count
                total_events += event_count
            risks.append(risk.cpu().numpy())
            outcomes.append(batch["survival"].cpu().numpy())
    if not risks:
        raise RuntimeError("Validation loader produced no batches.")
    risk_array = np.concatenate(risks)
    outcome_array = np.concatenate(outcomes)
    mean_loss = total_loss / total_events if total_events > 0 else float("nan")
    return mean_loss, c_index(outcome_array, risk_array)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["seed"])
    set_global_seed(seed)
    requested_device = str(cfg["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device)

    ensure_directories(cfg["paths"]["checkpoint_dir"], cfg["paths"]["output_dir"])
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["paths"]["radiologic_dir"])
    records = load_records(cfg["paths"]["internal_metadata"])
    train_records, val_records, _ = split_records(records, cfg["data"], seed)

    train_dataset = SurvivalDataset(
        train_records,
        cfg["paths"]["image_dirs"]["internal"],
        tokenizer,
        cfg["data"],
        "internal",
    )
    val_dataset = SurvivalDataset(
        val_records,
        cfg["paths"]["image_dirs"]["internal"],
        tokenizer,
        cfg["data"],
        "internal",
    )
    train_loader = make_dataloader(
        train_dataset,
        cfg["data"]["batch_size"],
        cfg["data"]["num_workers"],
        shuffle=True,
        drop_last=True,
        seed=seed,
    )
    val_loader = make_dataloader(
        val_dataset,
        cfg["data"]["evaluation_batch_size"],
        cfg["data"]["num_workers"],
        shuffle=False,
        seed=seed,
    )

    model = VLINC(cfg["model"], cfg["paths"]["radiologic_dir"]).to(device)
    image_parameters = [parameter for parameter in model.image_encoder.parameters() if parameter.requires_grad]
    image_ids = {id(parameter) for parameter in image_parameters}
    fusion_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in image_ids
    ]
    if not image_parameters or not fusion_parameters:
        raise RuntimeError("Optimizer parameter groups are empty.")

    image_optimizer = Adam(
        image_parameters,
        lr=float(cfg["training"]["learning_rate_image"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    fusion_optimizer = Adam(
        fusion_parameters,
        lr=float(cfg["training"]["learning_rate_fusion"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    schedulers = [
        StepLR(image_optimizer, int(cfg["training"]["scheduler_step_size"]), float(cfg["training"]["scheduler_gamma"])),
        StepLR(fusion_optimizer, int(cfg["training"]["scheduler_step_size"]), float(cfg["training"]["scheduler_gamma"])),
    ]

    checkpoint_path = Path(cfg["paths"]["checkpoint_dir"]) / str(cfg["training"]["checkpoint_name"]).format(endpoint=cfg["data"]["endpoint"])
    history = []
    best_val_c_index = float("-inf")
    patience = int(cfg["training"]["early_stopping_patience"])
    epochs_without_improvement = 0

    for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
        model.train()
        train_loss_sum, train_event_count = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['training']['epochs']}"):
            batch = move_batch(batch, device)
            event_count = int(batch["survival"][:, 1].sum().item())
            if event_count == 0:
                continue
            image_optimizer.zero_grad(set_to_none=True)
            fusion_optimizer.zero_grad(set_to_none=True)
            risk = model(batch)
            loss = cox_ph_loss(batch["survival"], risk)
            loss.backward()
            image_optimizer.step()
            fusion_optimizer.step()
            train_loss_sum += float(loss.detach()) * event_count
            train_event_count += event_count

        for scheduler in schedulers:
            scheduler.step()

        if train_event_count == 0:
            raise RuntimeError("No observed events were present in any training batch for this epoch.")
        train_loss = train_loss_sum / train_event_count
        val_loss, val_ci = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_c_index": val_ci,
            "image_learning_rate": image_optimizer.param_groups[0]["lr"],
            "fusion_learning_rate": fusion_optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row, indent=2))

        if val_ci > best_val_c_index:
            best_val_c_index = val_ci
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "val_c_index": val_ci,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping after {epoch} epochs. Best validation C-index: {best_val_c_index:.4f}")
                break

    history_path = Path(cfg["paths"]["output_dir"]) / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and validate V-LINC.")
    parser.add_argument("--config", default="configs/default.yaml")
    main(parser.parse_args().config)
