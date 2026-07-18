from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src.config import load_config
from src.dataset import SurvivalDataset, endpoint_keys, load_records, split_records


def validate_records(records: List[Dict], endpoint: str, cohort: str) -> None:
    label_key, time_key = endpoint_keys(endpoint)
    identifiers = []
    events = Counter()
    for index, record in enumerate(records):
        identifier = str(record.get("identifier") or record.get("ID_PI") or index)
        identifiers.append(identifier)
        if label_key not in record or time_key not in record:
            raise KeyError(f"{cohort}: record '{identifier}' is missing {label_key} or {time_key}.")
        event = int(record[label_key])
        time = float(record[time_key])
        if event not in (0, 1):
            raise ValueError(f"{cohort}: record '{identifier}' has event={event}; expected 0 or 1.")
        if time < 0:
            raise ValueError(f"{cohort}: record '{identifier}' has negative time={time}.")
        events[event] += 1
    duplicate_groups = sum(count > 1 for count in Counter(identifiers).values())
    print(
        f"{cohort}: {len(records)} records, events={events[1]}, censored={events[0]}, "
        f"repeated identifiers={duplicate_groups}"
    )


def main(config_path: str, check_images: bool, check_radiologic: bool) -> None:
    cfg = load_config(config_path)
    endpoint = str(cfg["data"]["endpoint"])

    internal_path = Path(cfg["paths"]["internal_metadata"])
    if internal_path.is_file():
        internal_records = load_records(internal_path)
        validate_records(internal_records, endpoint, "internal")
        train_records, val_records, test_records = split_records(internal_records, cfg["data"], int(cfg["seed"]))
        print(f"internal split: train={len(train_records)}, validation={len(val_records)}, test={len(test_records)}")
    else:
        print(f"internal: skipped because metadata is not present: {internal_path}")

    for cohort, path in cfg["paths"]["external_metadata"].items():
        records = load_records(path)
        validate_records(records, endpoint, cohort)

    if check_radiologic:
        from transformers import RobertaModel, RobertaTokenizerFast

        model_dir = cfg["paths"]["radiologic_dir"]
        tokenizer = RobertaTokenizerFast.from_pretrained(model_dir)
        model = RobertaModel.from_pretrained(model_dir, add_pooling_layer=False)
        print(f"RadioLOGIC: loaded successfully, hidden_size={model.config.hidden_size}")
    else:
        tokenizer = None

    if check_images:
        if tokenizer is None:
            from transformers import RobertaTokenizerFast

            tokenizer = RobertaTokenizerFast.from_pretrained(cfg["paths"]["radiologic_dir"])
        for cohort, image_dir in cfg["paths"]["image_dirs"].items():
            metadata_path = internal_path if cohort == "internal" else Path(cfg["paths"]["external_metadata"][cohort])
            if not metadata_path.is_file():
                print(f"{cohort}: image check skipped because metadata is unavailable.")
                continue
            records = load_records(metadata_path)
            dataset = SurvivalDataset(records, image_dir, tokenizer, cfg["data"], cohort)
            for index in range(len(dataset)):
                _ = dataset[index]
            print(f"{cohort}: all {len(dataset)} image triplets passed loading and shape checks.")

    print("Setup validation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate configuration, metadata, RadioLOGIC, and MRI files.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--check-images", action="store_true")
    parser.add_argument("--check-radiologic", action="store_true")
    args = parser.parse_args()
    main(args.config, args.check_images, args.check_radiologic)
