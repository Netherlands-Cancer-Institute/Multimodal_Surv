from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizerFast

from .features import encode_clinical_features, encode_mutations, encode_therapies


def clip_and_normalize(array: np.ndarray, clip_max: float) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D MRI volume, received shape {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError("MRI volume contains NaN or infinite values.")
    array = np.clip(array, a_min=None, a_max=float(clip_max))
    minimum, maximum = float(array.min()), float(array.max())
    if maximum <= minimum:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - minimum) / (maximum - minimum)).astype(np.float32)


def clean_report(report: Any) -> str:
    text = str(report or "None").replace("\r", " ").replace("\n", " ").strip()
    clinical_index = text.find("Klinische")
    if clinical_index >= 0:
        report_index = text.find("Verslag", clinical_index)
        if report_index >= 0:
            text = text[report_index:]
    return text or "None"


def endpoint_keys(endpoint: str) -> Tuple[str, str]:
    return ("label_drm", "time_drm") if endpoint.lower() == "dfs" else ("label", "time")


class SurvivalDataset(Dataset):
    def __init__(
        self,
        records: List[Dict[str, Any]],
        image_dir: Union[str, Path],
        tokenizer: RobertaTokenizerFast,
        data_config: Dict[str, Any],
        cohort: str,
    ) -> None:
        if not records:
            raise ValueError(f"No records were supplied for cohort '{cohort}'.")
        self.records = records
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.cfg = data_config
        self.cohort = cohort

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _case_id(item: Dict[str, Any]) -> str:
        source = item.get("image") or item.get("identifier") or item.get("ID_PI")
        if source is None:
            raise KeyError("Each metadata record must contain image, identifier, or ID_PI.")
        name = Path(str(source)).name
        return name[:-7] if name.endswith(".nii.gz") else Path(name).stem

    def _image_paths(self, item: Dict[str, Any]) -> Tuple[Path, Path, Path]:
        case_id = self._case_id(item)
        templates = self.cfg["image_filename_templates"]
        if self.cohort not in templates:
            raise KeyError(f"No image filename templates are configured for cohort '{self.cohort}'.")
        cohort_templates = templates[self.cohort]
        required = ("dce1", "dce2", "mask")
        missing_keys = [key for key in required if key not in cohort_templates]
        if missing_keys:
            raise KeyError(f"Missing filename template(s) for cohort '{self.cohort}': {missing_keys}")
        return tuple(
            self.image_dir / str(cohort_templates[key]).format(case_id=case_id)
            for key in required
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        dce1_path, dce2_path, mask_path = self._image_paths(item)
        missing = [str(path) for path in (dce1_path, dce2_path, mask_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing required MRI file(s): " + ", ".join(missing))

        dce1 = clip_and_normalize(nib.load(str(dce1_path)).get_fdata(dtype=np.float32), self.cfg["image_clip_max"])
        dce2 = clip_and_normalize(nib.load(str(dce2_path)).get_fdata(dtype=np.float32), self.cfg["image_clip_max"])
        mask = np.asarray(nib.load(str(mask_path)).get_fdata(), dtype=np.int16)
        if dce1.shape != dce2.shape or dce1.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch for {self._case_id(item)}: dce1={dce1.shape}, "
                f"dce2={dce2.shape}, mask={mask.shape}."
            )
        expected_shape = tuple(int(value) for value in self.cfg["expected_image_shape"])
        if dce1.shape != expected_shape:
            raise ValueError(
                f"Unexpected image shape for {self._case_id(item)}: {dce1.shape}; "
                f"expected {expected_shape}."
            )

        spatial_weight = np.where(mask == 1, float(self.cfg["tumor_mask_weight"]), 1.0).astype(np.float32)
        report_tokens = self.tokenizer(
            clean_report(item.get("reports_surv")),
            padding="max_length",
            truncation=True,
            max_length=int(self.cfg["max_report_length"]),
            return_tensors="pt",
        )

        label_key, time_key = endpoint_keys(str(self.cfg["endpoint"]))
        if label_key not in item or time_key not in item:
            raise KeyError(
                f"Record '{item.get('identifier', index)}' is missing endpoint fields "
                f"'{label_key}' and/or '{time_key}'."
            )
        try:
            event = int(item[label_key])
            time = float(item[time_key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid survival values in record '{item.get('identifier', index)}'.") from exc
        if event not in (0, 1):
            raise ValueError(f"Event indicator must be 0 or 1, received {event}.")
        if not np.isfinite(time) or time < 0:
            raise ValueError(f"Survival time must be non-negative and finite, received {time}.")

        primary = str(item.get("primary_therapy", "NA")).strip().upper()
        return {
            "image1": torch.from_numpy((dce1 * spatial_weight)[None]),
            "image2": torch.from_numpy((dce2 * spatial_weight)[None]),
            "clinical_features": torch.from_numpy(encode_clinical_features(item)),
            "mutation_features": torch.from_numpy(encode_mutations(item)),
            "therapy_features": torch.from_numpy(encode_therapies(item)),
            "input_ids": report_tokens["input_ids"].squeeze(0),
            "attention_mask": report_tokens["attention_mask"].squeeze(0),
            "prompt_type": torch.tensor(0 if primary == "NA" else 1, dtype=torch.long),
            "survival": torch.tensor([time, event], dtype=torch.float32),
            "identifier": str(item.get("identifier", self._case_id(item))),
        }


def load_records(metadata_path: Union[str, Path]) -> List[Dict[str, Any]]:
    path = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload.get("training")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Metadata must contain a non-empty 'training' list: {path}")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every entry in 'training' must be a JSON object: {path}")
    return records


def split_records(
    records: List[Dict[str, Any]], data_cfg: Dict[str, Any], seed: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_key = str(data_cfg.get("split_group_key", "identifier"))
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for index, record in enumerate(records):
        group_value = record.get(group_key) or record.get("identifier") or record.get("ID_PI")
        if group_value is None:
            group_value = f"__record_{index}"
        groups.setdefault(str(group_value), []).append(record)

    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    train_end = int(len(group_ids) * float(data_cfg["train_fraction"]))
    val_end = train_end + int(len(group_ids) * float(data_cfg["val_fraction"]))
    train_ids, val_ids, test_ids = group_ids[:train_end], group_ids[train_end:val_end], group_ids[val_end:]
    if min(len(train_ids), len(val_ids), len(test_ids)) == 0:
        raise ValueError(
            f"The configured split produced an empty patient group subset: train={len(train_ids)}, "
            f"validation={len(val_ids)}, test={len(test_ids)}."
        )

    def flatten(ids: List[str]) -> List[Dict[str, Any]]:
        return [record for group_id in ids for record in groups[group_id]]

    return flatten(train_ids), flatten(val_ids), flatten(test_ids)


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
    seed: int = 42,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(num_workers) > 0,
        drop_last=drop_last,
        generator=generator,
    )
