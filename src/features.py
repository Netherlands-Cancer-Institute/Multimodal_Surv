from __future__ import annotations

import re
from typing import Any, Dict, Iterable

import numpy as np

MUTATION_GENES = [
    "BRCA1", "BRCA2", "PALB2", "TP53", "PTEN", "CDH1", "ATM", "CHEK2",
    "BARD1", "BRIP1", "RAD51C", "RAD51D", "STK11", "PIK3CA", "AKT1",
    "ERBB2", "ESR1", "NTRK1", "NTRK2", "NTRK3", "CCND1", "FGFR1",
    "GATA3", "MAP3K1", "NF1", "ARID1A", "KMT2C", "PIK3R1", "MYC", "RB1",
]

THERAPY_KEYS = [
    "Neoadjuvant Radiation Therapy",
    "Adjuvant Radiation Therapy",
    "Neoadjuvant Chemotherapy",
    "Adjuvant Chemotherapy",
    "Neoadjuvant Endocrine Therapy Medications",
    "Adjuvant Endocrine Therapy Medications",
    "Neoadjuvant Anti-Her2 Neu Therapy",
    "Adjuvant Anti-Her2 Neu Therapy",
]

MISSING_STRINGS = {"", "-", "-1", "NONE", "NAN", "NA", "N/A", "UNKNOWN"}


def _normalize_stage(value: Any, n_post: bool = False) -> str:
    """Reproduce the coarse stage mapping used in the original loader."""
    if value is None:
        return "-1"
    text = str(value).strip().upper().replace(" ", "")
    if text in MISSING_STRINGS:
        return "-1"

    # Numeric JSON values are often serialized as 1.0, 2.0, and so on.
    numeric_match = re.fullmatch(r"([0-4])(?:\.0+)?", text)
    if numeric_match:
        stage = numeric_match.group(1)
    else:
        stage = next((token for token in ("0", "1", "2", "3", "4") if token in text), None)
        if stage is None:
            if "IS" in text:
                stage = "IS"
            elif "X" in text:
                stage = "X"
            else:
                return "-1"

    # The original pipeline marked post-treatment N stages containing 1 as missing.
    if n_post and stage == "1":
        return "-1"
    return stage


def encode_clinical_features(item: Dict[str, Any]) -> np.ndarray:
    """Encode the 12 clinical variable groups as a 25 x 12 matrix."""
    matrix = np.zeros((25, 12), dtype=np.float32)
    categories = {
        0: ["0", "1", "1A", "1B", "1C", "1M", "1MI", "2", "3", "4", "4A", "4B", "4D", "IS", "X"],
        1: ["0", "0IS", "0S", "1", "1MS", "1S", "1BS", "2A", "2B", "3", "3A", "3B", "3BS", "3C", "X"],
        2: ["0", "1", "X"],
        3: ["0", "1", "1A", "1B", "1C", "1MI", "2", "3", "4B", "IS", "X", "Y0", "Y1", "Y1A", "Y1B", "Y1C", "Y1MI", "Y2", "Y3", "Y4A", "Y4B", "Y4D", "YIS", "YX"],
        4: ["0", "0I", "0IS", "0S", "1", "1A", "1AS", "1B", "1BS", "1B1", "1B2", "1B3", "1B4", "1M", "1MI", "1MS", "2", "2A", "2B", "3A", "3B", "3C", "X"],
        5: ["0", "1", "X"],
    }
    values = [
        _normalize_stage(item.get("T_stage")),
        _normalize_stage(item.get("N_stage")),
        _normalize_stage(item.get("M_stage")),
        _normalize_stage(item.get("T_stage_post")),
        _normalize_stage(item.get("N_stage_post"), n_post=True),
        _normalize_stage(item.get("M_stage_post")),
    ]
    for column, value in enumerate(values):
        if value in categories[column]:
            matrix[categories[column].index(value), column] = 1.0

    family_history = str(item.get("family_history", "")).lower()
    if "kanker" in family_history or "cancer" in family_history:
        matrix[0, 6] = 1.0

    age = item.get("AGE", -1)
    try:
        age_value = float(age)
        if np.isfinite(age_value) and age_value >= 0:
            matrix[min(int(age_value / 4), 24), 7] = 1.0
    except (TypeError, ValueError):
        pass

    eph = item.get("EPH_surv", [-1, -1, -1])
    if not isinstance(eph, (list, tuple)) or len(eph) < 3:
        eph = [-1, -1, -1]
    for column, value in zip((8, 9, 10), eph[:3]):
        try:
            numeric = int(float(value))
        except (TypeError, ValueError):
            continue
        if numeric == -1:
            continue
        if column in (8, 9):
            matrix[1 if 0 < numeric / 4 < 25 else 0, column] = 1.0
        else:
            matrix[1 if 2 < numeric < 8 else 0, column] = 1.0

    tumor = str(item.get("tumor_types", "")).lower()
    rules: Iterable[tuple[int, bool]] = [
        (0, "ductaal" in tumor and "infiltrerend ductaal" not in tumor and "intraductaal carcinoom" not in tumor and "ductaal carcinoma in situ" not in tumor),
        (1, "infiltrerend ductaal" in tumor and "intraductaal carcinoom" not in tumor),
        (2, "lobulair" in tumor and "infiltrerend lobulair" not in tumor),
        (3, "infiltrerend lobulair" in tumor),
        (4, "tubular" in tumor),
        (5, "mucineus" in tumor),
        (6, "micropapillair" in tumor),
        (7, "papillair" in tumor and "micropapillair" not in tumor and "intraductaal papillair adenocarcinoom" not in tumor),
        (8, "ductaal carcinoma in situ" in tumor or "intraductaal carcinoom" in tumor or "intraductaal papillair adenocarcinoom" in tumor),
    ]
    for row, matched in rules:
        if matched:
            matrix[row, 11] = 1.0
    if tumor and matrix[:, 11].sum() == 0:
        matrix[9, 11] = 1.0
    return matrix


def encode_mutations(item: Dict[str, Any]) -> np.ndarray:
    mutation = item.get("mutation") or {}
    values = []
    for gene in MUTATION_GENES:
        value = mutation.get(gene, 0)
        try:
            numeric = int(value) if value is not None else 0
        except (TypeError, ValueError):
            numeric = 0
        values.append(0 if numeric == -1 else numeric)
    return np.asarray(values, dtype=np.float32)


def encode_therapies(item: Dict[str, Any]) -> np.ndarray:
    values = []
    for key in THERAPY_KEYS:
        value = item.get(key, 0)
        try:
            numeric = float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            numeric = 0.0
        values.append(0.0 if numeric == -1 else numeric)
    return np.asarray(values, dtype=np.float32)
