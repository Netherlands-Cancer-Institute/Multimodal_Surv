# V-LINC: Individualized Breast Cancer Prognosis

This repository contains the preprocessing, training, validation, and evaluation code for **V-LINC**, a multimodal survival model for individualized breast cancer prognosis. The model integrates breast DCE-MRI, radiology reports, structured clinical variables, treatment information, and optional mutation features.

The code supports:

- Overall survival (OS)
- Disease-free survival (DFS)
- Internal testing
- External evaluation on Duke and I-SPY1

## Repository structure

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── internal_survival.example.json
│   ├── duke_survival.json
│   └── ispy1_survival.json
├── preprocessing/
│   └── preprocess_mri.py
├── RadioLOGIC/
├── src/
│   ├── config.py
│   ├── dataset.py
│   ├── features.py
│   ├── losses.py
│   ├── metrics.py
│   ├── models.py
│   └── utils.py
├── validate_setup.py
├── train.py
├── test.py
└── requirements.txt
```

## Installation

Python 3.9 or later is recommended.

```bash
pip install -r requirements.txt
```


## RadioLOGIC

Place the pretrained report encoder and tokenizer files in:

```text
RadioLOGIC/
```

A typical directory contains:

```text
RadioLOGIC/
├── config.json
├── tokenizer_config.json
├── tokenizer.json
├── vocab.json
├── merges.txt
└── pytorch_model.bin
```

Large model files such as `pytorch_model.bin` should be uploaded with Git LFS or hosted separately.

## MRI segmentation and preprocessing

Tumor masks were obtained using our breast MRI segmentation model:

https://huggingface.co/spaces/zhang0319/breast-mri-seg

The preprocessing script expects one directory per case:

```text
CASE_ID/
├── dce1.nii.gz
├── dce2.nii.gz
└── seg.nii.gz
```

Run preprocessing with:

```bash
python preprocessing/preprocess_mri.py \
  --input-root /path/to/segmented_cases \
  --output-root /path/to/preprocessed_cases
```

The script selects the tumor-containing breast, applies the configured crop, and center-crops or pads the volumes to `160 × 160 × 160`.

## Metadata format

Each metadata file must contain a `training` list:

```json
{
  "name": "cohort_name",
  "training": [
    {
      "identifier": "CASE_ID",
      "time": 60,
      "label": 0,
      "time_drm": 48,
      "label_drm": 1,
      "reports_surv": "Radiology report text",
      "AGE": 55,
      "T_stage": "2",
      "N_stage": "1",
      "M_stage": "0",
      "EPH_surv": [90, 90, 0],
      "tumor_types": "Infiltrerend ductaal carcinoom",
      "family_history": "no."
    }
  ]
}
```

MRI file locations and filename patterns are configured separately for each cohort in `configs/default.yaml`.

## Configuration

Edit:

```text
configs/default.yaml
```

Set the following before running the code:

- metadata paths
- MRI directories
- RadioLOGIC directory
- output and checkpoint directories
- endpoint (`os` or `dfs`)
- batch size, learning rate, and number of epochs

## Validate the setup

Check metadata and configuration:

```bash
python validate_setup.py --config configs/default.yaml
```

Also verify RadioLOGIC:

```bash
python validate_setup.py \
  --config configs/default.yaml \
  --check-radiologic
```

Verify all MRI files:

```bash
python validate_setup.py \
  --config configs/default.yaml \
  --check-radiologic \
  --check-images
```

## Training

```bash
python train.py --config configs/default.yaml
```

The best checkpoint is selected using validation C-index and saved in the configured checkpoint directory.

## Evaluation

Internal evaluation:

```bash
python test.py --config configs/default.yaml --cohort internal
```

External evaluation:

```bash
python test.py --config configs/default.yaml --cohort duke
python test.py --config configs/default.yaml --cohort ispy1
```


## Citation

Citation information will be added after publication.

## License

License information will be added after institutional approval.
