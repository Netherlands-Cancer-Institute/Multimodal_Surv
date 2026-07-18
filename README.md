# V-LINC: Individualized Breast Cancer Prognosis

This repository contains the preprocessing, training, validation, and evaluation code for **V-LINC**, a multimodal survival model for individualized breast cancer prognosis. The model integrates breast DCE-MRI, radiology reports, structured clinical variables, treatment information, and optional mutation features.

### V-LINC framework
![image](https://github.com/Netherlands-Cancer-Institute/Multimodal_Surv/blob/main/Figures/Flowchart.png)
Note: Study setting and V-LINC framework. a, Clinical context of pretreatment breast MRI acquisition and subsequent treatment pathways, including neoadjuvant therapy and surgery-first management. b, Geographic distribution of the in-house cohort and external validation cohorts. c, Data sources used for prognostic modeling, including DCE-MRI and radiology reports from radiological evaluation, clinicopathological variables from pathological evaluation and longitudinal follow-up for survival endpoints. d, Overview of the V-LINC framework. DCE-MRI, radiology reports and structured clinical variables are encoded separately, combined with treatment-context prompts and integrated through an attention-based fusion network to estimate patient-level risk.

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

```bash
pip install -r requirements.txt
```


## RadioLOGIC

Place the pretrained report encoder and tokenizer files in:

```text
RadioLOGIC/
```

Contains:

```text
RadioLOGIC/
├── config.json
├── vocab.json
├── merges.txt
└── pytorch_model.bin
```

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
      "T_stage_post": "X",
      "N_stage_post": "X",
      "M_stage_post": "0",
      "EPH_surv": [90, 90, 0],
      "tumor_types": "Infiltrerend ductaal carcinoom",
      "family_history": "no.",
      "mutation": {
        "BRCA1": -1,
        "BRCA2": -1,
        "PALB2": -1,
        "TP53": -1,
        "PTEN": -1,
        "CDH1": -1,
        "ATM": -1,
        "CHEK2": -1,
        "BARD1": -1,
        "BRIP1": -1,
        "RAD51C": -1,
        "RAD51D": -1,
        "STK11": -1,
        "PIK3CA": -1,
        "AKT1": -1,
        "ERBB2": 1,
        "ESR1": 0,
        "NTRK1": -1,
        "NTRK2": -1,
        "NTRK3": -1,
        "CCND1": -1,
        "FGFR1": -1,
        "GATA3": -1,
        "MAP3K1": -1,
        "NF1": -1,
        "ARID1A": -1,
        "KMT2C": -1,
        "PIK3R1": -1,
        "MYC": -1,
        "RB1": -1
      }
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

### Results

* Overall survival
![image](https://github.com/Netherlands-Cancer-Institute/Multimodal_Surv/blob/main/Figures/Results_OS.png)
Note: Overall survival stratification across internal and external cohorts. a, Distribution of survival status across the in-house and external cohorts used for overall survival analysis. b–d, Kaplan–Meier curves for V-LINC-defined low-risk and high-risk groups in the NKI training, validation and test sets. e–h, Kaplan–Meier curves for the ISPY1, DUKE, Sun Yat-Sen and RUMC external test sets.

* Interpretation
![image](https://github.com/Netherlands-Cancer-Institute/Multimodal_Surv/blob/main/Figures/Analysis_OS.png)
Note: Model interpretation and time-dependent performance for overall survival. a, Representative DCE-MRI examples with lesion masks and Grad-CAM maps before and after contrast enhancement. b, Token-level attention maps from radiology reports, with higher-intensity tokens indicating greater contribution to the text-derived representation. c, Gradient-based attribution of structured clinical variables across patients. d, Decision curve analysis for 5-year overall survival. e, Volcano plot showing feature differences between V-LINC-defined high-risk and low-risk groups across imaging, report, clinical and treatment-prompt features. f, Time-dependent receiver operating characteristic curves for overall survival from 1 to 10 years.

## Citation

Citation information will be added after publication.

### Contact details
If you have any questions please contact us. 

Email: tianyu.zhang@radboudumc.nl / t.zhang@nki.nl (Dr. Tianyu Zhang); ritse.mann@radboudumc.nl (Prof. dr. Ritse M. Mann) 

Links: [Radboud University Medical Center](https://www.radboudumc.nl/en/patient-care), and [Netherlands Cancer Institute](https://www.nki.nl/)

<img src="https://github.com/Netherlands-Cancer-Institute/Multimodal_attention_DeepLearning/blob/main/Figures/RadboudUMC.png" width="231" height="74.58"/> <img src="https://github.com/Netherlands-Cancer-Institute/Multimodal_attention_DeepLearning/blob/main/Figures/NKI.png" width="166.98" height="87.12"/> 

