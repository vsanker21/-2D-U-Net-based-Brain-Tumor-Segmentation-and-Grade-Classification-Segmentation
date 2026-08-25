# Single-modality T2 glioma localization and grade classification (Project 1)

Exploratory two-stage pipeline for **jimaging-4516046** (MDPI *J. Imaging*):

1. **Stage 1:** slice-wise 2D U-Net tumor localization on T2 MRI
2. **Stage 2:** mask-guided 3D CNN HGG/LGG classification (predicted masks only at test time)

## Reproducibility artifacts

- **Processed metrics (hold-out, bootstrap/paired tests, 12- and 30-epoch CV):** [Zenodo DOI 10.5281/zenodo.22086675](https://doi.org/10.5281/zenodo.22086675)
- **Manuscript revision scripts:** see `scripts/` and `MDPI_Submission/` (local paths documented below)

## Repository layout

| Path | Purpose |
|------|---------|
| `project1_analysis_common.py` | Shared data loaders, models, split logic |
| `segmentation_model_loader.py` | 2D U-Net architecture + checkpoint loader |
| `run_project1_methodological_upgrade.py` | Main analysis driver (CV, benchmarks) |
| `project1_train_fair_t2_baseline.py` | Fair T2-only 3D classifier training |
| `project1_train_random_mask_ablation.py` | Random-mask negative control |
| `project1_official_seg_pipeline.py` | Official multi-class GT evaluation |
| `scripts/run_matched_cv_30epochs.py` | Matched 30-epoch repeated stratified CV |
| `Project1_Brain_Tumor_Segmentation_and_Classification.ipynb` | Original exploratory notebook |

## Environment

```bash
pip install -r requirements.txt
```

**Hardware used in the paper:** NVIDIA RTX 5000 Ada GPU, PyTorch 2.x, CUDA.

## Data and paths

BraTS2020 training data must be obtained from the [MICCAI BraTS challenge](https://www.med.upenn.edu/cbica/brats2020/).

The analysis scripts default to these local paths (edit `project1_analysis_common.py` if needed):

- `BASE`: project root with `archive/3D Slices Sorted/` and checkpoints
- `OUT_DIR`: writable output directory for JSON metrics and figures

Example layout:

```
BASE/
  archive/3D Slices Sorted/          # T2 NIfTI volumes
  archive/.../name_mapping.csv       # HGG/LGG labels
  segmentation_model_t2_best.pth     # Stage-1 checkpoint
OUT_DIR/
  classification_model_t2_fair_best.pth
  classification_model_maskguided_best.pth
  classification_repeated_cv_30epochs.json
```

## Key commands

```bash
# Repeated stratified CV (30 epochs/fold)
python scripts/run_matched_cv_30epochs.py

# Full methodological upgrade suite
python run_project1_methodological_upgrade.py --help
```

## Citation

If you use this code, please cite the MDPI manuscript (jimaging-4516046) and the Zenodo metrics deposit (10.5281/zenodo.22086675).

## Note on exploratory framing

This repository supports an **exploratory** study. Hold-out mask-guided gains were modest and not statistically significant; matched 30-epoch cross-validation did not clearly reproduce a mask-guided advantage over T2-only CNN.
