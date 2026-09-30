# Tumor-mask guidance for T2-only glioma grade classification (BraTS2020)

Code and per-patient results of a two-stage analysis restricted to the T2-weighted channel of the BraTS2020
training cohort:

1. **Segmentation.** A 2D U-Net trained with 5-fold out-of-fold (OOF) prediction in the development set, and an
   nnU-Net v2 (3d_fullres) benchmark, both evaluated on WT, TC and ET against the official labels.
2. **Grade classification.** A 3D CNN for the BraTS2020 HGG/LGG label, given T2 alone or T2 plus a binary
   whole-tumor mask. The mask comes from the OOF 2D U-Net, nnU-Net, the ground truth, or random noise (control).

Archived version: [10.5281/zenodo.23050626](https://doi.org/10.5281/zenodo.23050626)
(all versions: [10.5281/zenodo.22086674](https://doi.org/10.5281/zenodo.22086674)).

## Main results

Hold-out test set, n = 74 (59 HGG, 15 LGG); classification values are seed-ensemble means with 95% bootstrap CIs.

| Quantity | Value |
|---|---|
| WT Dice, 2D U-Net ensemble / nnU-Net | 0.807 / 0.877 |
| ROC-AUC, T2 only | 0.814 (0.702–0.913) |
| ROC-AUC, T2 + 2D U-Net mask | 0.714 (0.547–0.866) |
| ROC-AUC, all mask-guided variants | 0.677–0.742; none higher than T2 only (Holm-adjusted DeLong p = 1.00) |
| 5-fold CV in the development set (n = 295), mean fold ROC-AUC, T2 only / T2 + 2D U-Net mask | 0.782 / 0.707 |

All values can be recomputed from the files in `results/` with `src/classification_stats.py`.

## Layout

| Path | Content |
|---|---|
| `src/common.py` | Paths, cohort split and folds, image/label IO, Dice/HD95 |
| `src/stage1_oof.py`, `src/unet2d.py` | Stage 1: 5-fold OOF 2D U-Net, test-set ensemble, in-sample control masks |
| `src/nnunet_t2.py`, `src/nnunet_trainer/` | nnU-Net v2 dataset build, planning, training (fold 0), prediction, evaluation |
| `src/stage2.py` | Stage 2: classifier inputs, hold-out training (5 seeds), test-time mask substitution, CV, radiomics baseline |
| `src/classification_stats.py` | Bootstrap CIs, DeLong, McNemar, Holm, PR-AUC for both positive classes, CV summaries |
| `src/interpret_timing.py`, `src/time_nnunet.py` | Integrated Gradients summary; inference timing |
| `src/check_modality_channels.py`, `src/check_label_agreement.py` | Sequence identity of each NIfTI file; agreement of rebuilt and official labels |
| `src/data_prep/` | HDF5 to NIfTI conversion, multi-class label rebuild, official label alignment |
| `results/` | Cohort split, per-patient segmentation metrics, per-patient classifier probabilities for every condition and seed, statistics |
| `results/nnunet_Dataset502/` | nnU-Net plans, dataset fingerprint, fold split and fold-0 validation summary |
| `SHA256SUMS.txt` | Checksums of all files |

Imaging data, model weights and intermediate caches are not included.

## Data

BraTS2020 training cohort, 369 patients (293 HGG, 76 LGG; grade from `name_mapping.csv`), available from the
challenge organizers under the BraTS terms of use (https://www.med.upenn.edu/cbica/brats2020/). The scripts
expect:

- per-slice HDF5 files `volume_<id>_slice_<k>.h5` with `image` (240 × 240 × 4) and `mask` (240 × 240 × 3;
  NCR, ED, ET), together with `name_mapping.csv`;
- the official segmentation files `BraTS20_Training_XXX_seg.nii` (labels 1, 2, 4).

`src/data_prep/sync_official_labels.py` reorients the official labels to (155, 240, 240) and maps label 4 to 3.
For BraTS20_Training_355 no official segmentation file was available; the map rebuilt from the HDF5 masks
(`src/data_prep/rebuild_multiclass_labels.py`) was used. Rebuilt and official maps are voxel-identical in all
368 patients with both (`results/label_rebuild_vs_official.json`).

### Channel assignment

The HDF5 `image` channels are ordered FLAIR, T1, T1ce, T2. `convert_h5_to_nifti.py` writes them with the suffixes
`_T1`, `_T1ce`, `_T2`, `_FLAIR` in that order, so the suffixes do not name the content:

| File suffix | Content |
|---|---|
| `_T1.nii.gz` | FLAIR |
| `_T1ce.nii.gz` | T1 |
| `_T2.nii.gz` | T1ce |
| `_FLAIR.nii.gz` | **T2 (the only sequence used)** |

`common.TRUE_T2_SUFFIX = "_FLAIR.nii.gz"`. The assignment follows from tissue contrast relative to the official
labels (enhancing tumor bright only on T1ce; CSF bright on T2, suppressed on FLAIR) and is checked by
`src/check_modality_channels.py` (`results/modality_channel_check.json`).

## Environment

Python 3.13, PyTorch 2.7.1 (CUDA 11.8), nnU-Net v2 2.6.2 (`requirements.txt`); one NVIDIA RTX 5000 Ada (32 GB),
Windows 11. Copy `src/nnunet_trainer/nnUNetTrainer_250epochs_snap50.py` into
`nnunetv2/training/nnUNetTrainer/` before training nnU-Net. `src/nnunet_t2.py` calls the nnU-Net console
scripts from the `Scripts` folder of the active Python environment (Windows layout); adjust `SCRIPTS` on Linux.

Working directory: `$BTS_BASE` (default: the repository root).

```
$BTS_BASE/archive/BraTS2020_training_data/content/data/   *.h5, name_mapping.csv
$BTS_BASE/archive/3D Slices Sorted/                         NIfTI volumes written by convert_h5_to_nifti.py
$BTS_BASE/archive/3D Slices Sorted/masks_brats2020_official/
$BTS_BASE/archive/3D Slices Sorted/masks_brats2020_seg/
$BTS_BASE/external/brats2020_official/                      official *_seg.nii (or set BRATS2020_OFFICIAL_DIR)
$BTS_BASE/work/                                             outputs, caches and nnU-Net folders (created)
```

## Run order

From `src/`:

```
python data_prep/convert_h5_to_nifti.py
python data_prep/sync_official_labels.py
python data_prep/rebuild_multiclass_labels.py
python check_modality_channels.py
python check_label_agreement.py
python stage1_oof.py --cache
python stage1_oof.py --folds 0 1 2 3 4
python stage1_oof.py --ensemble --insample
python nnunet_t2.py --step all_train
python nnunet_t2.py --step predict
python nnunet_t2.py --step eval
python stage2.py --build-cache t2 gt m2d m2d_ins nn50 nn250
python stage2.py --holdout t2 gt rand m2d m2d_ins --seeds 42 43 44 45 46
python stage2.py --reeval
python stage2.py --cv --seeds 42 43 44
python stage2.py --holdout t2 --seeds 42 43 44 45 --balanced-sampler
python classification_stats.py
python interpret_timing.py --ig --timing
python time_nnunet.py
```

## Design

- Split: stratified 80:20 patient split (`random_state` 42) into a development set of 295 (234 HGG, 61 LGG) and a
  test set of 74 (59 HGG, 15 LGG). The test set is used only for final evaluation. StratifiedKFold (5, shuffled,
  `random_state` 42) within the development set defines the Stage-1 folds and the Stage-2 CV folds
  (`results/cohort.json`).
- Stage 1: 2D U-Net (1 input channel, 4 classes) on axial slices; each development patient's mask comes from the
  fold model that did not see that patient; test masks from the softmax ensemble of the five fold models.
- nnU-Net: Dataset502, 3d_fullres, fold 0 of the nnU-Net default split, 250 epochs with an extra checkpoint after
  epoch 50.
- Stage 2: 3D CNN on 128 × 128 × 96 inputs, weighted focal loss, AdamW, cosine warm restarts, 30 epochs, no model
  selection (final-epoch weights), five seeds for hold-out and three for CV.
- HGG is the positive class. PR-AUC is reported with HGG positive (no-skill 0.797) and with LGG positive
  (score 1 − P(HGG); no-skill 0.203). Holm correction is applied within the pre-specified family of comparisons
  with T2 only.

## Result files

- `results/stage2/holdout_<condition>_seed<seed>.json`: test patient IDs, labels and P(HGG) for each test-time mask
  source (`None` = T2 only, `m2d`, `nn50`, `nn250`, `gt`, `rand`).
- `results/stage2/cv_fold<k>_<condition>_seed<seed>.json`: validation-fold IDs, labels and P(HGG).
- `results/stage2_doublebalanced_recipe/`: T2-only hold-out runs with a class-balanced sampler in addition to the
  weighted loss (class-imbalance sensitivity analysis).
- `results/classification_stats.json`: classification metrics, CIs and paired tests.
- `results/seg_stage1_2d_trueT2.json`, `results/seg_nnunet_trueT2_test.json`, `results/stage1_fold*.json`,
  `results/stage1_insample_vs_oof.json`: per-patient Dice, HD and HD95 for WT, TC and ET.
- `results/ig_attribution_summary.json`, `results/timing_classifier.json`, `results/timing_nnunet.json`.

## License and citation

Released under CC BY 4.0. Please cite the archived version (`CITATION.cff`) and, for the data, the BraTS
reference publications:

- B. H. Menze et al., IEEE Trans Med Imaging 34(10):1993–2024, 2015, doi:10.1109/TMI.2014.2377694
- S. Bakas et al., Sci Data 4:170117, 2017, doi:10.1038/sdata.2017.117
- S. Bakas et al., arXiv:1811.02629, 2018, doi:10.48550/arXiv.1811.02629
