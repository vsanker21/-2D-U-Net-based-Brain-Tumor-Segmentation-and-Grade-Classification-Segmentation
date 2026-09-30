# Changelog

## 2.0.0

Complete re-analysis. Results of version 1 are superseded and should not be used.

- Input channel: the file used as "T2" in version 1 (`*_T2.nii.gz`) contains T1ce. All models now use
  `*_FLAIR.nii.gz`, which contains T2 (see the channel table in `README.md`).
- Labels: official multi-class labels (NCR, ED, ET) replace single-channel NCR masks; slices are axial.
- Stage 1: 5-fold out-of-fold 2D U-Net within the 295 development patients; test masks from the five-model
  ensemble. Development and test patients no longer overlap in segmentation training.
- Stage 2: no test-set epoch selection (final-epoch weights); class imbalance handled once (weighted loss) instead
  of twice (sampler and loss); cross-validation restricted to the development set.
- New conditions: nnU-Net and ground-truth masks at test time, a classifier trained with ground-truth masks, a
  random-mask control and an in-sample mask control.
- Statistics: five seeds, stratified bootstrap CIs, DeLong and McNemar tests with Holm correction, PR-AUC for both
  positive classes.

## Version 1

Initial deposit of metric files.
