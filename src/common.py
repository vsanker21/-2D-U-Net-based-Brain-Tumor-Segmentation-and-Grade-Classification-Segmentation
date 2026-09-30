#!/usr/bin/env python3
"""Shared paths, cohort split, image/label IO and segmentation metrics (T2 channel of BraTS2020).

Channel mapping (verified in check_modality_channels.py):
the H5->NIfTI converter assumed [T1, T1ce, T2, FLAIR] but the H5 order is [FLAIR, T1, T1ce, T2].
Therefore on disk:  *_FLAIR.nii.gz = true T2,  *_T2.nii.gz = T1ce,  *_T1.nii.gz = FLAIR,  *_T1ce.nii.gz = T1.

Volumes are stored as (155, 240, 240) = (axial z, y, x); axial slices are vol[z].
Labels: official MICCAI segmentation remapped to {0,1,2,3} (NCR, ED, ET); the H5-rebuilt map is used
only for BraTS20_Training_355 (no official file; rebuilt maps agree with official ones voxel-wise).
Grade coding: HGG = 1 (positive class), LGG = 0.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from sklearn.model_selection import StratifiedKFold, train_test_split

BASE = Path(os.environ.get("BTS_BASE", Path(__file__).resolve().parents[1]))
DATA_DIR = BASE / "archive" / "3D Slices Sorted"
OFFICIAL_MASK_DIR = DATA_DIR / "masks_brats2020_official"
SEG_MASK_DIR = DATA_DIR / "masks_brats2020_seg"
GRADE_CSV = BASE / "archive" / "BraTS2020_training_data" / "content" / "data" / "name_mapping.csv"

WORK_DIR = BASE / "work"
OUT = WORK_DIR / "outputs"
CACHE = WORK_DIR / "cache"
T2_CACHE = CACHE / "t2_fp16"
LAB_CACHE = CACHE / "lab_u8"
MASK_CACHE = CACHE / "masks"          # predicted label maps (uint8, full resolution)
CLS_CACHE = CACHE / "cls"             # resized 128x128x96 arrays for Stage 2

TRUE_T2_SUFFIX = "_FLAIR.nii.gz"      # file whose content is the T2-weighted sequence
RANDOM_STATE = 42
N_STAGE1_FOLDS = 5
CLASSIFY_SHAPE = (128, 128, 96)       # (H, W, D)

for _p in (OUT, T2_CACHE, LAB_CACHE, MASK_CACHE, CLS_CACHE):
    _p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- cohort / splits
def load_grade_mapping() -> dict[str, str]:
    df = pd.read_csv(GRADE_CSV)
    return {row["BraTS_2020_subject_ID"]: row["Grade"] for _, row in df.iterrows()}


def list_patient_ids(grade_mapping: dict[str, str]) -> list[int]:
    ids = []
    for f in sorted(DATA_DIR.glob(f"*{TRUE_T2_SUFFIX}")):
        pid = int(f.name.split("_")[2])
        if f"BraTS20_Training_{pid:03d}" in grade_mapping:
            ids.append(pid)
    return sorted(set(ids))


def patient_labels(pids: Sequence[int], grade_mapping: dict[str, str]) -> list[int]:
    return [1 if grade_mapping[f"BraTS20_Training_{p:03d}"] == "HGG" else 0 for p in pids]


def cohort() -> dict:
    """Stratified 80:20 patient split (random_state 42) + five Stage-1 folds in the development set."""
    gm = load_grade_mapping()
    pids = list_patient_ids(gm)
    labels = patient_labels(pids, gm)
    tr, te, ytr, yte = train_test_split(pids, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels)
    skf = StratifiedKFold(n_splits=N_STAGE1_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    folds = []
    for k, (a, b) in enumerate(skf.split(tr, ytr)):
        folds.append({"fold": k, "train": [int(tr[i]) for i in a], "heldout": [int(tr[i]) for i in b]})
    label_of = {int(p): int(y) for p, y in zip(pids, labels)}
    return {"all": [int(p) for p in pids], "train": [int(p) for p in tr], "test": [int(p) for p in te],
            "label_of": label_of, "folds": folds}


def fold_of_train_patient(c: dict) -> dict[int, int]:
    out = {}
    for f in c["folds"]:
        for p in f["heldout"]:
            out[p] = f["fold"]
    return out


# ----------------------------------------------------------------------------- image / label IO
def _load_true_t2_raw(pid: int) -> np.ndarray:
    return nib.load(DATA_DIR / f"BraTS20_Training_{pid:03d}{TRUE_T2_SUFFIX}").get_fdata(dtype=np.float32)


def load_t2(pid: int) -> np.ndarray:
    """True-T2 volume, per-volume min-max scaled to [0,1], float32 (155,240,240). Cached as fp16 .npy."""
    fp = T2_CACHE / f"{pid:03d}.npy"
    if fp.exists():
        return np.load(fp).astype(np.float32)
    v = _load_true_t2_raw(pid)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    np.save(fp, v.astype(np.float16))
    return v


def load_label(pid: int) -> np.ndarray:
    """Official multi-class label map {0,1,2,3} uint8 (155,240,240). Cached."""
    fp = LAB_CACHE / f"{pid:03d}.npy"
    if fp.exists():
        return np.load(fp)
    p = OFFICIAL_MASK_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz"
    if not p.exists():
        p = SEG_MASK_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz"
    lab = np.asarray(nib.load(p).dataobj).astype(np.uint8)
    lab = np.squeeze(lab)
    assert set(np.unique(lab)).issubset({0, 1, 2, 3}), (pid, np.unique(lab))
    np.save(fp, lab)
    return lab


def label_source(pid: int) -> str:
    return "official" if (OFFICIAL_MASK_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz").exists() else "h5_rebuilt"


def resize_to(vol: np.ndarray, shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    f = tuple(s / v for s, v in zip(shape, vol.shape))
    return zoom(vol.astype(np.float32), f, order=order).astype(np.float32)


def to_classifier_grid(vol_zyx: np.ndarray) -> np.ndarray:
    """(155,240,240) -> (128,128,96): transpose to (y,x,z) so H,W,D = in-plane, in-plane, axial."""
    return resize_to(np.transpose(vol_zyx, (1, 2, 0)), CLASSIFY_SHAPE, order=1)


# ----------------------------------------------------------------------------- segmentation metrics
REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}


def region_masks(lab: np.ndarray) -> dict[str, np.ndarray]:
    return {r: np.isin(lab, c) for r, c in REGIONS.items()}


def dice_bin(a: np.ndarray, b: np.ndarray) -> float:
    sa, sb = int(a.sum()), int(b.sum())
    if sa == 0 and sb == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / (sa + sb))


def hd95_and_hd(a: np.ndarray, b: np.ndarray, spacing=(1.0, 1.0, 1.0), max_points: int = 20000,
                seed: int = 0) -> tuple[float, float]:
    """Symmetric Hausdorff (max) and HD95 between surfaces of binary masks (mm).
    BraTS convention for empty regions: both empty -> 0 mm; exactly one empty -> 373.13 mm."""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree

    ea, eb = a.sum() == 0, b.sum() == 0
    if ea and eb:
        return 0.0, 0.0
    if ea or eb:
        return 373.13, 373.13
    sa = np.argwhere(a & ~binary_erosion(a)) * np.asarray(spacing)
    sb = np.argwhere(b & ~binary_erosion(b)) * np.asarray(spacing)
    rng = np.random.default_rng(seed)
    if len(sa) > max_points:
        sa = sa[rng.choice(len(sa), max_points, replace=False)]
    if len(sb) > max_points:
        sb = sb[rng.choice(len(sb), max_points, replace=False)]
    da, _ = cKDTree(sb).query(sa)
    db, _ = cKDTree(sa).query(sb)
    d = np.concatenate([da, db])
    return float(d.max()), float(np.percentile(d, 95))


def seg_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pm, gm = region_masks(pred), region_masks(gt)
    out = {}
    for r in REGIONS:
        out[f"dice_{r}"] = dice_bin(pm[r], gm[r])
        hd, hd95 = hd95_and_hd(pm[r], gm[r])
        out[f"hd_{r}"] = hd
        out[f"hd95_{r}"] = hd95
        out[f"gt_empty_{r}"] = bool(gm[r].sum() == 0)
        out[f"pred_empty_{r}"] = bool(pm[r].sum() == 0)
    return out


def summarize_seg(rows: list[dict]) -> dict:
    s = {"n": len(rows)}
    for r in REGIONS:
        d = np.array([x[f"dice_{r}"] for x in rows], float)
        s[f"dice_{r}_mean"] = float(d.mean())
        s[f"dice_{r}_sd"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        s[f"dice_{r}_median"] = float(np.median(d))
        for k in ("hd", "hd95"):
            v = np.array([x[f"{k}_{r}"] for x in rows], float)
            s[f"{k}_{r}_mean"] = float(v.mean())
            s[f"{k}_{r}_median"] = float(np.median(v))
        s[f"n_gt_empty_{r}"] = int(sum(x[f"gt_empty_{r}"] for x in rows))
        s[f"n_pred_empty_{r}"] = int(sum(x[f"pred_empty_{r}"] for x in rows))
    s["hd95_mean_regions"] = float(np.mean([s["hd95_WT_mean"], s["hd95_TC_mean"], s["hd95_ET_mean"]]))
    s["hd_mean_regions"] = float(np.mean([s["hd_WT_mean"], s["hd_TC_mean"], s["hd_ET_mean"]]))
    return s
