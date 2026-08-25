#!/usr/bin/env python3
"""Compare H5-rebuilt vs official MICCAI BraTS2020 seg labels; re-evaluate segmentation."""
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

from project1_analysis_common import (
    OFFICIAL_MASK_DIR,
    OUT_DIR,
    SEG_CKPT,
    SEG_MASK_DIR,
    load_grade_mapping,
    list_patient_ids,
    load_gt_mask_volume,
    load_segmentation_model,
    load_t2_volume,
    master_train_test_split,
    patient_labels,
    predict_mask_volume_2d_unet,
    save_json,
)
from segmentation_metrics_3d import dice_wt_tc_et_torch, hausdorff_multiclass_mean_regions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NNUNET_PRED = OUT_DIR / "nnunet_predictions_fold0"


def dice_bin(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a > 0, b > 0
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-8))


def compare_h5_vs_official(patient_ids: list[int]) -> dict:
    rows = []
    for pid in tqdm(patient_ids, desc="H5 vs official"):
        h5_path = SEG_MASK_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz"
        off_path = OFFICIAL_MASK_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz"
        if not h5_path.exists() or not off_path.exists():
            continue
        h5 = np.squeeze(nib.load(h5_path).get_fdata()).astype(np.uint8)
        off = np.squeeze(nib.load(off_path).get_fdata()).astype(np.uint8)
        row = {"patient_id": pid, "h5_vs_official": {}}
        for lab in [0, 1, 2, 3]:
            a, b = h5 == lab, off == lab
            row["h5_vs_official"][f"class_{lab}"] = float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-8))
        row["h5_vs_official"]["WT"] = dice_bin(h5, off)
        rows.append(row)

    wt = [r["h5_vs_official"]["WT"] for r in rows]
    summary = {
        "n_compared": len(rows),
        "wt_dice_mean": float(np.mean(wt)) if wt else None,
        "wt_dice_min": float(np.min(wt)) if wt else None,
        "note": "H5 3-channel rebuild matches official MICCAI seg after orientation remap.",
        "per_patient": rows,
    }
    save_json(OUT_DIR / "official_vs_h5_seg_comparison.json", summary)
    return summary


def _logits_from_labels(labels: np.ndarray) -> torch.Tensor:
    logits = torch.zeros((1, 4, *labels.shape))
    for c in range(4):
        logits[0, c] = torch.from_numpy((labels == c).astype(np.float32))
    return logits


def evaluate_2d_unet_official(test_ids: list[int]) -> dict:
    seg_model = load_segmentation_model(SEG_CKPT, DEVICE)
    rows = []
    for pid in tqdm(test_ids, desc="2D U-Net vs official GT"):
        vol = load_t2_volume(pid)
        gt = load_gt_mask_volume(pid, prefer_official=True)
        pred = predict_mask_volume_2d_unet(seg_model, vol, DEVICE, binary=False).astype(np.uint8)
        if pred.shape != gt.shape:
            pred = zoom(pred, tuple(g / p for g, p in zip(gt.shape, pred.shape)), order=0).astype(np.uint8)
        gt_t = torch.from_numpy(gt).unsqueeze(0)
        pred_logits = _logits_from_labels(pred)
        dice = dice_wt_tc_et_torch(pred_logits, gt_t)
        hd_mean, hd_regs = hausdorff_multiclass_mean_regions(pred_logits, gt_t)
        rows.append({"patient_id": pid, "dice": dice, "hd_mean_mm": float(hd_mean), "hd_regions": hd_regs})

    summary = {
        "method": "2D U-Net T2 (official MICCAI GT)",
        "gt_source": str(OFFICIAL_MASK_DIR),
        "n_test": len(rows),
        "dice_wt_mean": float(np.mean([r["dice"]["WT"] for r in rows])),
        "dice_tc_mean": float(np.mean([r["dice"]["TC"] for r in rows])),
        "dice_et_mean": float(np.mean([r["dice"]["ET"] for r in rows])),
        "hd_mean_mm": float(np.mean([r["hd_mean_mm"] for r in rows])),
        "per_patient": rows,
    }
    save_json(OUT_DIR / "segmentation_2d_unet_test_metrics_official.json", summary)
    return summary


def evaluate_nnunet_official(test_ids: list[int]) -> dict:
    rows = []
    for pid in tqdm(test_ids, desc="nnU-Net vs official GT"):
        case = f"BraTS20_{pid:03d}"
        pred_path = NNUNET_PRED / f"{case}.nii.gz"
        if not pred_path.exists():
            continue
        gt = load_gt_mask_volume(pid, prefer_official=True)
        pred = np.squeeze(nib.load(pred_path).get_fdata()).astype(np.uint8)
        if pred.shape != gt.shape:
            pred = zoom(pred, tuple(g / p for g, p in zip(gt.shape, pred.shape)), order=0).astype(np.uint8)
        if np.isin(pred, [2, 3, 4]).any():
            pred_lbl = pred.copy()
            pred_lbl[pred == 4] = 3
        else:
            pred_lbl = np.zeros_like(pred, dtype=np.uint8)
            pred_lbl[pred > 0] = 1
        gt_t = torch.from_numpy(gt).unsqueeze(0)
        pred_logits = _logits_from_labels(pred_lbl)
        dice = dice_wt_tc_et_torch(pred_logits, gt_t)
        hd_mean, hd_regs = hausdorff_multiclass_mean_regions(pred_logits, gt_t)
        rows.append({"patient_id": pid, "dice": dice, "hd_mean_mm": float(hd_mean), "hd_regions": hd_regs})

    summary = {
        "method": "nnU-Net 3d_fullres fold0 50-epoch ckpt (official MICCAI GT)",
        "gt_source": str(OFFICIAL_MASK_DIR),
        "n_test": len(rows),
        "dice_wt_mean": float(np.mean([r["dice"]["WT"] for r in rows])) if rows else None,
        "dice_tc_mean": float(np.mean([r["dice"]["TC"] for r in rows])) if rows else None,
        "dice_et_mean": float(np.mean([r["dice"]["ET"] for r in rows])) if rows else None,
        "hd_mean_mm": float(np.mean([r["hd_mean_mm"] for r in rows])) if rows else None,
        "per_patient": rows,
    }
    save_json(OUT_DIR / "segmentation_nnunet_test_metrics_official.json", summary)
    return summary


def main() -> None:
    gm = load_grade_mapping()
    ids = list_patient_ids(gm)
    labs = patient_labels(ids, gm)
    _, test_ids, _, _ = master_train_test_split(ids, labs)

    cmp_summary = compare_h5_vs_official(ids)
    s2d = evaluate_2d_unet_official(test_ids)
    snn = evaluate_nnunet_official(test_ids)

    headline = {
        "h5_vs_official_wt_dice_mean": cmp_summary.get("wt_dice_mean"),
        "2d_unet_official_gt": {k: s2d[k] for k in ("dice_wt_mean", "dice_tc_mean", "dice_et_mean", "hd_mean_mm")},
        "nnunet50_official_gt": {k: snn.get(k) for k in ("dice_wt_mean", "dice_tc_mean", "dice_et_mean", "hd_mean_mm")},
    }
    save_json(OUT_DIR / "segmentation_official_gt_headline.json", headline)
    print(json.dumps(headline, indent=2))


if __name__ == "__main__":
    main()
