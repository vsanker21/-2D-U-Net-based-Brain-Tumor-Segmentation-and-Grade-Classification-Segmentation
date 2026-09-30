#!/usr/bin/env python3
"""Voxel-wise agreement between H5-rebuilt multi-class labels and official MICCAI segmentations (all 368 pairs)."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OFFICIAL_MASK_DIR, OUT, SEG_MASK_DIR, save_json  # noqa: E402

rows = []
for f in sorted(OFFICIAL_MASK_DIR.glob("*_seg.nii.gz")):
    o = np.asarray(nib.load(f).dataobj).astype(np.uint8)
    s = np.asarray(nib.load(SEG_MASK_DIR / f.name).dataobj).astype(np.uint8)
    wt_o, wt_s = o > 0, s > 0
    d = 2 * np.logical_and(wt_o, wt_s).sum() / max(wt_o.sum() + wt_s.sum(), 1)
    rows.append({"file": f.name, "voxel_agreement": float((o == s).mean()), "wt_dice": float(d)})
va = np.array([r["voxel_agreement"] for r in rows])
wd = np.array([r["wt_dice"] for r in rows])
res = {"n": len(rows), "voxel_agreement_min": float(va.min()), "voxel_agreement_mean": float(va.mean()),
       "wt_dice_min": float(wd.min()), "wt_dice_mean": float(wd.mean()), "n_identical": int((va == 1).sum())}
save_json(OUT / "label_rebuild_vs_official.json", {"summary": res, "per_patient": rows})
print(res)
