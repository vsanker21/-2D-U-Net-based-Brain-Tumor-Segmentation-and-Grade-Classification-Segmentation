#!/usr/bin/env python3
"""Identify the true MRI sequence stored in each converted NIfTI channel file.

Uses sequence-specific tissue contrast relative to official BraTS labels:
enhancing tumor is bright only on T1ce; edema is bright on T2 and FLAIR;
CSF is bright on T2 but suppressed on FLAIR and dark on T1/T1ce.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, OFFICIAL_MASK_DIR, OUT  # noqa: E402

MODS = ["T1", "T1ce", "T2", "FLAIR"]
PIDS = [1, 5, 20, 50, 100, 150, 200, 252, 274, 300, 330, 350]
OUT_JSON = OUT / "modality_channel_check.json"


def main() -> None:
    acc = {m: {"et": [], "ncr": [], "ed": [], "csf": []} for m in MODS}
    for pid in PIDS:
        seg = np.squeeze(nib.load(OFFICIAL_MASK_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz").get_fdata()).astype(int)
        wt = seg > 0
        ring = binary_dilation(wt, iterations=6) & ~wt
        vols = {
            m: nib.load(DATA_DIR / f"BraTS20_Training_{pid:03d}_{m}.nii.gz").get_fdata().astype(np.float32)
            for m in MODS
        }
        brain = np.zeros_like(wt)
        for v in vols.values():
            brain |= v > 0
        normal = binary_erosion(brain, iterations=3) & ~binary_dilation(wt, iterations=3)
        for m, v in vols.items():
            r = v[ring].mean() + 1e-6
            if (seg == 3).sum() > 50:
                acc[m]["et"].append(float(v[seg == 3].mean() / r))
            if (seg == 1).sum() > 50:
                acc[m]["ncr"].append(float(v[seg == 1].mean() / r))
            acc[m]["ed"].append(float(v[seg == 2].mean() / r))
            b = v[normal]
            acc[m]["csf"].append(float(np.percentile(b, 99.5) / (np.median(b) + 1e-6)))

    summary = {m: {k: float(np.mean(vals)) for k, vals in acc[m].items()} for m in MODS}
    print("file label | ET/ring | NCR/ring | edema/ring | p99.5/median normal brain")
    for m in MODS:
        s = summary[m]
        print(f"{m:10s} | {s['et']:.2f}    | {s['ncr']:.2f}     | {s['ed']:.2f}       | {s['csf']:.2f}")
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"patients": PIDS, "summary": summary}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
