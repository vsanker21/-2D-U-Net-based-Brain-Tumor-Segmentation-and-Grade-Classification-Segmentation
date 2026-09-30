#!/usr/bin/env python3
"""Sync official MICCAI BraTS2020 seg labels into workspace orientation (155,240,240), labels 0-3."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

BASE = Path(os.environ.get("BTS_BASE", Path(__file__).resolve().parents[2]))
OFFICIAL_ROOT = Path(os.environ.get("BRATS2020_OFFICIAL_DIR", BASE / "external" / "brats2020_official"))
OUT_DIR = BASE / "archive" / "3D Slices Sorted" / "masks_brats2020_official"
MANIFEST = OUT_DIR / "sync_manifest.json"


def official_to_nnunet_labels(seg: np.ndarray) -> np.ndarray:
    out = np.zeros(seg.shape, dtype=np.uint8)
    out[seg == 1] = 1
    out[seg == 2] = 2
    out[seg == 4] = 3
    return out


def align_orientation(seg: np.ndarray) -> np.ndarray:
    """MICCAI (240,240,155) -> workspace (155,240,240)."""
    if seg.shape == (155, 240, 240):
        return seg
    if seg.shape == (240, 240, 155):
        return np.transpose(seg, (2, 0, 1))
    raise ValueError(f"Unexpected seg shape {seg.shape}")


def patient_id_from_path(path: Path) -> int:
    m = re.search(r"BraTS20_Training_(\d+)_seg", path.name)
    if not m:
        raise ValueError(path)
    return int(m.group(1))


def sync_all() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg_files = sorted(OFFICIAL_ROOT.rglob("*_seg.nii"))
    rows = []
    for src in tqdm(seg_files, desc="Sync official seg"):
        pid = patient_id_from_path(src)
        raw = np.squeeze(nib.load(src).get_fdata()).astype(np.uint8)
        aligned = align_orientation(raw)
        labels = official_to_nnunet_labels(aligned)
        out_path = OUT_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz"
        nib.save(nib.Nifti1Image(labels, np.eye(4)), out_path)
        rows.append({
            "patient_id": pid,
            "source": str(src),
            "output": str(out_path),
            "shape": list(labels.shape),
            "unique_labels": np.unique(labels).tolist(),
            "wt_voxels": int((labels > 0).sum()),
        })

    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_synced": len(rows),
        "source_root": str(OFFICIAL_ROOT),
        "output_dir": str(OUT_DIR),
        "transform": "transpose(2,0,1); BraTS label 4->3 for nnU-Net",
        "patients": rows,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    manifest = sync_all()
    print(f"Synced {manifest['n_synced']} official seg volumes to {OUT_DIR}")


if __name__ == "__main__":
    main()
