#!/usr/bin/env python3
"""
Rebuild BraTS2020 multi-class segmentation labels from the per-slice HDF5 files.

The HDF5 mask has shape (240, 240, 3) with mutually exclusive binary channels:
  channel 0 -> NCR (label 1)
  channel 1 -> peritumoral edema (label 2)
  channel 2 -> enhancing tumor (label 3; official BraTS uses 4)

All three channels are kept, so the maps define WT, TC and ET. They are used only where no official
segmentation file is available.
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm

BASE = Path(os.environ.get("BTS_BASE", Path(__file__).resolve().parents[2]))
H5_DIR = BASE / "archive" / "BraTS2020_training_data" / "content" / "data"
OUT_DIR = BASE / "archive" / "3D Slices Sorted" / "masks_brats2020_seg"
MANIFEST = OUT_DIR / "rebuild_manifest.json"


def slice_to_label_map(mask_hwc: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) binary channels to nnU-Net/BraTS label map {0,1,2,3}."""
    lab = np.zeros(mask_hwc.shape[:2], dtype=np.uint8)
    if mask_hwc.ndim == 2:
        return (mask_hwc > 0).astype(np.uint8)
    if mask_hwc.shape[-1] >= 3:
        lab[mask_hwc[:, :, 0] > 0] = 1
        lab[mask_hwc[:, :, 1] > 0] = 2
        lab[mask_hwc[:, :, 2] > 0] = 3
        return lab
    return (mask_hwc > 0).astype(np.uint8)


def group_h5_slices() -> dict[str, list[tuple[int, Path]]]:
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for h5_file in H5_DIR.glob("volume_*_slice_*.h5"):
        m = re.match(r"volume_(\d+)_slice_(\d+)\.h5", h5_file.name)
        if not m:
            continue
        groups[m.group(1)].append((int(m.group(2)), h5_file))
    for vid in groups:
        groups[vid].sort(key=lambda x: x[0])
    return groups


def rebuild_volume(volume_id: str, slices: list[tuple[int, Path]]) -> dict:
    label_slices: list[np.ndarray] = []
    for _, h5_path in slices:
        with h5py.File(h5_path, "r") as f:
            if "mask" not in f:
                raise KeyError(f"mask missing in {h5_path}")
            mask = np.array(f["mask"])
        label_slices.append(slice_to_label_map(mask))

    volume = np.stack(label_slices, axis=0).astype(np.uint8)
    out_name = f"BraTS20_Training_{volume_id.zfill(3)}_seg.nii.gz"
    out_path = OUT_DIR / out_name
    nib.save(nib.Nifti1Image(volume, np.eye(4)), out_path)

    uniq = np.unique(volume).tolist()
    wt = int((volume > 0).sum())
    return {
        "volume_id": volume_id,
        "path": str(out_path),
        "shape": list(volume.shape),
        "unique_labels": uniq,
        "wt_voxels": wt,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = group_h5_slices()
    rows = []
    for volume_id, slices in tqdm(sorted(groups.items(), key=lambda x: int(x[0])), desc="Rebuilding seg"):
        try:
            rows.append(rebuild_volume(volume_id, slices))
        except Exception as exc:
            rows.append({"volume_id": volume_id, "error": str(exc)})

    ok = [r for r in rows if "error" not in r]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": str(H5_DIR),
        "output_dir": str(OUT_DIR),
        "n_volumes_ok": len(ok),
        "n_volumes_failed": len(rows) - len(ok),
        "label_schema": {"0": "background", "1": "NCR", "2": "ED", "3": "ET"},
        "note": "Rebuilt from the 3-channel HDF5 masks (NCR, ED, ET).",
        "volumes": rows,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved {len(ok)} multi-class seg volumes to {OUT_DIR}")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
