#!/usr/bin/env python3
"""nnU-Net v2 benchmark on the TRUE T2 channel (Dataset502_BraTS20trueT2).

Steps (select with --step):
  build     hard-link true-T2 images (+ official labels) for the 295 development / 74 test patients
  plan      nnUNetv2_plan_and_preprocess -d 502 -c 3d_fullres (nnU-Net default 5-fold split; fold 0 = 236/59)
  train     fold 0, nnUNetTrainer_250epochs_snap50 (snapshot at epoch 50), final checkpoint used
  predict   test-set predictions from checkpoint_ep50.pth and checkpoint_final.pth
  eval      WT/TC/ET Dice, HD, HD95 against official labels on the 74 test patients
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (DATA_DIR, OFFICIAL_MASK_DIR, OUT, WORK_DIR, SEG_MASK_DIR, TRUE_T2_SUFFIX, cohort,  # noqa: E402
                       load_label, save_json, seg_metrics, summarize_seg)

NN = WORK_DIR / "nnunet"
RAW, PRE, RES = NN / "raw", NN / "preprocessed", NN / "results"
DS = "Dataset502_BraTS20trueT2"
TRAINER = "nnUNetTrainer_250epochs_snap50"
SCRIPTS = Path(sys.executable).parent / "Scripts"


def env() -> dict:
    e = os.environ.copy()
    e["nnUNet_raw"], e["nnUNet_preprocessed"], e["nnUNet_results"] = str(RAW), str(PRE), str(RES)
    e.setdefault("nnUNet_n_proc_DA", "6")
    e["PYTHONIOENCODING"] = "utf-8"
    for p in (RAW, PRE, RES):
        p.mkdir(parents=True, exist_ok=True)
    return e


def link(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def step_build() -> None:
    c = cohort()
    d = RAW / DS
    for sub in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    for split, img_sub, lab_sub in ((c["train"], "imagesTr", "labelsTr"), (c["test"], "imagesTs", "labelsTs")):
        for pid in split:
            img = DATA_DIR / f"BraTS20_Training_{pid:03d}{TRUE_T2_SUFFIX}"
            lab = OFFICIAL_MASK_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz"
            if not lab.exists():
                lab = SEG_MASK_DIR / f"BraTS20_Training_{pid:03d}_seg.nii.gz"
            ii, ll = nib.load(img), nib.load(lab)
            assert ii.shape == ll.shape and np.allclose(ii.affine, ll.affine), pid
            link(img, d / img_sub / f"BraTS20_{pid:03d}_0000.nii.gz")
            link(lab, d / lab_sub / f"BraTS20_{pid:03d}.nii.gz")
    save_json(d / "dataset.json", {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "NCR": 1, "ED": 2, "ET": 3},
        "numTraining": len(c["train"]),
        "file_ending": ".nii.gz",
        "name": "BraTS20trueT2",
        "description": "BraTS2020 T2-weighted channel only",
    })
    print("built", d, len(c["train"]), "train", len(c["test"]), "test")


def step_plan() -> None:
    e = env()
    subprocess.run([str(SCRIPTS / "nnUNetv2_plan_and_preprocess.exe"), "-d", "502", "-c", "3d_fullres",
                    "--verify_dataset_integrity", "-np", "4"], env=e, check=True)


def step_train() -> None:
    e = env()
    log = OUT / "nnunet_train_fold0.log"
    with open(log, "a", encoding="utf-8") as f:
        subprocess.run([str(SCRIPTS / "nnUNetv2_train.exe"), "502", "3d_fullres", "0", "-tr", TRAINER, "--c"]
                       if (RES / DS / f"{TRAINER}__nnUNetPlans__3d_fullres" / "fold_0" / "checkpoint_latest.pth").exists()
                       else [str(SCRIPTS / "nnUNetv2_train.exe"), "502", "3d_fullres", "0", "-tr", TRAINER],
                       env=e, stdout=f, stderr=subprocess.STDOUT, check=False)


def step_predict() -> None:
    e = env()
    for chk, tag in (("checkpoint_ep50.pth", "ep50"), ("checkpoint_final.pth", "ep250")):
        out = NN / f"pred_test_{tag}"
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(SCRIPTS / "nnUNetv2_predict.exe"), "-i", str(RAW / DS / "imagesTs"), "-o", str(out),
                        "-d", "502", "-c", "3d_fullres", "-f", "0", "-tr", TRAINER, "-chk", chk,
                        "-npp", "2", "-nps", "2"], env=e, check=True)


def step_eval() -> None:
    c = cohort()
    res = {}
    for tag in ("ep50", "ep250"):
        rows = []
        for pid in c["test"]:
            p = NN / f"pred_test_{tag}" / f"BraTS20_{pid:03d}.nii.gz"
            pred = np.asarray(nib.load(p).dataobj).astype(np.uint8)
            m = seg_metrics(pred, load_label(pid))
            m["patient_id"] = pid
            m["label"] = c["label_of"][pid]
            rows.append(m)
        res[tag] = {"summary": summarize_seg(rows), "per_patient": rows}
        print(tag, {k: round(v, 3) for k, v in res[tag]["summary"].items() if k.startswith("dice") and k.endswith("mean")})
    save_json(OUT / "seg_nnunet_trueT2_test.json", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["build", "plan", "train", "predict", "eval", "all_train"])
    a = ap.parse_args()
    if a.step == "all_train":
        step_build(); step_plan(); step_train()
    else:
        {"build": step_build, "plan": step_plan, "train": step_train, "predict": step_predict, "eval": step_eval}[a.step]()
