#!/usr/bin/env python3
"""Integrated Gradients (IG) summary and inference timing for the Stage-1 and Stage-2 models.

IG: zero baseline, 32 Riemann steps, target = HGG logit, computed on the T2 channel for every test patient
(seed-42 hold-out models). Attribution enrichment = (share of total |IG| inside the GT whole tumour) /
(share of brain voxels inside the GT whole tumour); 1 = no spatial preference.

Timing (run with the GPU otherwise idle): 2D U-Net single model and 5-fold ensemble per volume (155 axial
slices), 3D classifier per volume, preprocessing (NIfTI read + scaling + resampling), nnU-Net measured
separately from its prediction logs. Inference only; excludes DICOM export, registration, skull stripping.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, OUT, TRUE_T2_SUFFIX, cohort, load_json, save_json, to_classifier_grid  # noqa: E402
from stage2 import S2_CKPT, TumorGradeClassifier, load_cls, stack_inputs  # noqa: E402

DEV = torch.device("cuda")


def integrated_gradients(model, x: torch.Tensor, steps: int = 32, target: int = 1) -> torch.Tensor:
    model.eval()
    base = torch.zeros_like(x)
    total = torch.zeros_like(x)
    for a in torch.linspace(1.0 / steps, 1.0, steps, device=x.device):
        xi = (base + a * (x - base)).requires_grad_(True)
        out = model(xi)[:, target].sum()
        g, = torch.autograd.grad(out, xi)
        total += g
    return (x - base) * total / steps


def run_ig(seed: int = 42) -> None:
    c = cohort()
    te = c["test"]
    res = {}
    for cond, msrc in (("t2", None), ("m2d", "m2d")):
        model = TumorGradeClassifier(in_channels=1 if cond == "t2" else 2).to(DEV)
        model.load_state_dict(torch.load(S2_CKPT / f"holdout_{cond}_seed{seed}.pth", map_location=DEV))
        X = stack_inputs(te, msrc)
        rows, mean_map = [], None
        for i, pid in enumerate(te):
            x = torch.from_numpy(X[i:i + 1].astype(np.float32)).to(DEV)
            ig_all = integrated_gradients(model, x)[0].abs().detach().cpu().numpy()
            ig = ig_all[0]
            mask_share = float(ig_all[1].sum() / (ig_all.sum() + 1e-12)) if ig_all.shape[0] > 1 else 0.0
            wt = load_cls("gt", pid) > 0.5
            brain = X[i, 0].astype(np.float32) > 0.02
            share_ig = float(ig[wt].sum() / (ig[brain | wt].sum() + 1e-12))
            share_vol = float(wt.sum() / ((brain | wt).sum() + 1e-12))
            rows.append({"patient_id": pid, "label": c["label_of"][pid], "ig_share_in_WT": share_ig,
                         "WT_volume_share": share_vol, "enrichment": share_ig / (share_vol + 1e-12),
                         "mask_channel_share": mask_share})
            if pid in (274, 252):
                np.save(OUT / f"ig_{cond}_{pid:03d}.npy", ig.astype(np.float16))
        e = np.array([r["enrichment"] for r in rows])
        s = np.array([r["ig_share_in_WT"] for r in rows])
        res[cond] = {"per_patient": rows, "enrichment_median": float(np.median(e)),
                     "enrichment_iqr": [float(np.percentile(e, 25)), float(np.percentile(e, 75))],
                     "ig_share_in_WT_median": float(np.median(s)),
                     "WT_volume_share_median": float(np.median([r["WT_volume_share"] for r in rows])),
                     "mask_channel_share_median": float(np.median([r["mask_channel_share"] for r in rows])),
                     "frac_patients_enrichment_gt1": float(np.mean(e > 1))}
        print(cond, "IG enrichment median", round(res[cond]["enrichment_median"], 2), "IQR", res[cond]["enrichment_iqr"])
    save_json(OUT / "ig_attribution_summary.json", res)


@torch.no_grad()
def run_timing(n: int = 20) -> None:
    from stage1_oof import load_fold_models, predict_labels
    c = cohort()
    pids = c["test"][:n]
    models = load_fold_models()
    cls = TumorGradeClassifier(in_channels=2).to(DEV)
    cls.load_state_dict(torch.load(S2_CKPT / "holdout_m2d_seed42.pth", map_location=DEV))
    cls.eval()
    t_pre, t_1, t_5, t_cls = [], [], [], []
    for i, pid in enumerate(pids):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        v = nib.load(DATA_DIR / f"BraTS20_Training_{pid:03d}{TRUE_T2_SUFFIX}").get_fdata(dtype=np.float32)
        v = (v - v.min()) / (v.max() - v.min() + 1e-8)
        t1 = time.perf_counter()
        _ = predict_labels(models[:1], v)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        lab = predict_labels(models, v)
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        x = np.stack([to_classifier_grid(v), np.clip(to_classifier_grid((lab > 0).astype(np.float32)), 0, 1)])
        xt = torch.from_numpy(x[None]).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            _ = torch.softmax(cls(xt).float(), 1)
        torch.cuda.synchronize()
        t4 = time.perf_counter()
        if i > 0:  # first volume = warm-up
            t_pre.append(t1 - t0); t_1.append(t2 - t1); t_5.append(t3 - t2); t_cls.append(t4 - t3)
    f = lambda a: {"mean_s": float(np.mean(a)), "sd_s": float(np.std(a, ddof=1)), "n": len(a)}  # noqa: E731
    res = {"gpu": torch.cuda.get_device_name(0), "read_and_scale": f(t_pre), "unet2d_single_model": f(t_1),
           "unet2d_5fold_ensemble": f(t_5), "resample_plus_3dcnn": f(t_cls),
           "note": "inference only; excludes DICOM conversion, registration, skull stripping and reporting"}
    save_json(OUT / "timing_classifier.json", res)
    print(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ig", action="store_true")
    ap.add_argument("--timing", action="store_true")
    a = ap.parse_args()
    if a.ig:
        run_ig()
    if a.timing:
        run_timing()
