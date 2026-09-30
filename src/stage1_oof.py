#!/usr/bin/env python3
"""Stage 1: out-of-fold 2D U-Net on true-T2 axial slices with official multi-class labels.

Design
  * development set = 295 master-train patients; 5 stratified folds (common.cohort()).
  * fold k: model trained on the other 4 folds (~236 patients), of which a stratified 10% is held back
    as an inner validation set for checkpoint selection / LR scheduling; predicts the 59 held-out patients
    (out-of-fold, OOF). Test patients (n = 74) are never used for training or selection.
  * test masks: softmax average of the 5 fold models (ensemble).
  * recipe: UNet2D (base 32, 4 classes), 0.5*Dice + 0.5*CE, AdamW
    (lr 1e-3, wd 1e-4), ReduceLROnPlateau (factor 0.5, patience 3) on validation loss, batch 16, 30 epochs,
    no augmentation. Input: axial 240x240 slices of the per-volume min-max scaled true-T2 volume.

Usage
  python stage1_oof.py --cache                 # build fp16/uint8 caches for all 369 patients
  python stage1_oof.py --folds 0 1 2 3 4       # train + OOF-predict
  python stage1_oof.py --ensemble              # test ensemble + metrics
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, MASK_CACHE, OUT, RANDOM_STATE, REGIONS, cohort, dice_bin, load_label,  # noqa: E402
                       load_t2, save_json, seg_metrics, set_seed, summarize_seg)
from unet2d import UNet2D  # noqa: E402

CKPT_DIR = CACHE / "stage1_ckpt"
OOF_DIR = MASK_CACHE / "stage1_oof"
ENS_DIR = MASK_CACHE / "stage1_test_ensemble"
for _p in (CKPT_DIR, OOF_DIR, ENS_DIR):
    _p.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CombinedLoss(nn.Module):
    """0.5 * (1 - mean soft Dice over the four classes) + 0.5 * cross-entropy."""

    def forward(self, pred, target):
        pred_soft = F.softmax(pred.float(), dim=1)
        target_oh = F.one_hot(target.clamp(0, 3), num_classes=4).permute(0, 3, 1, 2).float()
        inter = (pred_soft * target_oh).sum(dim=(2, 3))
        union = pred_soft.sum(dim=(2, 3)) + target_oh.sum(dim=(2, 3))
        dice = (2.0 * inter + 1e-8) / (union + 1e-8)
        return 0.5 * (1 - dice.mean()) + 0.5 * F.cross_entropy(pred.float(), target.clamp(0, 3))


def _cache_one(pid: int) -> int:
    load_t2(pid)
    load_label(pid)
    return pid


def build_cache() -> None:
    c = cohort()
    t0 = time.time()
    with Pool(6) as pool:
        for i, _ in enumerate(pool.imap_unordered(_cache_one, c["all"]), 1):
            if i % 50 == 0:
                print(f"cached {i}/{len(c['all'])} ({time.time() - t0:.0f}s)", flush=True)
    save_json(OUT / "cohort.json", c)
    print("cache done", time.time() - t0)


class SliceStore:
    """Axial brain-containing slices served from memory-mapped per-patient caches
    (the OS page cache is shared when several folds train concurrently)."""

    def __init__(self, pids: list[int]):
        self.x = [np.load(CACHE / "t2_fp16" / f"{p:03d}.npy", mmap_mode="r") for p in pids]
        self.y = [np.load(CACHE / "lab_u8" / f"{p:03d}.npy", mmap_mode="r") for p in pids]
        idx = []
        for i, v in enumerate(self.x):
            keep = np.where(np.asarray(v).reshape(v.shape[0], -1).max(1) > 0)[0]
            idx += [(i, int(z)) for z in keep]
        self.index = np.asarray(idx, dtype=np.int32)

    def __len__(self) -> int:
        return len(self.index)

    def batch(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xs = np.stack([self.x[i][z] for i, z in self.index[ids]]).astype(np.float32)
        ys = np.stack([self.y[i][z] for i, z in self.index[ids]]).astype(np.int64)
        return xs, ys


@torch.no_grad()
def predict_softmax(models: list[nn.Module], vol: np.ndarray, bs: int = 32) -> np.ndarray:
    """vol (155,240,240) in [0,1] -> mean softmax (4,155,240,240) float32 over models."""
    out = np.zeros((4,) + vol.shape, dtype=np.float32)
    x_all = torch.from_numpy(vol.astype(np.float32))
    for m in models:
        m.eval()
        for z0 in range(0, vol.shape[0], bs):
            x = x_all[z0:z0 + bs].unsqueeze(1).to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.softmax(m(x).float(), dim=1)
            out[:, z0:z0 + bs] += p.permute(1, 0, 2, 3).cpu().numpy()
    return out / len(models)


def predict_labels(models, vol) -> np.ndarray:
    return predict_softmax(models, vol).argmax(0).astype(np.uint8)


def eval_patients(models, pids) -> tuple[list[dict], float]:
    """Volumetric WT/TC/ET Dice (no HD, for speed) and slice-level loss on brain slices."""
    rows, losses = [], []
    crit = CombinedLoss()
    for pid in pids:
        vol = load_t2(pid)
        lab = load_label(pid)
        sm = predict_softmax(models, vol)
        pred = sm.argmax(0).astype(np.uint8)
        d = {}
        for r, cl in REGIONS.items():
            d[f"dice_{r}"] = dice_bin(np.isin(pred, cl), np.isin(lab, cl))
        rows.append(d)
        keep = (vol > 0).reshape(vol.shape[0], -1).any(1)
        logits = torch.log(torch.from_numpy(sm[:, keep]).clamp_min(1e-6)).permute(1, 0, 2, 3)
        losses.append(float(crit(logits, torch.from_numpy(lab[keep]).long())))
    return rows, float(np.mean(losses))


def train_fold(k: int, epochs: int, bs: int = 16, max_batches: int | None = None) -> None:
    c = cohort()
    f = c["folds"][k]
    y = [c["label_of"][p] for p in f["train"]]
    fit, val = train_test_split(f["train"], test_size=0.1, random_state=RANDOM_STATE + k, stratify=y)
    set_seed(RANDOM_STATE + k)
    t0 = time.time()
    S = SliceStore(fit)
    print(f"[fold {k}] fit {len(fit)} pts / {len(S)} slices; inner-val {len(val)} pts; "
          f"held-out {len(f['heldout'])} pts; load {time.time() - t0:.0f}s", flush=True)

    model = UNet2D(in_channels=1, num_classes=4, base_features=32).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3, factor=0.5)
    scaler = torch.amp.GradScaler("cuda")
    crit = CombinedLoss()
    best, best_ep, hist = -1.0, -1, []
    rng = np.random.default_rng(RANDOM_STATE + k)
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(len(S))
        nb = len(perm) // bs if max_batches is None else min(max_batches, len(perm) // bs)
        tl, te = 0.0, time.time()
        for b in range(nb):
            xb, yb = S.batch(perm[b * bs:(b + 1) * bs])
            x = torch.from_numpy(xb).unsqueeze(1).to(DEV, non_blocking=True)
            t = torch.from_numpy(yb).to(DEV, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            loss = crit(out, t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item()
        tr_time = time.time() - te
        rows, vloss = eval_patients([model], val)
        vd = {r: float(np.mean([x[f"dice_{r}"] for x in rows])) for r in ("WT", "TC", "ET")}
        score = float(np.mean(list(vd.values())))
        sched.step(vloss)
        hist.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_loss": vloss, **{f"val_dice_{r}": v for r, v in vd.items()},
                     "val_mean_dice": score, "lr": opt.param_groups[0]["lr"], "train_sec": tr_time})
        if score > best:
            best, best_ep = score, ep
            torch.save(model.state_dict(), CKPT_DIR / f"fold{k}_best.pth")
        print(f"[fold {k}] ep {ep:02d} loss {tl / max(nb, 1):.4f} vloss {vloss:.4f} "
              f"val WT/TC/ET {vd['WT']:.3f}/{vd['TC']:.3f}/{vd['ET']:.3f} lr {opt.param_groups[0]['lr']:.1e} "
              f"({tr_time:.0f}s train, {nb} it)", flush=True)
    del S

    model.load_state_dict(torch.load(CKPT_DIR / f"fold{k}_best.pth", map_location=DEV))
    oof_rows = []
    for pid in f["heldout"]:
        vol = load_t2(pid)
        pred = predict_labels([model], vol)
        np.save(OOF_DIR / f"{pid:03d}.npy", pred)
        m = seg_metrics(pred, load_label(pid))
        m.update(patient_id=pid, label=c["label_of"][pid], fold=k)
        oof_rows.append(m)
    s = summarize_seg(oof_rows)
    save_json(OUT / f"stage1_fold{k}.json", {"fold": k, "fit": fit, "inner_val": val, "heldout": f["heldout"],
                                             "best_epoch": best_ep, "best_inner_val_mean_dice": best,
                                             "history": hist, "oof_summary": s, "oof_per_patient": oof_rows})
    print(f"[fold {k}] best ep {best_ep} (inner-val mean Dice {best:.3f}); OOF WT/TC/ET "
          f"{s['dice_WT_mean']:.3f}/{s['dice_TC_mean']:.3f}/{s['dice_ET_mean']:.3f} total {time.time() - t0:.0f}s",
          flush=True)


def load_fold_models() -> list[nn.Module]:
    ms = []
    for k in range(5):
        m = UNet2D(in_channels=1, num_classes=4, base_features=32).to(DEV)
        m.load_state_dict(torch.load(CKPT_DIR / f"fold{k}_best.pth", map_location=DEV))
        m.eval()
        ms.append(m)
    return ms


def ensemble() -> None:
    c = cohort()
    models = load_fold_models()
    rows_ens, rows_single = [], {k: [] for k in range(5)}
    for i, pid in enumerate(c["test"], 1):
        vol, lab = load_t2(pid), load_label(pid)
        pred = predict_labels(models, vol)
        np.save(ENS_DIR / f"{pid:03d}.npy", pred)
        m = seg_metrics(pred, lab)
        m.update(patient_id=pid, label=c["label_of"][pid])
        rows_ens.append(m)
        for k in range(5):
            pk = predict_labels([models[k]], vol)
            mk = seg_metrics(pk, lab)
            mk.update(patient_id=pid)
            rows_single[k].append(mk)
        if i % 10 == 0:
            print(f"test {i}/{len(c['test'])}", flush=True)
    single = {k: summarize_seg(v) for k, v in rows_single.items()}
    oof_rows = []
    for k in range(5):
        oof_rows += load_json_fold(k)["oof_per_patient"]
    save_json(OUT / "seg_stage1_2d_trueT2.json", {
        "test_ensemble": {"summary": summarize_seg(rows_ens), "per_patient": rows_ens},
        "test_single_fold_models": single,
        "oof_development": {"summary": summarize_seg(oof_rows), "per_patient": oof_rows},
    })
    s = summarize_seg(rows_ens)
    print("TEST ensemble WT/TC/ET", round(s["dice_WT_mean"], 3), round(s["dice_TC_mean"], 3), round(s["dice_ET_mean"], 3),
          "HD95 WT", round(s["hd95_WT_mean"], 1))


def insample() -> None:
    """In-sample masks for the 295 development patients: a patient in fold j is predicted by the first other
    fold model (j+1, j+2, ... mod 5) whose training (fit) set contained that patient. Used only to quantify the Stage-1 -> Stage-2
    train/test mismatch that out-of-fold masks avoid."""
    c = cohort()
    out_dir = MASK_CACHE / "stage1_insample"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = load_fold_models()
    fits = {k: set(load_json_fold(k)["fit"]) for k in range(5)}
    rows = []
    for f in c["folds"]:
        for pid in f["heldout"]:
            k = next(kk % 5 for kk in range(f["fold"] + 1, f["fold"] + 5) if pid in fits[kk % 5])
            fit_k = fits[k]
            vol, lab = load_t2(pid), load_label(pid)
            pred = predict_labels([models[k]], vol)
            np.save(out_dir / f"{pid:03d}.npy", pred)
            oof = np.load(OOF_DIR / f"{pid:03d}.npy")
            r = {"patient_id": pid, "fold": f["fold"], "predictor_fold": k, "in_fit_set": pid in fit_k}
            for reg, cl in REGIONS.items():
                r[f"insample_dice_{reg}"] = dice_bin(np.isin(pred, cl), np.isin(lab, cl))
                r[f"oof_dice_{reg}"] = dice_bin(np.isin(oof, cl), np.isin(lab, cl))
            rows.append(r)
    summ = {}
    for reg in REGIONS:
        a = np.array([r[f"insample_dice_{reg}"] for r in rows])
        b = np.array([r[f"oof_dice_{reg}"] for r in rows])
        from scipy.stats import wilcoxon
        summ[reg] = {"insample_mean": float(a.mean()), "oof_mean": float(b.mean()),
                     "mean_diff": float((a - b).mean()), "wilcoxon_p": float(wilcoxon(a, b).pvalue)}
    save_json(OUT / "stage1_insample_vs_oof.json", {"summary": summ, "per_patient": rows,
                                                    "n_in_fit_set": int(sum(r["in_fit_set"] for r in rows))})
    print("in-sample vs OOF", {k: (round(v["insample_mean"], 3), round(v["oof_mean"], 3)) for k, v in summ.items()})


def load_json_fold(k: int) -> dict:
    from common import load_json
    return load_json(OUT / f"stage1_fold{k}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--folds", type=int, nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bench", action="store_true", help="1 epoch x 200 batches on fold 0 (throughput)")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--insample", action="store_true")
    a = ap.parse_args()
    if a.cache:
        build_cache()
    if a.bench:
        train_fold(0, epochs=1, max_batches=200)
    for k in a.folds:
        train_fold(k, a.epochs)
    if a.ensemble:
        ensemble()
    if a.insample:
        insample()
