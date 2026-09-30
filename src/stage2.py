#!/usr/bin/env python3
"""Stage 2: 3D HGG/LGG classification on the true-T2 channel.

Conditions (second input channel = binary whole-tumour mask resampled to 128x128x96):
  t2    T2 only (1 channel)
  m2d   T2 + Stage-1 2D U-Net mask; TRAINING masks are out-of-fold (OOF) predictions,
        TEST masks come from the 5-fold ensemble (no model that produced a mask saw that patient)
  gt    T2 + ground-truth WT mask during training (localization-quality upper reference);
        evaluated with GT (oracle), nnU-Net (250 / 50 epochs) and 2D masks at test
  rand  T2 + patient-fixed uniform noise (negative control)
m2d is additionally evaluated with nnU-Net and GT masks substituted at test time.

Training recipe: TumorGradeClassifier, weighted focal loss (gamma 2, inverse-frequency class weights),
shuffled sampling (a class-balanced sampler is available via --balanced-sampler for a
class-imbalance sensitivity run only), AdamW (lr 1e-4, wd 1e-4),
CosineAnnealingWarmRestarts (T_0 = 10, T_mult = 2) stepped per epoch, batch 4, 30 epochs, random flips of the
two in-plane axes and multiplicative intensity scaling U(0.9, 1.1) on the T2 channel.
Model selection: NONE. The model after the last (30th) epoch is evaluated; with T_0 = 10, T_mult = 2 the
30th epoch closes the second cosine cycle at the minimum learning rate. No test or validation data are
used during training.

Usage
  python stage2.py --build-cache t2 gt            (independent of Stage 1)
  python stage2.py --build-cache m2d nn50 nn250   (after Stage 1 / nnU-Net)
  python stage2.py --holdout t2 gt rand --seeds 42 43 44 45 46
  python stage2.py --holdout m2d --seeds 42 43 44 45 46
  python stage2.py --cv --seeds 42 43 44
  python stage2.py --holdout t2 --seeds 42 43 44 45 --balanced-sampler   (sensitivity run; separate output folder)
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CACHE, CLASSIFY_SHAPE, CLS_CACHE, MASK_CACHE, OUT, WORK_DIR, RANDOM_STATE, cohort,  # noqa: E402
                       load_label, load_t2, save_json, set_seed, to_classifier_grid)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S2_OUT = OUT / "stage2"
S2_CKPT = CACHE / "stage2_ckpt"
for _p in (S2_OUT, S2_CKPT):
    _p.mkdir(parents=True, exist_ok=True)

TEST_MASKS = {"t2": [None], "m2d": ["m2d", "nn250", "nn50", "gt"], "gt": ["gt", "nn250", "nn50", "m2d"],
              "rand": ["rand"], "m2d_ins": ["m2d"]}
TRAIN_MASK = {"t2": None, "m2d": "m2d", "gt": "gt", "rand": "rand", "m2d_ins": "m2d_ins"}


# ----------------------------------------------------------------------------- model (verbatim architecture)
class TumorGradeClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout_rate: float = 0.5, in_channels: int = 1):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(64, 128, 3, padding=1), nn.BatchNorm3d(128), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(128, 256, 3, padding=1), nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 4 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(dropout_rate),
            nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_rate), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.feature_extractor(x))


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float32))
        self.alpha, self.gamma = alpha, gamma

    def forward(self, pred, target):
        ce = F.cross_entropy(pred, target, reduction="none")
        pt = torch.exp(-ce)
        return (self.class_weights[target] * self.alpha * (1 - pt) ** self.gamma * ce).mean()


# ----------------------------------------------------------------------------- caches
def _mask_full(source: str, pid: int, c: dict) -> np.ndarray | None:
    """Full-resolution binary WT mask (155,240,240) for a source, or None if unavailable."""
    if source == "gt":
        return (load_label(pid) > 0).astype(np.float32)
    if source in ("m2d", "m2d_ins"):
        if pid in set(c["train"]):
            sub = "stage1_oof" if source == "m2d" else "stage1_insample"
        else:
            sub = "stage1_test_ensemble"
        p = MASK_CACHE / sub / f"{pid:03d}.npy"
        return (np.load(p) > 0).astype(np.float32) if p.exists() else None
    if source in ("nn50", "nn250"):
        if pid not in set(c["test"]):
            return None
        tag = "ep50" if source == "nn50" else "ep250"
        p = WORK_DIR / "nnunet" / f"pred_test_{tag}" / f"BraTS20_{pid:03d}.nii.gz"
        return (np.asarray(nib.load(p).dataobj) > 0).astype(np.float32) if p.exists() else None
    raise ValueError(source)


def build_cache(sources: list[str]) -> None:
    c = cohort()
    for src in sources:
        d = CLS_CACHE / src
        d.mkdir(parents=True, exist_ok=True)
        n = 0
        t0 = time.time()
        for pid in c["all"]:
            fp = d / f"{pid:03d}.npy"
            if fp.exists():
                continue
            if src == "t2":
                arr = to_classifier_grid(load_t2(pid))
            else:
                m = _mask_full(src, pid, c)
                if m is None:
                    continue
                arr = np.clip(to_classifier_grid(m), 0.0, 1.0)
            np.save(fp, arr.astype(np.float16))
            n += 1
        print(f"cache {src}: +{n} ({time.time() - t0:.0f}s)", flush=True)


def random_mask(pid: int) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_STATE + pid * 10007)
    return rng.random(CLASSIFY_SHAPE, dtype=np.float32)


def load_cls(src: str, pid: int) -> np.ndarray:
    if src == "rand":
        return random_mask(pid)
    return np.load(CLS_CACHE / src / f"{pid:03d}.npy").astype(np.float32)


def stack_inputs(pids: list[int], mask_src: str | None) -> np.ndarray:
    ch = 1 if mask_src is None else 2
    X = np.zeros((len(pids), ch) + CLASSIFY_SHAPE, dtype=np.float16)
    for i, p in enumerate(pids):
        X[i, 0] = load_cls("t2", p)
        if mask_src is not None:
            X[i, 1] = load_cls(mask_src, p)
    return X


# ----------------------------------------------------------------------------- training / inference
def class_weights(y: np.ndarray) -> list[float]:
    counts = np.bincount(y, minlength=2)
    return [float(len(y) / (2 * counts[i])) for i in range(2)]


def augment(x: np.ndarray, rs: random.Random) -> np.ndarray:
    """x (C,H,W,D) float32; same geometric op on all channels, intensity scaling on T2 only."""
    if rs.random() < 0.5:
        if rs.random() < 0.5:
            x = x[:, ::-1]
        if rs.random() < 0.5:
            x = x[:, :, ::-1]
        x = np.ascontiguousarray(x)
        x[0] = np.clip(x[0] * rs.uniform(0.9, 1.1), 0.0, 1.0)
    return x


def train_model(X: np.ndarray, y: np.ndarray, seed: int, epochs: int = 30, bs: int = 4,
                balanced_sampler: bool = False) -> nn.Module:
    """Class imbalance is corrected once, by inverse-frequency weights in the focal loss; every patient is seen
    once per epoch. balanced_sampler=True adds a weighted sampler to the weighted loss,
    which corrects the imbalance twice (effective LGG:HGG weight ~3.8:1) and shifts P(HGG) downwards."""
    set_seed(seed)
    rs = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    model = TumorGradeClassifier(in_channels=X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)
    crit = WeightedFocalLoss(class_weights(y)).to(DEV)
    counts = np.bincount(y, minlength=2)
    w = torch.tensor([len(y) / (2 * counts[t]) for t in y], dtype=torch.double)
    n_batches = int(np.ceil(len(y) / bs))
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(epochs):
        model.train()
        if balanced_sampler:
            order = torch.multinomial(w, len(y), replacement=True, generator=g).numpy()
        else:
            order = torch.randperm(len(y), generator=g).numpy()
        tl = 0.0
        for b in range(n_batches):
            idx = order[b * bs:(b + 1) * bs]
            xb = np.stack([augment(X[i].astype(np.float32), rs) for i in idx])
            xt = torch.from_numpy(xb).to(DEV)
            yt = torch.from_numpy(y[idx].astype(np.int64)).to(DEV)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xt)
            loss = crit(logits.float(), yt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += float(loss.item())
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model: nn.Module, X: np.ndarray, bs: int = 4) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xt = torch.from_numpy(X[i:i + bs].astype(np.float32)).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(xt)
        out.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out).astype(float)


# ----------------------------------------------------------------------------- radiomics baseline
def radiomic_features(pid: int, mask_src: str) -> list[float]:
    """Five first-order features of the full-resolution T2 within the binary mask."""
    c = cohort()
    vol = load_t2(pid)
    m = _mask_full(mask_src, pid, c)
    t = m > 0
    if not t.any():
        return [0.0] * 5
    v = vol[t]
    return [float(t.sum()), float(v.mean()), float(v.std()), float(v.max()), float(np.percentile(v, 90))]


def radiomics_fit_predict(train_pids, y_train, test_pids, train_src="m2d", test_src="m2d") -> np.ndarray:
    Xtr = np.array([radiomic_features(p, train_src) for p in train_pids])
    Xte = np.array([radiomic_features(p, test_src) for p in test_pids])
    pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE))])
    pipe.fit(Xtr, y_train)
    return pipe.predict_proba(Xte)[:, 1]


# ----------------------------------------------------------------------------- experiments
def run_holdout(conds: list[str], seeds: list[int], epochs: int, balanced_sampler: bool = False) -> None:
    out_dir = OUT / "stage2_doublebalanced_recipe" if balanced_sampler else S2_OUT
    ckpt_dir = S2_CKPT / "doublebalanced" if balanced_sampler else S2_CKPT
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    c = cohort()
    tr, te = c["train"], c["test"]
    ytr = np.array([c["label_of"][p] for p in tr])
    yte = np.array([c["label_of"][p] for p in te])
    for cond in conds:
        Xtr = stack_inputs(tr, TRAIN_MASK[cond])
        Xte = {m: stack_inputs(te, m) for m in TEST_MASKS[cond]
               if m is None or m == "rand" or all((CLS_CACHE / m / f"{p:03d}.npy").exists() for p in te)}
        for seed in seeds:
            fp = out_dir / f"holdout_{cond}_seed{seed}.json"
            if fp.exists():
                print("skip", fp.name)
                continue
            t0 = time.time()
            model = train_model(Xtr, ytr, seed, epochs, balanced_sampler=balanced_sampler)
            torch.save(model.state_dict(), ckpt_dir / f"holdout_{cond}_seed{seed}.pth")
            res = {"condition": cond, "seed": seed, "epochs": epochs, "selection": "final epoch (no selection)",
                   "test_pids": te, "y": yte.tolist(), "train_mask": TRAIN_MASK[cond], "prob": {}}
            for m, X in Xte.items():
                res["prob"][str(m)] = predict(model, X).tolist()
            res["train_sec"] = time.time() - t0
            save_json(fp, res)
            from sklearn.metrics import roc_auc_score
            msg = " ".join(f"{k}:AUC={roc_auc_score(yte, v):.3f},acc={np.mean((np.array(v) >= .5) == yte):.3f}"
                           for k, v in res["prob"].items())
            print(f"[holdout {cond} s{seed}] {msg} ({res['train_sec']:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
    if "m2d" in conds and not balanced_sampler:
        fp = S2_OUT / "holdout_radiomics.json"
        p = radiomics_fit_predict(tr, ytr, te, "m2d", "m2d")
        save_json(fp, {"condition": "radiomics_lr", "test_pids": te, "y": yte.tolist(), "prob": {"m2d": p.tolist()},
                       "features": "volume, mean, SD, max, P90 of T2 within the 2D-U-Net WT mask (OOF for training)"})


def reeval_holdout() -> None:
    """Add predictions for test-mask sources that became available after a model was trained."""
    from common import load_json
    c = cohort()
    te = c["test"]
    for fp in sorted(S2_OUT.glob("holdout_*_seed*.json")):
        res = load_json(fp)
        cond, seed = res["condition"], res["seed"]
        todo = [m for m in TEST_MASKS[cond] if str(m) not in res["prob"]
                and all((CLS_CACHE / m / f"{p:03d}.npy").exists() for p in te)]
        if not todo:
            continue
        model = TumorGradeClassifier(in_channels=1 if cond == "t2" else 2).to(DEV)
        model.load_state_dict(torch.load(S2_CKPT / f"holdout_{cond}_seed{seed}.pth", map_location=DEV))
        for m in todo:
            res["prob"][str(m)] = predict(model, stack_inputs(te, m)).tolist()
        save_json(fp, res)
        print("re-evaluated", fp.name, todo, flush=True)


def run_cv(seeds: list[int], epochs: int, conds=("t2", "m2d")) -> None:
    """5-fold CV inside the 295-patient development set using the Stage-1 folds: validation-fold masks are
    OOF predictions of the Stage-1 model that did not see those patients."""
    c = cohort()
    for k, f in enumerate(c["folds"]):
        tr, va = f["train"], f["heldout"]
        ytr = np.array([c["label_of"][p] for p in tr])
        yva = np.array([c["label_of"][p] for p in va])
        for cond in conds:
            Xtr = stack_inputs(tr, TRAIN_MASK[cond])
            Xva = stack_inputs(va, TRAIN_MASK[cond])
            for seed in seeds:
                fp = S2_OUT / f"cv_fold{k}_{cond}_seed{seed}.json"
                if fp.exists():
                    continue
                t0 = time.time()
                model = train_model(Xtr, ytr, seed, epochs)
                prob = predict(model, Xva)
                save_json(fp, {"fold": k, "condition": cond, "seed": seed, "val_pids": va, "y": yva.tolist(),
                               "prob": prob.tolist(), "train_sec": time.time() - t0})
                from sklearn.metrics import roc_auc_score
                print(f"[cv f{k} {cond} s{seed}] AUC={roc_auc_score(yva, prob):.3f} "
                      f"acc={np.mean((prob >= .5) == yva):.3f} ({time.time() - t0:.0f}s)", flush=True)
                del model
                torch.cuda.empty_cache()
        fp = S2_OUT / f"cv_fold{k}_radiomics.json"
        masks_ready = all((MASK_CACHE / "stage1_oof" / f"{p:03d}.npy").exists() for p in tr + va)
        if not fp.exists() and masks_ready:
            p = radiomics_fit_predict(tr, ytr, va, "m2d", "m2d")
            save_json(fp, {"fold": k, "condition": "radiomics_lr", "val_pids": va, "y": yva.tolist(), "prob": p.tolist()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cache", nargs="*", default=[])
    ap.add_argument("--holdout", nargs="*", default=[])
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--cv-conds", nargs="*", default=["t2", "m2d"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--reeval", action="store_true")
    ap.add_argument("--balanced-sampler", action="store_true",
                    help="double-balanced recipe (sampler + weighted loss); hold-out only, own output folder")
    a = ap.parse_args()
    if a.build_cache:
        build_cache(a.build_cache)
    if a.holdout:
        run_holdout(a.holdout, a.seeds, a.epochs, a.balanced_sampler)
    if a.reeval:
        reeval_holdout()
    if a.cv:
        run_cv(a.seeds, a.epochs, tuple(a.cv_conds))
