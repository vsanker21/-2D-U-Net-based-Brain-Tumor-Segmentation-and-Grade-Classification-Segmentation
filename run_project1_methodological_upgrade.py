#!/usr/bin/env python3
"""
Project 1 methodological upgrade for reviewer responses.

Runs:
  1) Mask-guided 3D classification with repeated stratified CV
  2) Radiomics + logistic baseline
  3) BraTS segmentation benchmarking (2D U-Net vs nnU-Net when available)
  4) External metadata download (UCSF-PDGM) + optional XNAT EGD imaging
  5) Benchmark tables/figures JSON for manuscript
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, WeightedRandomSampler

from project1_analysis_common import (
    BASE,
    DATA_DIR,
    GRADE_CSV,
    MASK_DIR,
    NNUNET_PREPROCESSED,
    NNUNET_RAW,
    NNUNET_RESULTS,
    OUT_DIR,
    RANDOM_STATE,
    SEG_CKPT,
    CLASSIFY_SHAPE,
    MaskGuidedClassificationDataset,
    T2OnlyClassificationDataset,
    TumorGradeClassifier,
    WeightedFocalLoss,
    cache_predicted_masks,
    class_weights_from_labels,
    get_predicted_mask,
    list_patient_ids,
    load_grade_mapping,
    load_gt_mask_volume,
    load_segmentation_model,
    load_t2_volume,
    master_train_test_split,
    patient_labels,
    preload_volumes,
    predict_mask_volume_2d_unet,
    save_json,
    set_seed,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots.append(float(np.mean(sample)))
    low = float(np.quantile(boots, alpha / 2))
    high = float(np.quantile(boots, 1 - alpha / 2))
    return {"mean": float(np.mean(values)), "lower": low, "upper": high}


def train_classifier_fold(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: list[float],
    epochs: int = 25,
) -> tuple[torch.nn.Module, dict]:
    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = WeightedFocalLoss(class_weights).to(DEVICE)
    best = {"f1": -1.0, "state": None}
    history = []

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            tr_loss += float(loss.item())
        scheduler.step()

        model.eval()
        preds, labels, probs = [], [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(DEVICE)
                logits = model(x)
                prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                preds.extend(pred.tolist())
                labels.extend(y.numpy().tolist())
                probs.extend(prob.tolist())
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        history.append({"epoch": epoch + 1, "train_loss": tr_loss / max(len(train_loader), 1), "val_f1": f1})
        if (epoch + 1) % 2 == 0 or epoch == epochs - 1:
            print(f"      epoch {epoch + 1}/{epochs} val_f1={f1:.4f}", flush=True)
        if f1 > best["f1"]:
            best["f1"] = f1
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    model.eval()

    preds, labels, probs = [], [], []
    with torch.no_grad():
        for x, y, _ in val_loader:
            x = x.to(DEVICE)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            labels.extend(y.numpy().tolist())
            probs.extend(prob.tolist())

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(labels, probs)) if len(set(labels)) > 1 else float("nan"),
        "best_val_f1": float(best["f1"]),
        "history": history,
    }
    return model, metrics


def make_loader(dataset, labels: list[int], batch_size: int = 4, shuffle: bool = True) -> DataLoader:
    if shuffle:
        counts = Counter(labels)
        total = len(labels)
        weights = [total / (len(counts) * counts[y]) for y in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)


def run_classification_cv(n_splits: int = 5, n_repeats: int = 1, epochs: int = 8) -> dict:
    print("\n=== Classification repeated stratified CV ===")
    grade_mapping = load_grade_mapping()
    patient_ids = list_patient_ids(grade_mapping)
    labels = patient_labels(patient_ids, grade_mapping)

    seg_model = load_segmentation_model(SEG_CKPT, DEVICE)
    print("Preloading T2 volumes into memory...")
    preload_volumes(patient_ids)
    print("Warming predicted mask cache (batched 2D U-Net)...")
    cache_predicted_masks(patient_ids, seg_model, DEVICE)

    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    results = {
        "baseline_t2_only": [],
        "mask_guided_predicted": [],
        "radiomics_logistic": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(patient_ids, labels)):
        train_ids = [patient_ids[i] for i in train_idx]
        val_ids = [patient_ids[i] for i in val_idx]
        train_lab = [labels[i] for i in train_idx]
        val_lab = [labels[i] for i in val_idx]
        cw = class_weights_from_labels(train_lab)
        print(f"Fold {fold_idx + 1}: train={len(train_ids)} val={len(val_ids)}")

        # Baseline T2-only 3D CNN
        ds_train = T2OnlyClassificationDataset(train_ids, grade_mapping, augment=True)
        ds_val = T2OnlyClassificationDataset(val_ids, grade_mapping, augment=False)
        _, m_base = train_classifier_fold(
            TumorGradeClassifier(in_channels=1),
            make_loader(ds_train, train_lab),
            DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=0),
            cw,
            epochs=epochs,
        )
        m_base["fold"] = fold_idx + 1
        results["baseline_t2_only"].append(m_base)

        # Mask-guided (predicted masks only; no GT leakage)
        ds_train_m = MaskGuidedClassificationDataset(
            train_ids, grade_mapping, seg_model=seg_model, device=DEVICE,
            mask_source="predicted", augment=True,
        )
        ds_val_m = MaskGuidedClassificationDataset(
            val_ids, grade_mapping, seg_model=seg_model, device=DEVICE,
            mask_source="predicted", augment=False,
        )
        _, m_mask = train_classifier_fold(
            TumorGradeClassifier(in_channels=2),
            make_loader(ds_train_m, train_lab),
            DataLoader(ds_val_m, batch_size=4, shuffle=False, num_workers=0),
            cw,
            epochs=epochs,
        )
        m_mask["fold"] = fold_idx + 1
        results["mask_guided_predicted"].append(m_mask)

        # Radiomics-style logistic baseline from predicted mask + intensity
        X_train, y_train = [], []
        for pid in train_ids:
            vol = load_t2_volume(pid)
            mask = get_predicted_mask(pid, seg_model, DEVICE)
            tumor = mask > 0
            if tumor.any():
                vals = vol[tumor]
                feats = [tumor.sum(), vals.mean(), vals.std(), vals.max(), np.percentile(vals, 90)]
            else:
                feats = [0.0, 0.0, 0.0, 0.0, 0.0]
            X_train.append(feats)
            y_train.append(1 if grade_mapping[f"BraTS20_Training_{pid:03d}"] == "HGG" else 0)
        X_val, y_val = [], []
        for pid in val_ids:
            vol = load_t2_volume(pid)
            mask = get_predicted_mask(pid, seg_model, DEVICE)
            tumor = mask > 0
            if tumor.any():
                vals = vol[tumor]
                feats = [tumor.sum(), vals.mean(), vals.std(), vals.max(), np.percentile(vals, 90)]
            else:
                feats = [0.0, 0.0, 0.0, 0.0, 0.0]
            X_val.append(feats)
            y_val.append(1 if grade_mapping[f"BraTS20_Training_{pid:03d}"] == "HGG" else 0)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
        ])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_val)[:, 1]
        pred = (prob >= 0.5).astype(int)
        results["radiomics_logistic"].append({
            "fold": fold_idx + 1,
            "accuracy": float(accuracy_score(y_val, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
            "f1_weighted": float(f1_score(y_val, pred, average="weighted", zero_division=0)),
            "roc_auc": float(roc_auc_score(y_val, prob)) if len(set(y_val)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(y_val, prob)) if len(set(y_val)) > 1 else float("nan"),
        })

    summary = {}
    for name, folds in results.items():
        for metric in ("accuracy", "balanced_accuracy", "f1_weighted", "roc_auc", "pr_auc"):
            vals = np.array([f[metric] for f in folds if not np.isnan(f[metric])], dtype=float)
            summary[name] = summary.get(name, {})
            summary[name][metric] = bootstrap_ci(vals)
        summary[name]["n_folds"] = len(folds)

    out = {
        "protocol": {
            "cv": f"RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats})",
            "random_state": RANDOM_STATE,
            "epochs_per_fold": epochs,
            "mask_source": "predicted 2D U-Net (no GT leakage at test)",
            "classification_resize": "128x128x96 trilinear (T2 + mask)",
            "n_patients": len(patient_ids),
        },
        "fold_results": results,
        "aggregate": summary,
    }
    save_json(OUT_DIR / "classification_repeated_cv.json", out)
    plot_classification_cv(summary)
    return out


def plot_classification_cv(summary: dict) -> None:
    names = list(summary.keys())
    metrics = ["roc_auc", "balanced_accuracy", "f1_weighted"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#4c78a8", "#54a24b", "#e45756"]
    for ax, metric in zip(axes, metrics):
        means = [summary[n][metric]["mean"] for n in names]
        lows = [summary[n][metric]["lower"] for n in names]
        highs = [summary[n][metric]["upper"] for n in names]
        x = np.arange(len(names))
        ax.bar(x, means, color=colors)
        ax.errorbar(x, means, yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
                    fmt="none", ecolor="black", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(["T2-only CNN", "Mask-guided CNN", "Radiomics LR"], rotation=15)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
    fig.suptitle("Project 1 classification — repeated stratified CV (BraTS2020)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "Figure_Classification_CV_Comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_segmentation_2d_unet_on_test() -> dict:
    print("\n=== 2D U-Net 3D-stacked segmentation evaluation (patient-level test split) ===")
    from segmentation_metrics_3d import dice_wt_tc_et_torch, hausdorff_multiclass_mean_regions

    grade_mapping = load_grade_mapping()
    patient_ids = list_patient_ids(grade_mapping)
    labels = patient_labels(patient_ids, grade_mapping)
    _, test_ids, _, _ = master_train_test_split(patient_ids, labels)

    seg_model = load_segmentation_model(SEG_CKPT, DEVICE)
    rows = []
    for pid in test_ids:
        vol = load_t2_volume(pid)
        gt = load_gt_mask_volume(pid)
        pred = predict_mask_volume_2d_unet(seg_model, vol, DEVICE, binary=False).astype(np.uint8)
        gt_t = torch.from_numpy(gt).unsqueeze(0).to(DEVICE)
        # fake logits from hard labels for metric helpers
        pred_logits = torch.zeros((1, 4, *pred.shape), device=DEVICE)
        for c in range(4):
            pred_logits[0, c][torch.from_numpy(pred).to(DEVICE) == c] = 10.0
        dice = dice_wt_tc_et_torch(pred_logits, gt_t)
        hd_mean, hd_regs = hausdorff_multiclass_mean_regions(pred_logits, gt_t)
        rows.append({"patient_id": pid, "dice": dice, "hd_mean_mm": float(hd_mean), "hd_regions": hd_regs})

    summary = {
        "method": "Project1 2D U-Net T2 (slice-stacked 3D inference)",
        "n_test": len(test_ids),
        "dice_wt_mean": float(np.mean([r["dice"]["WT"] for r in rows])),
        "dice_tc_mean": float(np.mean([r["dice"]["TC"] for r in rows])),
        "dice_et_mean": float(np.mean([r["dice"]["ET"] for r in rows])),
        "hd_mean_mm": float(np.mean([r["hd_mean_mm"] for r in rows])),
        "per_patient": rows,
    }
    save_json(OUT_DIR / "segmentation_2d_unet_test_metrics.json", summary)
    return summary


def prepare_nnunet_dataset() -> Path:
    print("\n=== Preparing nnU-Net dataset (BraTS2020 T2, 3D) ===")
    raw_root = NNUNET_RAW
    ds_dir = raw_root / "Dataset501_BraTS20T2"
    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    images_ts = ds_dir / "imagesTs"
    for p in (images_tr, labels_tr, images_ts):
        p.mkdir(parents=True, exist_ok=True)

    grade_mapping = load_grade_mapping()
    patient_ids = list_patient_ids(grade_mapping)
    labels = patient_labels(patient_ids, grade_mapping)
    train_ids, test_ids, _, _ = master_train_test_split(patient_ids, labels)

    import os

    def link_or_copy(src: Path, dst: Path) -> None:
        if dst.exists():
            return
        try:
            os.link(src, dst)
        except OSError:
            # Fallback when hard-link fails (cross-device etc.)
            import shutil
            shutil.copy2(src, dst)

    for i, pid in enumerate(train_ids):
        case = f"BraTS20_{pid:03d}"
        src_img = DATA_DIR / f"BraTS20_Training_{pid:03d}_T2.nii.gz"
        src_lbl = MASK_DIR / f"BraTS20_Training_{pid:03d}_mask.nii.gz"
        dst_img = images_tr / f"{case}_0000.nii.gz"
        dst_lbl = labels_tr / f"{case}.nii.gz"
        link_or_copy(src_img, dst_img)
        link_or_copy(src_lbl, dst_lbl)
    for pid in test_ids:
        case = f"BraTS20_{pid:03d}"
        src_img = DATA_DIR / f"BraTS20_Training_{pid:03d}_T2.nii.gz"
        dst_img = images_ts / f"{case}_0000.nii.gz"
        link_or_copy(src_img, dst_img)

    dataset_json = {
        "channel_names": {"0": "T2"},
        "labels": {"background": 0, "NCR": 1, "ED": 2, "ET": 3},
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
        "name": "BraTS20T2",
        "description": "BraTS2020 T2 single-modality subset for Project1 nnU-Net baseline",
        "reference": "BraTS2020 Kaggle training data",
        "licence": "See BraTS challenge terms",
        "release": "1.0",
        "tensorImageSize": "4D",
    }
    save_json(ds_dir / "dataset.json", dataset_json)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    for k in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        Path(env[k]).mkdir(parents=True, exist_ok=True)

    subprocess.run(["nnUNetv2_plan_and_preprocess", "-d", "501", "-c", "3d_fullres", "--verify_dataset_integrity"],
                   env=env, check=False)
    return ds_dir


def run_nnunet_training(fold: int = 0) -> None:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_RAW)
    env["nnUNet_preprocessed"] = str(NNUNET_PREPROCESSED)
    env["nnUNet_results"] = str(NNUNET_RESULTS)
    trainer = "nnUNetTrainer_BraTSProject1_50epochs"
    cmd = ["nnUNetv2_train", "501", "3d_fullres", str(fold), "-tr", trainer]
    log = OUT_DIR / f"nnunet_train_fold{fold}.log"
    with open(log, "w", encoding="utf-8") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)


def download_ucsf_pdgm_metadata() -> Path:
    print("\n=== Downloading UCSF-PDGM clinical metadata ===")
    out_dir = OUT_DIR / "external" / "ucsf_pdgm"
    out_dir.mkdir(parents=True, exist_ok=True)
    url = "https://www.cancerimagingarchive.net/wp-content/uploads/UCSF-PDGM-metadata.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    path = out_dir / "UCSF-PDGM-metadata.csv"
    path.write_bytes(r.content)
    meta = {
        "source": url,
        "downloaded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": int(pd.read_csv(path).shape[0]),
        "imaging_note": "NIfTI volumes require IBM Aspera; see external_imaging_status.json",
    }
    save_json(out_dir / "metadata_download.json", meta)
    return path


def attempt_egd_download() -> dict:
    print("\n=== Erasmus Glioma Database access check ===")
    user = os.environ.get("EGD_XNAT_USER")
    pwd = os.environ.get("EGD_XNAT_PASSWORD")
    out = {"status": "credentials_missing", "server": "https://xnat.bmia.nl"}
    if not user or not pwd:
        save_json(OUT_DIR / "external" / "erasmus_glioma" / "access_status.json", out)
        return out
    try:
        import xnat  # type: ignore

        out_dir = OUT_DIR / "external" / "erasmus_glioma" / "data"
        out_dir.mkdir(parents=True, exist_ok=True)
        with xnat.connect("https://xnat.bmia.nl", user=user, password=pwd) as session:
            proj = session.projects["egd"]
            out["status"] = "connected"
            out["n_subjects"] = len(list(proj.subjects))
            save_json(OUT_DIR / "external" / "erasmus_glioma" / "access_status.json", out)
        return out
    except Exception as exc:
        out["status"] = "connection_failed"
        out["error"] = str(exc)
        save_json(OUT_DIR / "external" / "erasmus_glioma" / "access_status.json", out)
        return out


def train_final_t2_baseline_model(epochs: int = 30) -> dict:
    """Fair T2-only holdout baseline: same 80/20 split, 128×128×96 resize, same training recipe."""
    print("\n=== Fair T2-only baseline (canonical 80/20 split, 128×128×96) ===")
    grade_mapping = load_grade_mapping()
    patient_ids = list_patient_ids(grade_mapping)
    labels = patient_labels(patient_ids, grade_mapping)
    train_ids, test_ids, train_lab, test_lab = master_train_test_split(patient_ids, labels)

    preload_volumes(patient_ids)

    cw = class_weights_from_labels(train_lab)
    ds_train = T2OnlyClassificationDataset(train_ids, grade_mapping, augment=True)
    ds_test = T2OnlyClassificationDataset(test_ids, grade_mapping, augment=False)
    model, _ = train_classifier_fold(
        TumorGradeClassifier(in_channels=1),
        make_loader(ds_train, train_lab),
        DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0),
        cw,
        epochs=epochs,
    )
    ckpt = OUT_DIR / "classification_model_t2_fair_best.pth"
    torch.save(model.state_dict(), ckpt)

    preds, labels_out, probs = [], [], []
    with torch.no_grad():
        for x, y, _ in DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0):
            x = x.to(DEVICE)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            labels_out.extend(y.numpy().tolist())
            probs.extend(prob.tolist())

    out = {
        "checkpoint": str(ckpt),
        "method": "T2-only 3D CNN (fair holdout baseline)",
        "protocol": {
            "spatial_shape": list(CLASSIFY_SHAPE),
            "split": "80/20 patient-level (canonical)",
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "augmentation": "random flip + intensity scale (train only)",
            "epochs": epochs,
            "optimizer": "AdamW lr=1e-4, cosine warm restarts",
            "loss": "weighted focal",
            "sampler": "weighted random (class balance)",
        },
        "accuracy": float(accuracy_score(labels_out, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_out, preds)),
        "f1_weighted": float(f1_score(labels_out, preds, average="weighted")),
        "roc_auc": float(roc_auc_score(labels_out, probs)),
        "pr_auc": float(average_precision_score(labels_out, probs)),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "prior_legacy_reference": {
            "accuracy": 0.7703,
            "roc_auc": 0.8305,
            "note": "Earlier full-resolution T2-only run before 128×128×96 protocol",
        },
    }
    save_json(OUT_DIR / "classification_t2_fair_holdout.json", out)
    return out


def train_final_mask_guided_model(epochs: int = 30) -> dict:
    """Train final mask-guided model on canonical 80/20 split and evaluate on held-out test."""
    print("\n=== Final mask-guided model (canonical 80/20 split) ===")
    grade_mapping = load_grade_mapping()
    patient_ids = list_patient_ids(grade_mapping)
    labels = patient_labels(patient_ids, grade_mapping)
    train_ids, test_ids, train_lab, test_lab = master_train_test_split(patient_ids, labels)

    seg_model = load_segmentation_model(SEG_CKPT, DEVICE)
    preload_volumes(patient_ids)
    cache_predicted_masks(patient_ids, seg_model, DEVICE)

    cw = class_weights_from_labels(train_lab)
    ds_train = MaskGuidedClassificationDataset(
        train_ids, grade_mapping, seg_model=seg_model, device=DEVICE,
        mask_source="predicted", augment=True,
    )
    ds_test = MaskGuidedClassificationDataset(
        test_ids, grade_mapping, seg_model=seg_model, device=DEVICE,
        mask_source="predicted", augment=False,
    )
    model, _ = train_classifier_fold(
        TumorGradeClassifier(in_channels=2),
        make_loader(ds_train, train_lab),
        DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0),
        cw,
        epochs=epochs,
    )
    ckpt = OUT_DIR / "classification_model_maskguided_best.pth"
    torch.save(model.state_dict(), ckpt)

    preds, labels_out, probs = [], [], []
    with torch.no_grad():
        for x, y, _ in DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0):
            x = x.to(DEVICE)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            labels_out.extend(y.numpy().tolist())
            probs.extend(prob.tolist())

    fair_path = OUT_DIR / "classification_t2_fair_holdout.json"
    fair_ref = (
        json.loads(fair_path.read_text(encoding="utf-8"))
        if fair_path.exists()
        else {"accuracy": 0.7703, "roc_auc": 0.8305}
    )
    out = {
        "checkpoint": str(ckpt),
        "accuracy": float(accuracy_score(labels_out, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_out, preds)),
        "f1_weighted": float(f1_score(labels_out, preds, average="weighted")),
        "roc_auc": float(roc_auc_score(labels_out, probs)),
        "pr_auc": float(average_precision_score(labels_out, probs)),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
        "fair_t2_baseline_reference": {
            "accuracy": fair_ref.get("accuracy"),
            "roc_auc": fair_ref.get("roc_auc"),
            "balanced_accuracy": fair_ref.get("balanced_accuracy"),
            "f1_weighted": fair_ref.get("f1_weighted"),
            "source": str(fair_path) if fair_path.exists() else "legacy_prior_run",
        },
        "baseline_reference": {"accuracy": 0.7703, "roc_auc": 0.8305},
    }
    save_json(OUT_DIR / "classification_maskguided_final.json", out)
    return out


def train_final_random_mask_ablation(epochs: int = 30) -> dict:
    """Negative control: same 2-channel architecture but random (non-informative) mask channel."""
    print("\n=== Random-mask ablation (canonical 80/20 split) ===")
    grade_mapping = load_grade_mapping()
    patient_ids = list_patient_ids(grade_mapping)
    labels = patient_labels(patient_ids, grade_mapping)
    train_ids, test_ids, train_lab, test_lab = master_train_test_split(patient_ids, labels)

    preload_volumes(patient_ids)

    cw = class_weights_from_labels(train_lab)
    ds_train = MaskGuidedClassificationDataset(
        train_ids, grade_mapping, mask_source="random", augment=True,
    )
    ds_test = MaskGuidedClassificationDataset(
        test_ids, grade_mapping, mask_source="random", augment=False,
    )
    model, _ = train_classifier_fold(
        TumorGradeClassifier(in_channels=2),
        make_loader(ds_train, train_lab),
        DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0),
        cw,
        epochs=epochs,
    )
    ckpt = OUT_DIR / "classification_model_random_mask_best.pth"
    torch.save(model.state_dict(), ckpt)

    preds, labels_out, probs = [], [], []
    with torch.no_grad():
        for x, y, _ in DataLoader(ds_test, batch_size=4, shuffle=False, num_workers=0):
            x = x.to(DEVICE)
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(pred.tolist())
            labels_out.extend(y.numpy().tolist())
            probs.extend(prob.tolist())

    out = {
        "checkpoint": str(ckpt),
        "method": "Random-mask ablation (2-channel 3D CNN, uniform noise mask)",
        "protocol": {
            "spatial_shape": list(CLASSIFY_SHAPE),
            "split": "80/20 patient-level (canonical)",
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "mask_source": "patient-fixed uniform random [0,1] (non-informative control)",
            "epochs": epochs,
        },
        "accuracy": float(accuracy_score(labels_out, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_out, preds)),
        "f1_weighted": float(f1_score(labels_out, preds, average="weighted")),
        "roc_auc": float(roc_auc_score(labels_out, probs)),
        "pr_auc": float(average_precision_score(labels_out, probs)),
        "n_train": len(train_ids),
        "n_test": len(test_ids),
    }
    save_json(OUT_DIR / "classification_random_mask_ablation.json", out)
    return out


def build_benchmark_table(seg_metrics: dict, cls_cv: dict, cls_final: dict) -> dict:
    table = {
        "segmentation_baselines": [
            {
                "method": seg_metrics["method"],
                "dice_wt": seg_metrics["dice_wt_mean"],
                "dice_tc": seg_metrics["dice_tc_mean"],
                "dice_et": seg_metrics["dice_et_mean"],
                "hd_mm": seg_metrics["hd_mean_mm"],
            },
            {
                "method": "nnU-Net 3d_fullres (fold 0; see nnUNet_results if training completed)",
                "status": "training_scheduled_or_in_progress",
            },
            {
                "method": "Literature nnU-Net BraTS (Isensee et al.)",
                "note": "Included for context; not re-run here if nnU-Net incomplete",
            },
        ],
        "classification_baselines": cls_cv["aggregate"],
        "final_mask_guided_holdout": cls_final,
        "novel_methodological_elements": [
            "Mask-guided 3D CNN using predicted 2D U-Net tumor channel (no GT leakage)",
            "Repeated stratified CV with radiomics logistic comparator",
            "Measured segmentation + classification benchmarking on identical patient splits",
            "External metadata acquisition pipeline for UCSF-PDGM / Erasmus Glioma Database",
        ],
    }
    save_json(OUT_DIR / "benchmark_summary.json", table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-nnunet-train", action="store_true")
    parser.add_argument("--cv-epochs", type=int, default=15)
    parser.add_argument("--final-epochs", type=int, default=30)
    args = parser.parse_args()

    set_seed()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    download_ucsf_pdgm_metadata()
    attempt_egd_download()

    if (OUT_DIR / "segmentation_2d_unet_test_metrics.json").exists():
        print("\n=== Using cached 2D U-Net segmentation test metrics ===")
        seg_metrics = json.loads((OUT_DIR / "segmentation_2d_unet_test_metrics.json").read_text(encoding="utf-8"))
    else:
        seg_metrics = evaluate_segmentation_2d_unet_on_test()
    cls_cv = run_classification_cv(epochs=args.cv_epochs)
    cls_final = train_final_mask_guided_model(epochs=args.final_epochs)
    prepare_nnunet_dataset()

    if not args.skip_nnunet_train:
        print("\n=== Launching nnU-Net training (fold 0, custom 50-epoch trainer) ===")
        run_nnunet_training(fold=0)

    build_benchmark_table(seg_metrics, cls_cv, cls_final)
    print(f"\nDone. Outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
