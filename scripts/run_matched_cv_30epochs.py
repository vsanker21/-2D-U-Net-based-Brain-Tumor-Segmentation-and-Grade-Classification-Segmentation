#!/usr/bin/env python3
"""Matched 30-epoch repeated stratified CV (Reviewer 3). Preserves 12-epoch JSON/figure."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"E:\Brain Tumor Segmentation")

from project1_analysis_common import OUT_DIR, save_json
from run_project1_methodological_upgrade import run_classification_cv

OUT_JSON_30 = OUT_DIR / "classification_repeated_cv_30epochs.json"
OUT_FIG_30 = OUT_DIR / "Figure_Classification_CV_Comparison_30epochs.png"
LEGACY_JSON = OUT_DIR / "classification_repeated_cv.json"
LEGACY_JSON_BAK = OUT_DIR / "classification_repeated_cv_12epochs.json"
LEGACY_FIG = OUT_DIR / "Figure_Classification_CV_Comparison.png"
MF_FIG = Path(r"E:\Brain Tumor Segmentation\manuscript_figures\Figure_Classification_CV_Comparison.png")


def main() -> None:
    # Preserve prior 12-epoch results before overwrite
    if LEGACY_JSON.exists():
        try:
            proto = json.loads(LEGACY_JSON.read_text(encoding="utf-8")).get("protocol", {})
            if proto.get("epochs_per_fold", 12) == 12 or not LEGACY_JSON_BAK.exists():
                shutil.copy2(LEGACY_JSON, LEGACY_JSON_BAK)
                print("Backed up prior CV JSON →", LEGACY_JSON_BAK)
        except Exception as e:
            print("Backup warning:", e)
            shutil.copy2(LEGACY_JSON, LEGACY_JSON_BAK)

    print("Starting matched 30-epoch CV (5-fold × T2-only + mask-guided + radiomics)...")
    out = run_classification_cv(n_splits=5, n_repeats=1, epochs=30)
    save_json(OUT_JSON_30, out)

    if LEGACY_FIG.exists():
        shutil.copy2(LEGACY_FIG, OUT_FIG_30)
    if MF_FIG.exists():
        shutil.copy2(MF_FIG, LEGACY_FIG)
    if LEGACY_JSON_BAK.exists():
        shutil.copy2(LEGACY_JSON_BAK, LEGACY_JSON)

    agg = out.get("aggregate", {})
    summary = {
        "protocol": out.get("protocol", {}),
        "aggregate": agg,
        "headline": {
            k: {
                "accuracy_mean": agg.get(k, {}).get("accuracy", {}).get("mean"),
                "accuracy_ci": [
                    agg.get(k, {}).get("accuracy", {}).get("lower"),
                    agg.get(k, {}).get("accuracy", {}).get("upper"),
                ],
                "roc_auc_mean": agg.get(k, {}).get("roc_auc", {}).get("mean"),
                "roc_auc_ci": [
                    agg.get(k, {}).get("roc_auc", {}).get("lower"),
                    agg.get(k, {}).get("roc_auc", {}).get("upper"),
                ],
                "balanced_accuracy_mean": agg.get(k, {}).get("balanced_accuracy", {}).get("mean"),
            }
            for k in ("baseline_t2_only", "mask_guided_predicted", "radiomics_logistic")
        },
        "comparison_to_12epoch": {
            "note": "12-epoch results retained in classification_repeated_cv_12epochs.json",
            "question": "Does matched 30-epoch budget reverse/restore mask-guided vs T2-only ranking?",
        },
    }
    (OUT_DIR / "classification_repeated_cv_30epochs_headline.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Wrote", OUT_JSON_30)
    print("Wrote headline JSON")
    print("Wrote", OUT_FIG_30)


if __name__ == "__main__":
    main()
