#!/usr/bin/env python3
"""Statistics for Stage-2 classification (hold-out and development-set CV).

Positive class for all threshold-free metrics: HGG (label 1). PR-AUC is reported twice:
  PR-AUC(HGG+)  no-skill baseline = HGG prevalence (59/74 = 0.797 on the hold-out set)
  PR-AUC(LGG+)  computed with LGG as positive and 1 - P(HGG) as score; baseline = 15/74 = 0.203
Primary prediction per condition = mean P(HGG) over training seeds (seed ensemble); per-seed spread reported.
CIs: stratified patient-level bootstrap (2,000 resamples, HGG and LGG resampled separately).
Paired tests vs T2-only: DeLong test for correlated ROC-AUCs, bootstrap ΔAUC, exact McNemar (threshold 0.5),
bootstrap Δ balanced accuracy. Holm correction within each pre-specified family.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, load_json, save_json  # noqa: E402

S2 = OUT / "stage2"
NBOOT = 2000


# ----------------------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {
        "accuracy": float(accuracy_score(y, yhat)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yhat)),
        "sensitivity_HGG": float(np.mean(yhat[y == 1] == 1)),
        "specificity_LGG": float(np.mean(yhat[y == 0] == 0)),
        "f1_weighted": float(f1_score(y, yhat, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc_hgg_pos": float(average_precision_score(y, p)),
        "pr_auc_lgg_pos": float(average_precision_score(1 - y, 1 - p)),
        "brier": float(brier_score_loss(y, p)),
        "n_pred_LGG": int((yhat == 0).sum()),
    }


def strat_boot_idx(y: np.ndarray, rng: np.random.Generator):
    i1, i0 = np.where(y == 1)[0], np.where(y == 0)[0]
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])


def boot_ci(y, p, n=NBOOT, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    keys = list(metrics(y, p).keys())
    acc = {k: [] for k in keys}
    for _ in range(n):
        idx = strat_boot_idx(y, rng)
        m = metrics(y[idx], p[idx])
        for k in keys:
            acc[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in acc.items()}


# ----------------------------------------------------------------------------- DeLong (Sun & Xu 2014)
def _midrank(x):
    j = np.argsort(x)
    z = x[j]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        k = i
        while k < n and z[k] == z[i]:
            k += 1
        t[i:k] = 0.5 * (i + k - 1) + 1
        i = k
    out = np.empty(n)
    out[j] = t
    return out


def delong_paired(y, pa, pb) -> dict:
    order = np.argsort(-y)
    y = y[order]
    preds = np.vstack([pa[order], pb[order]])
    m = int(y.sum())
    n = len(y) - m
    tx = np.array([_midrank(r[:m]) for r in preds])
    ty = np.array([_midrank(r[m:]) for r in preds])
    tz = np.array([_midrank(r) for r in preds])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s = np.cov(v01) / m + np.cov(v10) / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    d = aucs[0] - aucs[1]
    z = d / np.sqrt(var) if var > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    se = np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": float(d), "se": float(se),
            "ci95": [float(d - 1.96 * se), float(d + 1.96 * se)], "z": float(z), "p": float(p)}


def mcnemar_exact(y, pa, pb) -> dict:
    ca = (pa >= 0.5).astype(int) == y
    cb = (pb >= 0.5).astype(int) == y
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    p = stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return {"a_only_correct": b, "b_only_correct": c, "p": float(p)}


def boot_delta(y, pa, pb, n=NBOOT, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    dauc, dbal = [], []
    for _ in range(n):
        idx = strat_boot_idx(y, rng)
        dauc.append(roc_auc_score(y[idx], pa[idx]) - roc_auc_score(y[idx], pb[idx]))
        dbal.append(balanced_accuracy_score(y[idx], (pa[idx] >= .5).astype(int))
                    - balanced_accuracy_score(y[idx], (pb[idx] >= .5).astype(int)))
    dauc, dbal = np.array(dauc), np.array(dbal)

    def two_sided(v):
        return float(min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean())))
    return {"dAUC_mean": float(dauc.mean()), "dAUC_ci95": [float(np.percentile(dauc, 2.5)), float(np.percentile(dauc, 97.5))],
            "dAUC_p": two_sided(dauc),
            "dBalAcc_mean": float(dbal.mean()), "dBalAcc_ci95": [float(np.percentile(dbal, 2.5)), float(np.percentile(dbal, 97.5))],
            "dBalAcc_p": two_sided(dbal)}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * pvals[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


# ----------------------------------------------------------------------------- hold-out
LABELS = {
    ("t2", "None"): "T2-only 3D CNN",
    ("m2d", "m2d"): "Mask-guided (2D U-Net mask; OOF-trained)",
    ("m2d", "nn250"): "Mask-guided, nnU-Net-250 mask at test",
    ("m2d", "nn50"): "Mask-guided, nnU-Net-50 mask at test",
    ("m2d", "gt"): "Mask-guided, GT mask at test",
    ("gt", "gt"): "GT-trained, GT mask at test (oracle)",
    ("gt", "nn250"): "GT-trained, nnU-Net-250 mask at test",
    ("gt", "nn50"): "GT-trained, nnU-Net-50 mask at test",
    ("gt", "m2d"): "GT-trained, 2D U-Net mask at test",
    ("rand", "rand"): "Random-mask control",
    ("m2d_ins", "m2d"): "Mask-guided, in-sample-trained (2D mask at test)",
}


def load_holdout() -> tuple[np.ndarray, dict]:
    runs: dict[tuple[str, str], dict[int, np.ndarray]] = {}
    y = None
    for fp in sorted(S2.glob("holdout_*_seed*.json")):
        r = load_json(fp)
        y = np.array(r["y"])
        for m, p in r["prob"].items():
            runs.setdefault((r["condition"], m), {})[r["seed"]] = np.array(p)
    return y, runs


def analyse_holdout() -> dict:
    y, runs = load_holdout()
    if y is None:
        return {}
    out = {"n": int(len(y)), "n_HGG": int(y.sum()), "n_LGG": int((1 - y).sum()),
           "pr_auc_baseline_hgg_pos": float(y.mean()), "pr_auc_baseline_lgg_pos": float(1 - y.mean()),
           "conditions": {}, "comparisons_vs_t2": {}}
    ens = {}
    for key, seeds in sorted(runs.items()):
        P = np.vstack([seeds[s] for s in sorted(seeds)])
        pe = P.mean(0)
        ens[key] = pe
        per_seed = {int(s): metrics(y, seeds[s]) for s in sorted(seeds)}
        agg = {}
        for k in next(iter(per_seed.values())):
            v = np.array([m[k] for m in per_seed.values()])
            agg[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                      "min": float(v.min()), "max": float(v.max())}
        out["conditions"][f"{key[0]}|{key[1]}"] = {
            "label": LABELS.get(key, str(key)), "n_seeds": len(seeds), "seeds": sorted(int(s) for s in seeds),
            "seed_ensemble": metrics(y, pe), "seed_ensemble_ci95": boot_ci(y, pe),
            "per_seed": per_seed, "per_seed_summary": agg, "prob_seed_ensemble": pe.tolist(),
        }
    rad = S2 / "holdout_radiomics.json"
    if rad.exists():
        r = load_json(rad)
        p = np.array(r["prob"]["m2d"])
        ens[("radiomics", "m2d")] = p
        out["conditions"]["radiomics|m2d"] = {"label": "Radiomics logistic regression (2D mask)",
                                              "seed_ensemble": metrics(y, p), "seed_ensemble_ci95": boot_ci(y, p),
                                              "prob_seed_ensemble": p.tolist()}
    ref = ens.get(("t2", "None"))
    if ref is not None:
        fam = [k for k in [("m2d", "m2d"), ("m2d", "nn250"), ("gt", "gt"), ("gt", "nn250"), ("rand", "rand")] if k in ens]
        comps = {}
        for k in ens:
            if k == ("t2", "None"):
                continue
            comps[f"{k[0]}|{k[1]}"] = {"delong": delong_paired(y, ens[k], ref), "mcnemar": mcnemar_exact(y, ens[k], ref),
                                       "bootstrap": boot_delta(y, ens[k], ref), "in_primary_family": k in fam}
        fk = [f"{k[0]}|{k[1]}" for k in fam]
        for test in ("delong", "mcnemar"):
            adj = holm([comps[k][test]["p"] for k in fk])
            for k, a in zip(fk, adj):
                comps[k][test]["p_holm"] = a
        out["comparisons_vs_t2"] = comps
        out["primary_family"] = fk
        # seed-paired consistency: fraction of seeds in which condition AUC > T2-only AUC (same seed)
        cons = {}
        for k in ens:
            if k[0] in ("t2", "radiomics"):
                continue
            common = sorted(set(runs[k]) & set(runs[("t2", "None")]))
            d = [roc_auc_score(y, runs[k][s]) - roc_auc_score(y, runs[("t2", "None")][s]) for s in common]
            db = [balanced_accuracy_score(y, (runs[k][s] >= .5).astype(int))
                  - balanced_accuracy_score(y, (runs[("t2", "None")][s] >= .5).astype(int)) for s in common]
            cons[f"{k[0]}|{k[1]}"] = {"seeds": common, "dAUC_per_seed": d, "dBalAcc_per_seed": db,
                                      "n_seeds_AUC_higher": int(np.sum(np.array(d) > 0))}
        out["seed_paired_consistency"] = cons
    return out


# ----------------------------------------------------------------------------- CV
def analyse_cv() -> dict:
    files = sorted(S2.glob("cv_fold*_seed*.json"))
    if not files:
        return {}
    by = {}
    for fp in files:
        r = load_json(fp)
        by.setdefault(r["condition"], {}).setdefault(r["seed"], {})[r["fold"]] = r
    for fp in sorted(S2.glob("cv_fold*_radiomics.json")):
        r = load_json(fp)
        by.setdefault("radiomics", {}).setdefault(0, {})[r["fold"]] = r
    out = {"conditions": {}}
    pooled_ens = {}
    y_pooled = None
    for cond, seeds in by.items():
        complete = {s: f for s, f in seeds.items() if len(f) == 5}
        if not complete:
            continue
        per_seed, fold_metrics = {}, {}
        probs_by_pid = {}
        for s, folds in complete.items():
            yy = np.concatenate([np.array(folds[k]["y"]) for k in range(5)])
            pp = np.concatenate([np.array(folds[k]["prob"]) for k in range(5)])
            pids = np.concatenate([np.array(folds[k]["val_pids"]) for k in range(5)])
            per_seed[int(s)] = metrics(yy, pp)
            for k in range(5):
                fold_metrics.setdefault(k, []).append(metrics(np.array(folds[k]["y"]), np.array(folds[k]["prob"])))
            for pid, p_, y_ in zip(pids, pp, yy):
                probs_by_pid.setdefault(int(pid), []).append(p_)
            y_map = {int(p): int(v) for p, v in zip(pids, yy)}
        pid_sorted = sorted(probs_by_pid)
        pe = np.array([np.mean(probs_by_pid[p]) for p in pid_sorted])
        ye = np.array([y_map[p] for p in pid_sorted])
        y_pooled = ye
        pooled_ens[cond] = pe
        # fold-level mean and t-based 95% CI (folds averaged over seeds)
        fm = {}
        for key in ("accuracy", "balanced_accuracy", "roc_auc", "pr_auc_hgg_pos", "pr_auc_lgg_pos"):
            v = np.array([np.mean([m[key] for m in fold_metrics[k]]) for k in range(5)])
            h = stats.t.ppf(0.975, 4) * v.std(ddof=1) / np.sqrt(5)
            fm[key] = {"mean": float(v.mean()), "ci95": [float(v.mean() - h), float(v.mean() + h)], "per_fold": v.tolist()}
        ps = {}
        for key in ("accuracy", "balanced_accuracy", "roc_auc"):
            v = np.array([m[key] for m in per_seed.values()])
            ps[key] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                       "min": float(v.min()), "max": float(v.max())}
        out["conditions"][cond] = {"n_seeds": len(complete), "pooled_oof_seed_ensemble": metrics(ye, pe),
                                   "pooled_oof_ci95": boot_ci(ye, pe), "fold_level": fm,
                                   "pooled_per_seed": per_seed, "pooled_per_seed_summary": ps}
    if "t2" in pooled_ens:
        comps = {}
        for cond, pe in pooled_ens.items():
            if cond == "t2":
                continue
            comps[cond] = {"delong": delong_paired(y_pooled, pe, pooled_ens["t2"]),
                           "mcnemar": mcnemar_exact(y_pooled, pe, pooled_ens["t2"]),
                           "bootstrap": boot_delta(y_pooled, pe, pooled_ens["t2"])}
        out["comparisons_vs_t2"] = comps
        out["n"] = int(len(y_pooled))
        out["n_HGG"] = int(y_pooled.sum())
    return out


if __name__ == "__main__":
    res = {"holdout": analyse_holdout(), "cv": analyse_cv()}
    save_json(OUT / "classification_stats.json", res)
    h = res["holdout"]
    for k, v in h.get("conditions", {}).items():
        m, ci = v["seed_ensemble"], v["seed_ensemble_ci95"]
        print(f"{v['label'][:48]:48s} n_seeds={v.get('n_seeds', '-')} acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} "
              f"AUC {m['roc_auc']:.3f} [{ci['roc_auc'][0]:.3f},{ci['roc_auc'][1]:.3f}] PR+HGG {m['pr_auc_hgg_pos']:.3f} "
              f"PR+LGG {m['pr_auc_lgg_pos']:.3f}")
    for k, v in h.get("comparisons_vs_t2", {}).items():
        print(f"  vs T2: {k:12s} dAUC {v['delong']['delta']:+.3f} DeLong p={v['delong']['p']:.3f} "
              f"(Holm {v['delong'].get('p_holm', float('nan')):.3f}) McNemar p={v['mcnemar']['p']:.3f}")
    for k, v in res["cv"].get("conditions", {}).items():
        m = v["pooled_oof_seed_ensemble"]
        print(f"CV {k:10s} seeds={v['n_seeds']} acc {m['accuracy']:.3f} bal {m['balanced_accuracy']:.3f} AUC {m['roc_auc']:.3f}")
