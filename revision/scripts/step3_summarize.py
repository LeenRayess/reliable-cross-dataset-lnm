"""Step 3 analysis: seed variability, original-run position, UNI2-h vs Virchow2 ranking stability,
two-level (slide-cluster x training-seed) bootstrap intervals, checkpoint-rule sensitivity.

Inputs : revision/results/step3/runs.csv, data/revision_runs/step3/*.npz|json, revision/results/test_clusters/*.csv,
         original prediction files (experiments/...)
Outputs: revision/results/step3/{seed_summary.csv, original_vs_seeds.csv, ranking.csv, two_level_bootstrap.csv,
         checkpoint_rule_sensitivity.csv, summary.json}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_common import all_metrics, ece15, fit_temperature_binary, sigmoid  # noqa: E402

RES = ROOT / "revision" / "results" / "step3"
RUNS = ROOT / "data" / "revision_runs" / "step3"
CL = ROOT / "revision" / "results" / "test_clusters"
COND = {"C1": ("pcam", "pcam"), "C2": ("pcam", "cam17"), "C3": ("cam17", "cam17"), "C4": ("cam17", "pcam")}
ORIG = {"virchow2": ROOT / "experiments/virchow_colab/evals_cross_domain",
        "uni2h": ROOT / "experiments/uni2h_colab/evals_cross_domain_uni2h"}
ORIG_FOLDER = {"C1": "pcam_trained_on_pcam_test", "C2": "pcam_trained_on_cam17_test",
               "C3": "cam17_trained_on_cam17_test", "C4": "cam17_trained_on_pcam_test"}
METRICS = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
           "brier", "log_loss", "ece15", "sens_at_spec95", "sens_at_spec98", "spec_at_sens90"]
LOWER_BETTER = {"brier", "log_loss", "ece15"}
BOOT_METRICS = ["roc_auc", "accuracy", "recall", "f1", "brier", "log_loss", "ece15"]
B = 2000


def fast_auc(y, s):
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s))
    # average ranks for ties
    _, inv, cnt = np.unique(s_sorted, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    avg = cum - (cnt - 1) / 2.0
    ranks[order] = avg[inv]
    npos = y.sum(); nneg = len(y) - npos
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def boot_metrics(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pred = p >= 0.5
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"roc_auc": fast_auc(y, p), "accuracy": float(np.mean(pred == (y == 1))), "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "brier": float(np.mean((p - y) ** 2)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), "ece15": ece15(y, p)}


def calibrated_probs(model, arm, train, seed, test):
    name = "{}_{}_{}_seed{}".format(model, arm, train, seed)
    h = json.loads((RUNS / (name + "_history.json")).read_text())
    z = np.load(RUNS / (name + ".npz"))["test_logits_" + test][h["best_epoch"] - 1].astype(np.float64)
    return sigmoid(z / h["temperature"])


def main():
    runs = pd.read_csv(RES / "runs.csv")
    global ARMS
    ARMS = [a for a in ["harmonized", "original", "dropout02"] if a in set(runs.arm)]
    seeds = sorted(runs.seed.unique())
    out = {"n_seeds": len(seeds), "seeds": [int(s) for s in seeds]}

    # 1. seed summary
    g = runs.groupby(["arm", "model", "condition"])
    rows = []
    for (arm, model, cond), d in g:
        for m in METRICS + ["best_epoch", "temperature", "val_accuracy"]:
            v = d[m].to_numpy(float)
            rows.append({"arm": arm, "model": model, "condition": cond, "metric": m, "mean": v.mean(), "sd": v.std(ddof=1),
                         "min": v.min(), "max": v.max(), "n": len(v)})
    seed_summary = pd.DataFrame(rows)
    seed_summary.to_csv(RES / "seed_summary.csv", index=False)

    # 2. original single run vs seed distribution (same configuration = 'original' arm)
    rows = []
    for model in ["virchow2", "uni2h"]:
        for cond, folder in ORIG_FOLDER.items():
            z = np.load(ORIG[model] / folder / "test_predictions.npz")
            mo = all_metrics(z["y_true"], z["prob_after_temperature"])
            for arm in ["original", "harmonized"]:
                d = runs[(runs.arm == arm) & (runs.model == model) & (runs.condition == cond)]
                for m in METRICS:
                    v = d[m].to_numpy(float)
                    rows.append({"model": model, "condition": cond, "arm": arm, "metric": m, "original_run": mo[m],
                                 "seed_mean": v.mean(), "seed_sd": v.std(ddof=1), "seed_min": v.min(), "seed_max": v.max(),
                                 "z_original_vs_seeds": (mo[m] - v.mean()) / v.std(ddof=1) if v.std(ddof=1) > 0 else np.nan,
                                 "original_within_seed_range": bool(v.min() <= mo[m] <= v.max())})
    pd.DataFrame(rows).to_csv(RES / "original_vs_seeds.csv", index=False)

    # 3. ranking stability (all 10 x 10 seed pairs)
    rows = []
    for arm in ARMS:
        for cond in COND:
            for m in METRICS:
                u = runs[(runs.arm == arm) & (runs.model == "uni2h") & (runs.condition == cond)][m].to_numpy(float)
                v = runs[(runs.arm == arm) & (runs.model == "virchow2") & (runs.condition == cond)][m].to_numpy(float)
                diff = u[:, None] - v[None, :]
                better = (diff < 0) if m in LOWER_BETTER else (diff > 0)
                rows.append({"arm": arm, "condition": cond, "metric": m, "uni2h_mean": u.mean(), "uni2h_sd": u.std(ddof=1),
                             "virchow2_mean": v.mean(), "virchow2_sd": v.std(ddof=1), "mean_diff_uni2h_minus_virchow2": u.mean() - v.mean(),
                             "frac_seed_pairs_uni2h_better": float(better.mean()), "n_pairs": int(better.size)})
    pd.DataFrame(rows).to_csv(RES / "ranking.csv", index=False)

    # 4. two-level bootstrap: slides resampled (shared by both models) x one random training seed per model
    rng = np.random.default_rng(20261114)
    clusters = {"pcam": pd.read_csv(CL / "pcam_test_clusters.csv"), "cam17": pd.read_csv(CL / "cam17_test_clusters.csv")}
    rows = []
    for arm in ARMS:
        for cond, (train, test) in COND.items():
            c = clusters[test]
            y = c.label.to_numpy(float)
            groups = [np.flatnonzero(c.slide.to_numpy() == s) for s in c.slide.unique()]
            P = {mdl: np.stack([calibrated_probs(mdl, arm, train, s, test) for s in seeds]) for mdl in ["uni2h", "virchow2"]}
            draws = {k: [] for k in ["uni2h", "virchow2", "diff"]}
            for b in range(B):
                idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                su, sv = rng.integers(0, len(seeds)), rng.integers(0, len(seeds))
                mu, mv = boot_metrics(y[idx], P["uni2h"][su, idx]), boot_metrics(y[idx], P["virchow2"][sv, idx])
                draws["uni2h"].append(mu); draws["virchow2"].append(mv)
                draws["diff"].append({k: mu[k] - mv[k] for k in BOOT_METRICS})
            for who in draws:
                df = pd.DataFrame(draws[who])
                for m in BOOT_METRICS:
                    v = df[m].to_numpy()
                    rows.append({"arm": arm, "condition": cond, "quantity": who if who != "diff" else "uni2h_minus_virchow2",
                                 "metric": m, "boot_mean": v.mean(), "ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5),
                                 "excludes_zero": bool(who == "diff" and (np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0))})
            print("bootstrap done", arm, cond, flush=True)
    pd.DataFrame(rows).to_csv(RES / "two_level_bootstrap.csv", index=False)

    # 5. checkpoint-rule sensitivity (harmonized arm): val accuracy (primary) vs val ROC-AUC vs val loss vs last epoch
    rows = []
    for model in ["virchow2", "uni2h"]:
        for train in ["pcam", "cam17"]:
            yva = np.load(RUNS.parent.parent / "revision_embeddings" / model / ("pcam_macenko" if train == "pcam" else "cam17_macenko") / "valid" / "y.npy").astype(float)
            for s in seeds:
                name = "{}_harmonized_{}_seed{}".format(model, train, s)
                h = json.loads((RUNS / (name + "_history.json")).read_text())["history"]
                npz = np.load(RUNS / (name + ".npz"))
                rules = {"val_accuracy": max(range(len(h)), key=lambda e: (h[e]["val_accuracy"], -e)),
                         "val_roc_auc": max(range(len(h)), key=lambda e: (h[e]["val_roc_auc"], -e)),
                         "val_loss": min(range(len(h)), key=lambda e: (h[e]["val_loss"], e)),
                         "last_epoch": len(h) - 1}
                for rule, e in rules.items():
                    T = fit_temperature_binary(npz["val_logits"][e].astype(float), yva, grid=60)
                    for test in ["pcam", "cam17"]:
                        cond = [k for k, v in COND.items() if v == (train, test)][0]
                        yt = clusters[test].label.to_numpy(float)
                        mm = all_metrics(yt, sigmoid(npz["test_logits_" + test][e].astype(float) / T), operating_points=False)
                        rows.append({"model": model, "condition": cond, "seed": s, "rule": rule, "epoch": e + 1, "T": T,
                                     **{k: mm[k] for k in ["roc_auc", "accuracy", "recall", "f1", "ece15"]}})
    ck = pd.DataFrame(rows)
    ck.to_csv(RES / "checkpoint_rule_sensitivity.csv", index=False)
    out["checkpoint_rule_means"] = ck.groupby(["model", "condition", "rule"])[["roc_auc", "recall", "f1", "ece15"]].mean().round(4).reset_index().to_dict("records")
    (RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
