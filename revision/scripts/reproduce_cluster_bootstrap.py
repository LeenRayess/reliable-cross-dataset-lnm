"""Re-implementation of the manuscript's slide-cluster bootstrap (the original code was not kept).

Method as described in the submitted manuscript: for each condition and model, 2,000 replicates; each replicate samples
test slides with replacement (PCam: whole-slide image id from the official test metadata, 129 slides; CAMELYON17-WILDS:
WILDS slide id, 10 slides) and includes all retained patches of each sampled slide; metrics from temperature-calibrated
probabilities, threshold 0.5; 95% percentile intervals. The C4 UNI2-h − Virchow2 contrast uses the same resampled slides
for both models.

Applied to the ORIGINAL per-patch predictions (experiments/.../test_predictions.npz). Because the original random seed is
unknown, the procedure is repeated with 5 bootstrap seeds to show Monte-Carlo variation of the interval end points.

Output: revision/results/cluster_bootstrap_reproduction/{intervals.csv, c4_contrast.csv, comparison_with_manuscript.csv}
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_summarize import boot_metrics  # noqa: E402

OUT = ROOT / "revision" / "results" / "cluster_bootstrap_reproduction"
CL = ROOT / "revision" / "results" / "test_clusters"
ORIG = {"virchow2": ROOT / "experiments/virchow_colab/evals_cross_domain",
        "uni2h": ROOT / "experiments/uni2h_colab/evals_cross_domain_uni2h"}
FOLDER = {"C1": ("pcam_trained_on_pcam_test", "pcam"), "C2": ("pcam_trained_on_cam17_test", "cam17"),
          "C3": ("cam17_trained_on_cam17_test", "cam17"), "C4": ("cam17_trained_on_pcam_test", "pcam")}
B = 2000
BOOT_SEEDS = [0, 1, 2, 3, 42]
METRICS = ["roc_auc", "accuracy", "recall", "f1", "brier", "log_loss", "ece15"]

# Values printed in the submitted manuscript (Table III and C4 paired-contrast text)
MANUSCRIPT = {
    ("C1", "virchow2"): {"roc_auc": (0.9747, 0.9875), "recall": (0.8533, 0.9277), "f1": (0.9071, 0.9480), "ece15": (0.0120, 0.0420)},
    ("C1", "uni2h"): {"roc_auc": (0.9798, 0.9907), "recall": (0.8807, 0.9447), "f1": (0.9263, 0.9604), "ece15": (0.0142, 0.0362)},
    ("C2", "virchow2"): {"roc_auc": (0.9948, 0.9977), "recall": (0.9467, 0.9899), "f1": (0.9394, 0.9915), "ece15": (0.0060, 0.0266)},
    ("C2", "uni2h"): {"roc_auc": (0.9975, 0.9986), "recall": (0.9761, 0.9940), "f1": (0.9436, 0.9923), "ece15": (0.0239, 0.0698)},
    ("C3", "virchow2"): {"roc_auc": (0.9918, 0.9986), "recall": (0.9410, 0.9930), "f1": (0.9225, 0.9900), "ece15": (0.0122, 0.0472)},
    ("C3", "uni2h"): {"roc_auc": (0.9925, 0.9990), "recall": (0.9169, 0.9907), "f1": (0.9511, 0.9935), "ece15": (0.0068, 0.0381)},
    ("C4", "virchow2"): {"roc_auc": (0.9340, 0.9650), "recall": (0.5458, 0.7135), "f1": (0.7022, 0.8290), "ece15": (0.1128, 0.2035)},
    ("C4", "uni2h"): {"roc_auc": (0.9739, 0.9875), "recall": (0.7222, 0.8478), "f1": (0.8360, 0.9144), "ece15": (0.0705, 0.1363)},
    ("C4", "uni2h_minus_virchow2"): {"roc_auc": (0.0201, 0.0427), "accuracy": (0.0531, 0.0937), "recall": (0.1147, 0.1948),
                                     "f1": (0.0760, 0.1432), "ece15": (-0.0745, -0.0372)},
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    clusters = {"pcam": pd.read_csv(CL / "pcam_test_clusters.csv"), "cam17": pd.read_csv(CL / "cam17_test_clusters.csv")}
    rows = []
    for cond, (folder, test) in FOLDER.items():
        c = clusters[test]
        groups = [np.flatnonzero(c.slide.to_numpy() == s) for s in c.slide.unique()]
        P, y = {}, None
        for m in ORIG:
            z = np.load(ORIG[m] / folder / "test_predictions.npz")
            yy = z["y_true"].astype(float)
            assert np.array_equal(yy, c.label.to_numpy(float)), "label order mismatch"
            y, P[m] = yy, z["prob_after_temperature"].astype(float)
        point = {m: boot_metrics(y, P[m]) for m in P}
        point["uni2h_minus_virchow2"] = {k: point["uni2h"][k] - point["virchow2"][k] for k in METRICS}
        for bs in BOOT_SEEDS:
            rng = np.random.default_rng(bs)
            draws = {k: [] for k in point}
            for _ in range(B):
                idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                mu, mv = boot_metrics(y[idx], P["uni2h"][idx]), boot_metrics(y[idx], P["virchow2"][idx])
                draws["uni2h"].append(mu); draws["virchow2"].append(mv)
                draws["uni2h_minus_virchow2"].append({k: mu[k] - mv[k] for k in METRICS})
            for who, d in draws.items():
                df = pd.DataFrame(d)
                for k in METRICS:
                    rows.append({"condition": cond, "quantity": who, "metric": k, "boot_seed": bs, "point": point[who][k],
                                 "ci_low": np.percentile(df[k], 2.5), "ci_high": np.percentile(df[k], 97.5)})
        print("done", cond, flush=True)
    iv = pd.DataFrame(rows)
    iv.to_csv(OUT / "intervals.csv", index=False)

    comp = []
    for (cond, who), mets in MANUSCRIPT.items():
        for k, (lo, hi) in mets.items():
            d = iv[(iv.condition == cond) & (iv.quantity == who) & (iv.metric == k)]
            comp.append({"condition": cond, "quantity": who, "metric": k, "point": d.point.iloc[0],
                         "manuscript_low": lo, "manuscript_high": hi,
                         "reproduced_low_min": d.ci_low.min(), "reproduced_low_max": d.ci_low.max(),
                         "reproduced_high_min": d.ci_high.min(), "reproduced_high_max": d.ci_high.max(),
                         "max_abs_endpoint_diff": max(np.abs(d.ci_low - lo).max(), np.abs(d.ci_high - hi).max())})
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / "comparison_with_manuscript.csv", index=False)
    pd.set_option("display.width", 220)
    print(comp.round(4).to_string())


if __name__ == "__main__":
    main()
