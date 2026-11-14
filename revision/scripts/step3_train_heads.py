"""Step 3: repeated-seed linear-head training on cached embeddings (Virchow2, UNI2-h; PCam- and CAMELYON17-trained).

Two configuration arms, each with N seeds:
  harmonized : identical for every model and training set — batch 256, head dropout 0, Adam lr 1e-3, weight decay 0,
               10 epochs, BCEWithLogits, checkpoint = epoch with highest source-validation accuracy,
               temperature fitted on source validation (grid 60, T in [0.2, 5]).
  original   : effective configuration of the submitted runs: as harmonized, except Virchow2 CAMELYON17 batch 320
               (the listed head dropout of 0.2 was never active in the original code; see ORIGINAL below).
  dropout02  : sensitivity arm with the listed dropout values actually applied (0.2 except UNI2-h CAMELYON17).
The seed controls head initialisation, dropout masks and the training-batch order.

Per run: per-epoch validation curve, selected epoch, temperature, metrics on both test sets (C1–C4),
and the test logits of every epoch (for checkpoint-rule sensitivity) -> data/revision_runs/step3/ (not in git).
Run-level metric rows -> revision/results/step3/runs.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emb_io import load_split  # noqa: E402
from metrics_common import all_metrics, fit_temperature_binary, sigmoid  # noqa: E402

EMB = ROOT / "data" / "revision_embeddings"
RUNS = ROOT / "data" / "revision_runs" / "step3"
RES = ROOT / "revision" / "results" / "step3"
TAGS = {"pcam": "pcam_macenko", "cam17": "cam17_macenko"}
COND = {("pcam", "pcam"): "C1", ("pcam", "cam17"): "C2", ("cam17", "cam17"): "C3", ("cam17", "pcam"): "C4"}

# Effective configuration of the submitted runs. The original scripts listed head dropout 0.2 for some runs, but their
# forward() always switched the dropout layer to eval mode, so dropout was never applied during training (verified:
# no-dropout retraining reproduces the original training-loss curves). Only the batch size differed.
ORIGINAL = {("virchow2", "pcam"): (256, 0.0), ("uni2h", "pcam"): (256, 0.0),
            ("virchow2", "cam17"): (320, 0.0), ("uni2h", "cam17"): (256, 0.0)}
# Sensitivity arm: the dropout values written in the original configs, actually applied.
DROPOUT02 = {("virchow2", "pcam"): (256, 0.2), ("uni2h", "pcam"): (256, 0.2),
             ("virchow2", "cam17"): (320, 0.2), ("uni2h", "cam17"): (256, 0.0)}
HARMONIZED = (256, 0.0)
EPOCHS, LR = 10, 1e-3


class Head(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.drop = nn.Dropout(p) if p > 0 else nn.Identity()
        self.lin = nn.Linear(d, 1)

    def forward(self, x):
        return self.lin(self.drop(x)).squeeze(-1)


@torch.no_grad()
def logits_of(head, X, bs=8192):
    head.eval()
    return torch.cat([head(X[i:i + bs].float()) for i in range(0, len(X), bs)]).double().cpu().numpy()


def train_one(Xtr, ytr, Xva, yva, tests, bs, p, seed, dev):
    torch.manual_seed(seed)
    np.random.seed(seed)
    head = Head(Xtr.shape[1], p).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=0.0)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator(device=dev).manual_seed(seed)
    hist, test_logits, val_logits_all = [], [], []
    for ep in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(len(Xtr), generator=g, device=dev)
        tl, nb = 0.0, 0
        for i in range(0, len(Xtr), bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            loss = lossf(head(Xtr[idx].float()), ytr[idx])
            loss.backward()
            opt.step()
            tl += float(loss); nb += 1
        zv = logits_of(head, Xva)
        pv = np.clip(sigmoid(zv), 1e-7, 1 - 1e-7)
        hist.append({"epoch": ep, "train_loss": tl / nb, "val_accuracy": float(np.mean((zv >= 0) == yva)),
                     "val_loss": float(-np.mean(yva * np.log(pv) + (1 - yva) * np.log(1 - pv))),
                     "val_roc_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yva, zv))})
        val_logits_all.append(zv.astype(np.float32))
        test_logits.append({k: logits_of(head, X).astype(np.float32) for k, X in tests.items()})
    # original rule: first epoch with the highest validation accuracy (strict improvement, as in the original loop)
    best = max(range(EPOCHS), key=lambda e: (hist[e]["val_accuracy"], -e))
    return hist, best, val_logits_all, test_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--arms", nargs="+", default=["harmonized", "original"])
    ap.add_argument("--models", nargs="+", default=["virchow2", "uni2h"])
    ap.add_argument("--train-sets", nargs="+", default=["pcam", "cam17"])
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RUNS.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    rows_path = RES / "runs.csv"
    rows = pd.read_csv(rows_path).to_dict("records") if rows_path.exists() else []
    done = {(r["model"], r["arm"], r["train"], int(r["seed"])) for r in rows}

    for model in args.models:
        tests_np = {t: load_split(EMB, model, TAGS[t], "test") for t in ["pcam", "cam17"]}
        tests = {t: torch.tensor(X, dtype=torch.float16, device=dev) for t, (X, _) in tests_np.items()}
        for train in args.train_sets:
            Xtr, ytr = load_split(EMB, model, TAGS[train], "train")
            Xva, yva = load_split(EMB, model, TAGS[train], "valid")
            Xtr_t = torch.tensor(Xtr, dtype=torch.float16, device=dev); ytr_t = torch.tensor(ytr, device=dev)
            Xva_t = torch.tensor(Xva, dtype=torch.float16, device=dev)
            del Xtr
            for arm in args.arms:
                bs, p = HARMONIZED if arm == "harmonized" else (ORIGINAL if arm == "original" else DROPOUT02)[(model, train)]
                for seed in args.seeds:
                    if (model, arm, train, seed) in done:
                        continue
                    t0 = time.time()
                    hist, best, vlog, tlog = train_one(Xtr_t, ytr_t, Xva_t, yva, tests, bs, p, seed, dev)
                    T = fit_temperature_binary(vlog[best], yva, grid=60)
                    name = "{}_{}_{}_seed{}".format(model, arm, train, seed)
                    np.savez_compressed(RUNS / (name + ".npz"), val_logits=np.stack(vlog),
                                        **{"test_logits_" + k: np.stack([e[k] for e in tlog]) for k in tests})
                    (RUNS / (name + "_history.json")).write_text(json.dumps(
                        {"history": hist, "best_epoch": best + 1, "temperature": T, "batch_size": bs, "dropout": p}, indent=1))
                    for test in ["pcam", "cam17"]:
                        z = tlog[best][test].astype(np.float64)
                        m = all_metrics(tests_np[test][1], sigmoid(z / T))
                        rows.append({"model": model, "arm": arm, "train": train, "test": test, "condition": COND[(train, test)],
                                     "seed": seed, "batch_size": bs, "dropout": p, "best_epoch": best + 1, "temperature": T,
                                     "val_accuracy": hist[best]["val_accuracy"], "val_roc_auc": hist[best]["val_roc_auc"], **m})
                    pd.DataFrame(rows).to_csv(rows_path, index=False)
                    r = [x for x in rows[-2:]]
                    print("{:<40} ep{:>2} T={:.3f}  {}  ({:.0f}s)".format(
                        name, best + 1, T, "  ".join("{} AUC {:.4f} rec {:.4f}".format(x["condition"], x["roc_auc"], x["recall"]) for x in r),
                        time.time() - t0), flush=True)
            del Xtr_t, ytr_t, Xva_t
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
