"""Step 3 diagnostic: why did the original runs reach a lower training loss than all repeated-seed runs?

Compares, on the SAME cached (fp16) training embeddings, the eval-mode (no dropout) training loss and the
validation/test metrics of (a) the original Virchow2 CAMELYON17 head (selected epoch 10) and (b) heads retrained
from the cached embeddings with the original configuration (batch 320, dropout 0.2), 3 seeds.
If (a) fits the cached training embeddings much better than (b), the optimisation itself differed;
if (a) and (b) fit equally, the original run's lower loss came from its fp32 inputs.
Output: revision/results/step3/diag_trainloss.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emb_io import load_split  # noqa: E402
from metrics_common import all_metrics, fit_temperature_binary, sigmoid  # noqa: E402

EMB = ROOT / "data" / "revision_embeddings"


def bce(z, y):
    p = np.clip(sigmoid(z), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    dev = torch.device("cuda")
    Xtr, ytr = load_split(EMB, "virchow2", "cam17_macenko", "train")
    Xva, yva = load_split(EMB, "virchow2", "cam17_macenko", "valid")
    Xte, yte = load_split(EMB, "virchow2", "pcam_macenko", "test")
    o = np.load(ROOT / "revision/results/step2/original_heads/virchow2_cam17_trained.npz")
    res = {}

    def evaluate(w, b, tag):
        ztr, zva, zte = Xtr @ w + b, Xva @ w + b, Xte @ w + b
        T = fit_temperature_binary(zva, yva)
        m = all_metrics(yte, sigmoid(zte / T), operating_points=False)
        res[tag] = {"train_loss_eval_mode": bce(ztr, ytr), "train_acc": float(np.mean((ztr >= 0) == ytr)),
                    "val_loss": bce(zva, yva), "val_acc": float(np.mean((zva >= 0) == yva)), "T": T,
                    "weight_l2_norm": float(np.linalg.norm(w)),
                    "C4_roc_auc": m["roc_auc"], "C4_recall": m["recall"], "C4_ece15": m["ece15"]}
        print(tag, {k: round(v, 4) for k, v in res[tag].items()})

    evaluate(o["weight"].astype(np.float64), float(o["bias"]), "original_head_epoch10")
    Xt = torch.tensor(Xtr, dtype=torch.float16, device=dev); yt = torch.tensor(ytr, device=dev)
    for seed in range(3):
        torch.manual_seed(seed)
        lin = nn.Linear(Xtr.shape[1], 1).to(dev); drop = nn.Dropout(0.2)
        opt = torch.optim.Adam(lin.parameters(), lr=1e-3); lf = nn.BCEWithLogitsLoss()
        g = torch.Generator(device=dev).manual_seed(seed)
        for ep in range(10):
            perm = torch.randperm(len(Xt), generator=g, device=dev)
            for i in range(0, len(Xt), 320):
                idx = perm[i:i + 320]
                opt.zero_grad(); lf(lin(drop(Xt[idx].float())).squeeze(-1), yt[idx]).backward(); opt.step()
        evaluate(lin.weight.detach().cpu().double().numpy().reshape(-1), float(lin.bias), "retrained_seed{}_epoch10".format(seed))
    out = ROOT / "revision/results/step3/diag_trainloss.json"
    out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
