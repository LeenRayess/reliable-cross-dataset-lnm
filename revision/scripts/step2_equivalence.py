"""Step 2: do the ORIGINAL trained models reproduce their original test predictions on the cached embeddings?

For each model (Virchow2, UNI2-h) and condition C1-C4:
  * take the original trained linear head (from the original model_best.pt files),
  * apply it to the new cached test embeddings (L4, fp16),
  * compare with the original per-patch predictions (test_predictions.npz).
Also: fingerprint of the frozen encoder weights inside every original checkpoint vs the encoder used for
extraction (identifies the exact Hugging Face version), and the original heads are exported to small .npz files.

Output: revision/results/step2/equivalence.json, original_heads/*.npz
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emb_io import load_split  # noqa: E402
from fm_common import state_dict_fingerprint  # noqa: E402

EMB = ROOT / "data" / "revision_embeddings"
OUT = ROOT / "revision" / "results" / "step2"

COND = {  # id: (eval folder name, train dataset, test dataset tag)
    "C1": ("pcam_trained_on_pcam_test", "pcam", "pcam_macenko"),
    "C2": ("pcam_trained_on_cam17_test", "pcam", "cam17_macenko"),
    "C3": ("cam17_trained_on_cam17_test", "cam17", "cam17_macenko"),
    "C4": ("cam17_trained_on_pcam_test", "cam17", "pcam_macenko"),
}
EVAL_DIR = {
    "virchow2": ROOT / "experiments/virchow_colab/evals_cross_domain",
    "uni2h": ROOT / "experiments/uni2h_colab/evals_cross_domain_uni2h",
}
# every original checkpoint found on disk; which one belongs to which training set is determined from the data
CKPTS = {
    "virchow2": sorted((ROOT / "experiments/virchow_colab").rglob("model_best.pt")),
    "uni2h": sorted((ROOT / "experiments/uni2h_colab/evals_cross_domain_uni2h").glob("model_best-*.pt"))
    + sorted((ROOT / "project_outputs/04_uni2h_weights").glob("*.pt")),
}


def load_ckpt(p: Path):
    sd = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
    sd = sd.get("model_state_dict", sd)
    w = sd["head.weight"].float().numpy().reshape(-1)
    b = float(sd["head.bias"].float().reshape(-1)[0])
    return w, b, state_dict_fingerprint(sd, "backbone.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "original_heads").mkdir(exist_ok=True)
    report = {"models": {}}
    for model in ["virchow2", "uni2h"]:
        meta = json.loads((EMB / model / "extraction_meta_nvidia_l4.json").read_text())
        heads = {}
        for p in CKPTS[model]:
            w, b, fp = load_ckpt(p)
            key = "{:.6f}_{:.6f}".format(w[:4].sum(), b)  # identical files collapse to one head
            heads.setdefault(key, {"w": w, "b": b, "files": [], "encoder_hash": fp})["files"].append(
                str(p.relative_to(ROOT)).replace("\\", "/"))
        print(model, "unique original heads:", len(heads))
        tests = {t: load_split(EMB, model, t, "test") for t in ["pcam_macenko", "cam17_macenko"]}
        mres = {"encoder_hf_commit_used_for_extraction": meta["hf_commit_sha_remote_main"],
                "encoder_hash_extraction": meta["backbone_weight_sha256"], "conditions": {}, "heads": []}
        for h in heads.values():
            mres["heads"].append({"files": h["files"], "encoder_hash_in_checkpoint": h["encoder_hash"],
                                  "encoder_matches_extraction": h["encoder_hash"] == meta["backbone_weight_sha256"]})
        for cid, (folder, train, test_tag) in COND.items():
            npz = EVAL_DIR[model] / folder / "test_predictions.npz"
            if not npz.exists():
                mres["conditions"][cid] = {"error": "missing " + str(npz)}
                continue
            old = np.load(npz)
            X, y = tests[test_tag]
            assert np.array_equal(old["y_true"].astype(np.float32), y), "label order differs"
            T = float(old["temperature_T"])
            # pick the original head that reproduces this condition's logits best
            best = None
            for key, h in heads.items():
                z = X @ h["w"] + h["b"]
                err = float(np.abs(z - old["logits"]).mean())
                if best is None or err < best[0]:
                    best = (err, key, z)
            _, key, z = best
            zo = old["logits"].astype(np.float64)
            po, pn = 1 / (1 + np.exp(-zo / T)), 1 / (1 + np.exp(-z / T))
            r = {
                "n_test": int(len(y)),
                "head_files": heads[key]["files"],
                "same_decision_at_0.5": float(np.mean((pn >= 0.5) == (po >= 0.5))),
                "n_decisions_changed": int(np.sum((pn >= 0.5) != (po >= 0.5))),
                "max_abs_logit_diff": float(np.abs(z - zo).max()),
                "mean_abs_logit_diff": float(np.abs(z - zo).mean()),
                "max_abs_prob_diff_calibrated": float(np.abs(pn - po).max()),
                "roc_auc_original": float(roc_auc_score(y, zo)),
                "roc_auc_cached": float(roc_auc_score(y, z)),
                "accuracy_original": float(np.mean((po >= 0.5) == y)),
                "accuracy_cached": float(np.mean((pn >= 0.5) == y)),
                "temperature_T": T,
            }
            mres["conditions"][cid] = r
            tag = "{}_{}_trained".format(model, train)
            np.savez(OUT / "original_heads" / (tag + ".npz"), weight=heads[key]["w"], bias=heads[key]["b"],
                     temperature_T=T, source_files=np.array(heads[key]["files"]))
            print("  {} {:>6} n={:6d}  same decisions {:.5f} ({} changed)  AUC {:.4f} -> {:.4f}  acc {:.4f} -> {:.4f}  max|dlogit| {:.3f}".format(
                cid, model, r["n_test"], r["same_decision_at_0.5"], r["n_decisions_changed"], r["roc_auc_original"],
                r["roc_auc_cached"], r["accuracy_original"], r["accuracy_cached"], r["max_abs_logit_diff"]))
        report["models"][model] = mres
    (OUT / "equivalence.json").write_text(json.dumps(report, indent=2))
    print("wrote", OUT / "equivalence.json")


if __name__ == "__main__":
    main()
