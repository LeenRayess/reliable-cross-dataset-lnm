"""Step 2b: are embeddings reproducible across GPUs? Compares UNI2-h embeddings extracted on a Colab T4
(archived partial run) with the L4 embeddings used in all experiments, for the five splits both runs finished.

Output: revision/results/step2/cross_gpu_t4_vs_l4.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emb_io import load_split  # noqa: E402

L4 = ROOT / "data" / "revision_embeddings"
T4 = Path(r"G:\My Drive\GP_ECG_REVISION\embeddings_t4_archive")
HEADS = ROOT / "revision" / "results" / "step2" / "original_heads"


def main():
    res = {}
    head = {t: np.load(HEADS / "uni2h_{}_trained.npz".format(t)) for t in ["pcam", "cam17"]}
    for tag in ["pcam_macenko", "cam17_macenko"]:
        for split in ["train", "valid", "test"]:
            try:
                a, ya = load_split(T4, "uni2h", tag, split)
            except FileNotFoundError:
                continue
            b, yb = load_split(L4, "uni2h", tag, split)
            assert np.array_equal(ya, yb)
            cos = np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
            r = {"n": int(len(a)), "min_cosine": float(cos.min()), "mean_cosine": float(cos.mean()),
                 "max_abs_diff": float(np.abs(a - b).max()),
                 "relative_l2_diff_mean": float(np.mean(np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)))}
            for t, h in head.items():
                za, zb = a @ h["weight"] + h["bias"], b @ h["weight"] + h["bias"]
                r["same_decision_{}_head".format(t)] = float(np.mean((za > 0) == (zb > 0)))
                r["n_changed_{}_head".format(t)] = int(np.sum((za > 0) != (zb > 0)))
            res["{}/{}".format(tag, split)] = r
            print(tag, split, r)
    out = ROOT / "revision" / "results" / "step2" / "cross_gpu_t4_vs_l4.json"
    out.write_text(json.dumps(res, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
