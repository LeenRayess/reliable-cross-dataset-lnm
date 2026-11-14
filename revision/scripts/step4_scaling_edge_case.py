"""Quantify the CNN value-scaling edge case (reviewer item: '/255 only if patch max > 1.5').

Original rule (scripts/cnn_baseline_full_pipeline.py): per patch, divide by 255 only if max > 1.5.
Fixed rule (step4_cnn_seeds.py): divide by 255 iff the stored dtype is uint8.
The rules disagree only for (a) uint8 patches with max <= 1 and (b) float patches with max > 1.5.
This script scans every split of every CNN surface and counts both cases.
Output: revision/results/step4/scaling_edge_case.json
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step4_cnn_seeds import ROOT, surfaces  # noqa: E402

out = {}
for surf, splits in surfaces(None).items():
    for split, (xp, _) in splits.items():
        with h5py.File(xp, "r") as f:
            d = f["x"]
            n_a = n_b = 0
            gmin, gmax = np.inf, -np.inf
            step = 4096
            for i in range(0, d.shape[0], step):
                a = d[i:i + step].reshape(min(step, d.shape[0] - i), -1)
                pm, pn = a.max(1), a.min(1)
                gmin, gmax = min(gmin, float(pn.min())), max(gmax, float(pm.max()))
                if d.dtype == np.uint8:
                    n_a += int((pm <= 1).sum())
                elif d.dtype.kind == "f":
                    n_b += int((pm > 1.5).sum())
            out[f"{surf}/{split}"] = {"dtype": str(d.dtype), "n": int(d.shape[0]), "uint8_patches_max_le_1": n_a,
                                      "float_patches_max_gt_1p5": n_b, "global_min": gmin, "global_max": gmax,
                                      "patches_where_rules_differ": n_a + n_b}
        print(surf, split, out[f"{surf}/{split}"], flush=True)
p = ROOT / "revision/results/step4/scaling_edge_case.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=1))
print("wrote", p)
