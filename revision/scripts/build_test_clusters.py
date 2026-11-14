"""Slide / patient identifiers for every retained (post-QC) test patch, in the order of the preprocessed H5 files.

PCam: whole-slide image id from the official test metadata (camelyonpatch_level_2_split_test_meta.csv, column wsi).
CAMELYON17-WILDS: slide, patient and node from metadata.csv; test = center 2 rows in metadata order
(same rule as scripts/camelyon17/prepare_wilds_camelyon17_h5.py).
The mapping is verified by checking that labels agree for every patch.

Output: revision/results/test_clusters/{pcam,cam17}_test_clusters.csv
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "revision" / "results" / "test_clusters"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- PCam
    pre = ROOT / "pcam_data/preprocessed_macenko_benchmark_style"
    kept = np.load(pre / "checkpoint_step1_test.npz")["kept_indices"]
    meta = pd.read_csv(ROOT / "pcam_data/test/camelyonpatch_level_2_split_test_meta.csv", index_col=0)
    with h5py.File(ROOT / "pcam_data/test/camelyonpatch_level_2_split_test_y.h5") as f:
        y_raw = np.asarray(f["y"][:]).reshape(-1)
    with h5py.File(pre / "test_y.h5") as f:
        y_pre = np.asarray(f["y"][:]).reshape(-1)
    assert len(meta) == len(y_raw) == 32768 and len(kept) == len(y_pre)
    assert np.array_equal(y_raw[kept], y_pre), "PCam label mismatch"
    assert np.array_equal(meta["center_tumor_patch"].to_numpy()[kept].astype(int), y_pre.astype(int))
    pc = pd.DataFrame({"row": np.arange(len(kept)), "source_index": kept, "label": y_pre.astype(int),
                       "slide": meta["wsi"].to_numpy()[kept]})
    pc["patient"] = pc["slide"]  # CAMELYON16 test: one slide per patient
    pc.to_csv(OUT / "pcam_test_clusters.csv", index=False)
    print("PCam test: n={} slides={}".format(len(pc), pc.slide.nunique()))

    # ---- CAMELYON17-WILDS
    pre = ROOT / "data/wilds/camelyon17_h5_full_oodval/preprocessed_macenko_benchmark_style"
    kept = np.load(pre / "checkpoint_step1_test.npz")["kept_indices"]
    md = pd.read_csv(ROOT / "data/wilds/camelyon17_v1.0/metadata.csv", index_col=0)
    test_rows = np.flatnonzero(md["center"].to_numpy() == 2)
    with h5py.File(ROOT / "data/wilds/camelyon17_h5_full_oodval/test_y.h5") as f:
        y_raw = np.asarray(f["y"][:]).reshape(-1)
    with h5py.File(pre / "test_y.h5") as f:
        y_pre = np.asarray(f["y"][:]).reshape(-1)
    assert len(test_rows) == len(y_raw)
    assert np.array_equal(md["tumor"].to_numpy()[test_rows].astype(int), y_raw.astype(int)), "WILDS order mismatch"
    assert np.array_equal(y_raw[kept], y_pre), "CAM17 label mismatch"
    rows = test_rows[kept]
    cc = pd.DataFrame({"row": np.arange(len(kept)), "source_index": kept, "metadata_row": rows, "label": y_pre.astype(int),
                       "slide": md["slide"].to_numpy()[rows], "patient": md["patient"].to_numpy()[rows],
                       "node": md["node"].to_numpy()[rows], "center": md["center"].to_numpy()[rows]})
    cc.to_csv(OUT / "cam17_test_clusters.csv", index=False)
    print("CAM17 test: n={} slides={} patients={}".format(len(cc), cc.slide.nunique(), cc.patient.nunique()))
    print(cc.groupby(["patient", "slide"]).label.agg(["size", "mean"]))


if __name__ == "__main__":
    main()
