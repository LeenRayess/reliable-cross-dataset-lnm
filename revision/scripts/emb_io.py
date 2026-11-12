"""Load cached embeddings written by extract_embeddings.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_split(root, model: str, dataset_tag: str, split: str, mmap: bool = False):
    d = Path(root) / model / dataset_tag / split
    if not (d / "done.json").exists():
        raise FileNotFoundError("extraction not finished: {}".format(d))
    meta = json.loads((d / "done.json").read_text())
    chunks = sorted(d.glob("chunk_[0-9][0-9][0-9][0-9][0-9].npy"))
    if len(chunks) != meta["n_chunks"]:
        raise RuntimeError("{}: expected {} chunks, found {}".format(d, meta["n_chunks"], len(chunks)))
    X = np.concatenate([np.load(c, mmap_mode="r" if mmap else None) for c in chunks], axis=0)
    y = np.load(d / "y.npy").astype(np.float32).reshape(-1)
    if len(X) != len(y):
        raise RuntimeError("{}: {} embeddings vs {} labels".format(d, len(X), len(y)))
    return X, y
