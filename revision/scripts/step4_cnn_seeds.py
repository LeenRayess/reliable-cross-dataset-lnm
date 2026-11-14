"""Step 4: repeated-seed shallow-CNN baseline (raw vs preprocessed), PCam and CAMELYON17-WILDS.

Architecture, optimiser, loss, batch size (320), lr (1e-3), epochs (10), checkpoint rule (first epoch with the best
source-validation ROC-AUC on raw sigmoid), and temperature fit (80-point grid on source validation) are identical to the locked
baseline (scripts/cnn_baseline_full_pipeline.py). Dropout 0.5 is active during training there and here (model.train()).

Deliberate change: value scaling is decided by the stored data type (uint8 -> /255; float -> used as stored, clipped to [0,1]),
instead of the original per-patch heuristic "divide by 255 only if the patch maximum > 1.5". The two rules differ only for uint8
patches whose maximum value is <= 1; these are counted for every split while loading (scaling_audit in result.json).

Speed: the training and validation splits of one arm are loaded into RAM once and reused for all seeds (values kept exactly as
stored). Mini-batch order uses torch.randperm with a seeded generator, i.e. the same mechanism as DataLoader(shuffle=True).
Test surfaces are streamed from HDF5.

Saves per (arm, seed): history.json, weights_best_val_auc.pt, result.json, test logits (in-domain and external) of the selected
checkpoint. Raw arms are scored on the full raw test sets; restriction to the quality-passed indices happens in step4_summarize.py.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_common import all_metrics, fit_temperature_binary, sigmoid  # noqa: E402

ARMS = ["pcam_raw", "pcam_preprocessed", "cam17_raw", "cam17_preprocessed"]
EXTERNAL = {"pcam_raw": "cam17_raw", "pcam_preprocessed": "cam17_preprocessed",
            "cam17_raw": "pcam_raw", "cam17_preprocessed": "pcam_preprocessed"}


def surfaces(data_root: Path | None):
    """data_root=None -> this repository's layout; otherwise <data_root>/<surface>/<split>_{x,y}.h5 (Colab copy)."""
    if data_root is not None:
        return {s: {sp: (data_root / s / f"{sp}_x.h5", data_root / s / f"{sp}_y.h5") for sp in ["train", "valid", "test"]}
                for s in ARMS}
    pc, pcp = ROOT / "pcam_data", ROOT / "pcam_data/preprocessed_macenko_benchmark_style"
    cw = ROOT / "data/wilds/camelyon17_h5_full_oodval"
    cwp = cw / "preprocessed_macenko_benchmark_style"
    raw = {"train": "training/camelyonpatch_level_2_split_train", "valid": "val/camelyonpatch_level_2_split_valid",
           "test": "test/camelyonpatch_level_2_split_test"}
    return {"pcam_raw": {sp: (pc / f"{raw[sp]}_x.h5", pc / f"{raw[sp]}_y.h5") for sp in raw},
            "pcam_preprocessed": {sp: (pcp / f"{sp}_x.h5", pcp / f"{sp}_y.h5") for sp in raw},
            "cam17_raw": {sp: (cw / f"{sp}_x.h5", cw / f"{sp}_y.h5") for sp in raw},
            "cam17_preprocessed": {sp: (cwp / f"{sp}_x.h5", cwp / f"{sp}_y.h5") for sp in raw}}


class BaselineCNN(nn.Module):
    """Identical to scripts/cnn_baseline_full_pipeline.BaselineCNN (same layers, same order)."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.features(x))


def to_model_input(xb: torch.Tensor) -> torch.Tensor:
    """NHWC stored values -> NCHW float in [0,1]; scaling decided by dtype."""
    x = xb.float() / 255.0 if xb.dtype == torch.uint8 else xb.float()
    return x.clamp_(0.0, 1.0).permute(0, 3, 1, 2).contiguous()


def read_y(yp: Path) -> np.ndarray:
    import h5py
    with h5py.File(str(yp), "r") as f:
        return np.asarray(f["y"][:]).reshape(-1).astype(np.float32)


def load_x(xp: Path, chunk: int = 8192):
    """Read the whole x array into RAM in its stored dtype; audit value range per patch."""
    import h5py
    with h5py.File(str(xp), "r") as f:
        d = f["x"]
        need = int(np.prod(d.shape)) * d.dtype.itemsize
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except ImportError:
            avail = need * 2
        if need < 0.8 * avail:
            X, mode = np.empty(d.shape, dtype=d.dtype), "ram"
        else:  # not enough RAM: decompress once to an uncompressed .npy on local disk and memory-map it (same values)
            X, mode = np.lib.format.open_memmap(str(Path(xp).with_suffix(".decompressed.npy")), "w+", d.dtype, d.shape), "memmap"
        print(f"loading {xp} ({need / 1e9:.1f} GB, available RAM {avail / 1e9:.1f} GB) -> {mode}", flush=True)
        audit = {"dtype": str(d.dtype), "n": int(d.shape[0]), "uint8_patch_max_le_1": 0, "float_patch_max_gt_1p5": 0,
                 "float_values_outside_0_1_patches": 0, "global_min": None, "global_max": None,
                 "held_in": None}
        gmin, gmax = np.inf, -np.inf
        for i in range(0, d.shape[0], chunk):
            a = d[i:i + chunk]
            X[i:i + len(a)] = a
            pm = a.reshape(len(a), -1).max(1); pn = a.reshape(len(a), -1).min(1)
            gmin, gmax = min(gmin, float(pn.min())), max(gmax, float(pm.max()))
            if d.dtype == np.uint8:
                audit["uint8_patch_max_le_1"] += int((pm <= 1).sum())
            else:
                audit["float_patch_max_gt_1p5"] += int((pm > 1.5).sum())
                audit["float_values_outside_0_1_patches"] += int(((pm > 1) | (pn < 0)).sum())
        audit["global_min"], audit["global_max"], audit["held_in"] = gmin, gmax, mode
    return X, audit


class H5Stream(Dataset):
    def __init__(self, xp: Path, yp: Path):
        self.xp, self.y = str(xp), read_y(yp)
        self._f = None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self._f is None:
            import h5py
            self._f = h5py.File(self.xp, "r")
        return torch.from_numpy(np.asarray(self._f["x"][i])), torch.tensor(self.y[i])


def set_seeds(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def logits_ram(model, X, device, bs=1024):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(model(to_model_input(torch.from_numpy(X[i:i + bs]).to(device))).squeeze(-1).float().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


@torch.no_grad()
def logits_stream(model, xp, yp, device, workers, bs=1024):
    ds = H5Stream(xp, yp)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, pin_memory=True)
    model.eval()
    out = [model(to_model_input(x.to(device, non_blocking=True))).squeeze(-1).float().cpu().numpy() for x, _ in dl]
    return np.concatenate(out).astype(np.float64), ds.y.astype(np.float64)


def bce(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_seed(arm, seed, Xtr, ytr, Xva, yva, S, a, device, out_runs: Path, out_res: Path, audit):
    from sklearn.metrics import roc_auc_score
    tag = f"{arm}_seed{seed}"
    od = out_runs / tag
    if (od / "done.json").is_file():
        print("skip (done)", tag, flush=True)
        return
    od.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)
    model = BaselineCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    crit = nn.BCEWithLogitsLoss()
    g = torch.Generator(); g.manual_seed(seed)
    ytr_t = torch.from_numpy(ytr)
    hist, best_auc, best_ep, t0 = [], -1.0, None, time.time()
    for ep in range(a.epochs):
        te = time.time(); model.train()
        perm = torch.randperm(len(ytr), generator=g).numpy()
        tot = 0.0; corr = 0
        nb = len(perm) if a.max_batches is None else min(len(perm), a.max_batches * a.batch_size)
        for i in range(0, nb, a.batch_size):
            idx = perm[i:i + a.batch_size]
            x = to_model_input(torch.from_numpy(Xtr[idx]).to(device, non_blocking=True))
            y = ytr_t[idx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            z = model(x).squeeze(-1)
            loss = crit(z, y); loss.backward(); opt.step()
            tot += loss.item() * len(idx); corr += int(((z >= 0) == (y > 0.5)).sum())
        t_tr = time.time() - te
        vz = logits_ram(model, Xva, device); vp = sigmoid(vz)
        rec = {"epoch": ep + 1, "train_loss": tot / nb, "train_accuracy": corr / nb, "val_roc_auc": float(roc_auc_score(yva, vp)),
               "val_loss": bce(yva, vp), "val_accuracy": float(np.mean((vp >= 0.5) == (yva > 0.5))),
               "train_seconds": t_tr, "epoch_seconds": time.time() - te}
        if a.max_batches is not None:
            print(f"TIMING {tag}: {nb} patches in {t_tr:.1f}s = {nb / t_tr:.0f}/s -> full epoch ~{len(ytr) / (nb / t_tr) / 60:.2f} min;"
                  f" validation {time.time() - te - t_tr:.1f}s", flush=True)
            return
        hist.append(rec)
        if rec["val_roc_auc"] > best_auc:  # strict '>' = first best epoch, as in the locked script
            best_auc, best_ep = rec["val_roc_auc"], ep + 1
            torch.save(model.state_dict(), od / "weights_best_val_auc.pt")
        print(tag, {k: round(v, 4) for k, v in rec.items()}, flush=True)
        (od / "history.json").write_text(json.dumps(hist, indent=1))

    if best_ep is None:  # validation ROC-AUC undefined in every epoch (only possible on toy data)
        best_ep = a.epochs
        torch.save(model.state_dict(), od / "weights_best_val_auc.pt")
    model.load_state_dict(torch.load(od / "weights_best_val_auc.pt", map_location=device))
    vz = logits_ram(model, Xva, device)
    T = float(fit_temperature_binary(vz, yva.astype(np.float64), grid=80))
    np.save(od / "val_logits_best.npy", vz.astype(np.float32))
    rows = []
    for which, surf in [("in_domain", arm), ("external", EXTERNAL[arm])]:
        z, y = logits_stream(model, *S[surf]["test"], device, a.workers)
        np.save(od / f"test_logits_{surf}.npy", z.astype(np.float32))
        rows.append({"arm": arm, "seed": seed, "test": which, "surface": surf, "n": len(y), "best_epoch": best_ep,
                     "temperature": T, **all_metrics(y, sigmoid(z / T))})
    res = {"arm": arm, "seed": seed, "best_epoch": best_ep, "best_val_auc": best_auc, "temperature": T,
           "train_seconds_total": time.time() - t0, "n_train": len(ytr), "n_valid": len(yva),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
           "scaling_audit": audit, "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
           "history": hist, "test_metrics": rows}
    (od / "result.json").write_text(json.dumps(res, indent=1, default=float))
    import pandas as pd
    out_res.mkdir(parents=True, exist_ok=True)
    csv = out_res / "runs.csv"
    df = pd.DataFrame(rows)
    if csv.is_file():
        old = pd.read_csv(csv)
        df = pd.concat([old[~((old.arm == arm) & (old.seed == seed))], df], ignore_index=True)
    df.sort_values(["arm", "seed", "test"]).to_csv(csv, index=False)
    (od / "done.json").write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print("DONE", tag, f"best_ep={best_ep} T={T:.3f}", {r["test"]: round(r["roc_auc"], 4) for r in rows}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=320)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-runs", type=Path, default=ROOT / "data/revision_runs/step4")
    ap.add_argument("--out-res", type=Path, default=ROOT / "revision/results/step4")
    ap.add_argument("--max-batches", type=int, default=None, help="timing mode")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = surfaces(a.data_root)
    for arm in a.arms:
        todo = [s for s in a.seeds if not (a.out_runs / f"{arm}_seed{s}" / "done.json").is_file()] if a.max_batches is None else a.seeds[:1]
        if not todo:
            print("arm complete:", arm, flush=True)
            continue
        t = time.time()
        Xtr, au_tr = load_x(S[arm]["train"][0]); ytr = read_y(S[arm]["train"][1])
        Xva, au_va = load_x(S[arm]["valid"][0]); yva = read_y(S[arm]["valid"][1])
        assert len(Xtr) == len(ytr) and len(Xva) == len(yva)
        audit = {"train": au_tr, "valid": au_va}
        print(f"loaded {arm}: train {Xtr.shape} {Xtr.dtype}, valid {Xva.shape} in {time.time() - t:.0f}s; audit {audit}", flush=True)
        for seed in todo:
            train_seed(arm, seed, Xtr, ytr, Xva, yva, S, a, device, a.out_runs, a.out_res, audit)
        del Xtr, Xva


if __name__ == "__main__":
    main()
