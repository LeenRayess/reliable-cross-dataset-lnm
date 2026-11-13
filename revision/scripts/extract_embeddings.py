"""Extract frozen foundation-model embeddings once and cache them (IEEE Access revision, Step 1).

Why: the encoder is frozen and inputs are deterministic, so every epoch of the original runs
recomputed identical embeddings. Caching them once makes repeated-seed / harmonized head
training take minutes instead of days, with an identical model.

Output layout (resumable; each chunk is written atomically):
  <out-root>/<model>/<dataset_tag>/<split>/chunk_00000.npy ...   float32 (n_chunk, D)
  <out-root>/<model>/<dataset_tag>/<split>/y.npy                 labels
  <out-root>/<model>/<dataset_tag>/<split>/done.json             rows, timing, throughput
  <out-root>/<model>/extraction_meta.json                        versions, HF commit, weight hash, GPU

Usage (Colab):
  python revision/scripts/extract_embeddings.py --model uni2h \
      --h5-dir /content/local_pcam --dataset-tag pcam_macenko \
      --out-root /content/drive/MyDrive/GP_ECG_REVISION/embeddings
  python revision/scripts/extract_embeddings.py --model uni2h --benchmark-only --out-root ...
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fm_common import MODELS, embed, load_backbone, state_dict_fingerprint, transform_batch  # noqa: E402

SPLIT_FILES = {"train": "train", "valid": "valid", "test": "test"}


class H5ChunkDataset(Dataset):
    """Item i = rows [starts[i], ends[i]) of x, already transformed to the model input."""

    def __init__(self, x_path: str, ranges):
        self.x_path = x_path
        self.ranges = ranges
        self._f = None

    def __len__(self):
        return len(self.ranges)

    def __getitem__(self, i):
        if self._f is None:
            self._f = h5py.File(self.x_path, "r")
        s, e = self.ranges[i]
        return i, transform_batch(self._f["x"][s:e])


def _hf_commit(repo: str):
    if repo is None:
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo).sha
    except Exception as exc:  # offline / gated without token
        return "unavailable: {}".format(exc)


def _cached_snapshot(repo: str):
    if repo is None:
        return None
    try:
        from huggingface_hub import constants

        snap = Path(constants.HF_HUB_CACHE) / ("models--" + repo.replace("/", "--")) / "snapshots"
        return sorted(p.name for p in snap.iterdir()) if snap.is_dir() else None
    except Exception:
        return None


def gpu_tag(device) -> str:
    name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def write_meta(args, backbone, device, out_model_dir: Path):
    import timm
    import huggingface_hub

    info = MODELS[args.model]
    meta = {
        "model": args.model,
        "hf_repo": info["hf_repo"],
        "hf_commit_sha_remote_main": _hf_commit(info["hf_repo"]),
        "hf_cached_snapshot_shas": _cached_snapshot(info["hf_repo"]),
        "backbone_weight_sha256": state_dict_fingerprint(backbone.state_dict()),
        "backbone_n_params": int(sum(p.numel() for p in backbone.parameters())),
        "embed_dim": info["embed_dim"],
        "precision": args.precision,
        "tf32": False,
        "input": "96x96 RGB in [0,1] -> bicubic 224x224 (align_corners=False) -> clamp[0,1] -> ImageNet mean/std",
        "versions": {
            "python": sys.version.split()[0], "torch": torch.__version__, "timm": timm.__version__,
            "huggingface_hub": huggingface_hub.__version__, "numpy": np.__version__, "h5py": h5py.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        },
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_model_dir.mkdir(parents=True, exist_ok=True)
    (out_model_dir / "extraction_meta_{}.json".format(gpu_tag(device))).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def autocast_ctx(precision: str, device):
    if precision == "fp32" or device.type != "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def extract_split(args, backbone, device, h5_dir: Path, split: str, out_dir: Path):
    x_path = h5_dir / "{}_x.h5".format(SPLIT_FILES[split])
    y_path = h5_dir / "{}_y.h5".format(SPLIT_FILES[split])
    if not x_path.exists():
        print("skip (missing):", x_path)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "done.json").exists():
        print("already done:", out_dir)
        return

    with h5py.File(y_path, "r") as fy:
        y = np.asarray(fy["y"][:]).reshape(-1)
    n = len(y) if args.limit is None else min(len(y), args.limit)
    y = y[:n]
    np.save(out_dir / "y.npy", y)

    cs = args.chunk_rows
    chunks = [(c, c * cs, min(n, (c + 1) * cs)) for c in range((n + cs - 1) // cs)]
    todo = [c for c in chunks if not (out_dir / "chunk_{:05d}.npy".format(c[0])).exists()]
    print("{} {}: n={} chunks={} todo={}".format(h5_dir.name, split, n, len(chunks), len(todo)))

    bs = args.batch_size
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_split = time.time()
    t_gpu = 0.0
    rows_done = 0
    for c_idx, s, e in todo:
        ranges = [(b, min(e, b + bs)) for b in range(s, e, bs)]
        loader = DataLoader(H5ChunkDataset(str(x_path), ranges), batch_size=None, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
        buf = np.empty((e - s, MODELS[args.model]["embed_dim"]), dtype=np.float32)
        for i, xb in loader:
            t0 = time.time()
            with autocast_ctx(args.precision, device):
                z = embed(args.model, backbone, xb.to(device, non_blocking=True))
            z = z.float().cpu().numpy()
            if not np.isfinite(z).all():
                raise RuntimeError("non-finite embeddings (precision={}); rerun with --precision fp32".format(args.precision))
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_gpu += time.time() - t0
            rs, re_ = ranges[i]
            buf[rs - s:re_ - s] = z
        tmp = out_dir / "chunk_{:05d}.tmp.npy".format(c_idx)
        np.save(tmp, buf.astype(args.store_dtype, copy=False))
        os.replace(tmp, out_dir / "chunk_{:05d}.npy".format(c_idx))
        rows_done += e - s
        el = time.time() - t_split
        print("  chunk {}/{}  rows {}  {:.1f} patches/s (wall)".format(
            c_idx + 1, len(chunks), e, rows_done / max(el, 1e-9)), flush=True)

    wall = time.time() - t_split
    done = {
        "n_rows": int(n), "n_chunks": len(chunks), "chunk_rows": cs, "batch_size": bs,
        "precision": args.precision, "store_dtype": args.store_dtype,
        "rows_this_session": rows_done,
        "wall_seconds_this_session": round(wall, 2),
        "model_seconds_this_session": round(t_gpu, 2),
        "patches_per_second_wall": round(rows_done / wall, 2) if wall > 0 and rows_done else None,
        "patches_per_second_model": round(rows_done / t_gpu, 2) if t_gpu > 0 else None,
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None,
        "source_x": str(x_path),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    (out_dir / "done.json").write_text(json.dumps(done, indent=2))
    print("done", out_dir, done)


def benchmark(args, backbone, device, out_model_dir: Path):
    """Pure forward-pass throughput / latency / memory on synthetic 96x96 input (for cost reporting)."""
    res = []
    x96 = np.random.RandomState(0).rand(max(args.bench_batches), 96, 96, 3).astype(np.float32)
    for prec in ["fp32", "fp16"] if device.type == "cuda" else ["fp32"]:
        for bs in args.bench_batches:
            xb = transform_batch(x96[:bs]).to(device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            with autocast_ctx(prec, device):
                for _ in range(2):
                    embed(args.model, backbone, xb)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                reps = 5 if bs <= 64 else 3
                t0 = time.time()
                for _ in range(reps):
                    embed(args.model, backbone, xb)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            dt = (time.time() - t0) / reps
            row = {"precision": prec, "batch_size": bs, "seconds_per_batch": round(dt, 4),
                   "ms_per_patch": round(1000 * dt / bs, 3), "patches_per_second": round(bs / dt, 1),
                   "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if device.type == "cuda" else None}
            print(row, flush=True)
            res.append(row)
    # CPU-side transform cost (bicubic resize + normalisation), per patch
    t0 = time.time()
    transform_batch(x96[:256])
    res.append({"cpu_transform_ms_per_patch": round(1000 * (time.time() - t0) / 256, 3)})
    (out_model_dir / "benchmark_forward_{}.json".format(gpu_tag(device))).write_text(json.dumps(res, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--h5-dir", action="append", default=[], help="folder with {split}_x.h5/{split}_y.h5 (repeatable)")
    ap.add_argument("--dataset-tag", action="append", default=[], help="name for each --h5-dir (same order)")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--chunk-rows", type=int, default=16384)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32",
                    help="fp32 matches the original runs exactly; fp16 is ~4-8x faster on T4/L4")
    ap.add_argument("--store-dtype", choices=["float32", "float16"], default="float32",
                    help="dtype of saved embeddings (float16 halves disk use)")
    ap.add_argument("--limit", type=int, default=None, help="only first N rows per split (testing)")
    ap.add_argument("--benchmark-only", action="store_true")
    ap.add_argument("--bench-batches", type=int, nargs="+", default=[1, 32, 256])
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_model_dir = Path(args.out_root) / args.model

    bench_file = out_model_dir / "benchmark_forward_{}.json".format(gpu_tag(device))
    if args.benchmark_only and bench_file.exists():
        print("benchmark already done on this GPU, skipping:", bench_file)
        return
    t0 = time.time()
    backbone = load_backbone(args.model, device)
    print("backbone loaded in {:.1f}s".format(time.time() - t0))
    if not (out_model_dir / "extraction_meta_{}.json".format(gpu_tag(device))).exists() or args.benchmark_only:
        write_meta(args, backbone, device, out_model_dir)

    if args.benchmark_only:
        benchmark(args, backbone, device, out_model_dir)
        return
    if len(args.h5_dir) != len(args.dataset_tag):
        ap.error("give one --dataset-tag per --h5-dir")
    for h5_dir, tag in zip(args.h5_dir, args.dataset_tag):
        for split in args.splits:
            extract_split(args, backbone, device, Path(h5_dir), split, out_model_dir / tag / split)


if __name__ == "__main__":
    main()
