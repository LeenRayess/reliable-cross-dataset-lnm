"""Generate revision/notebooks/colab_step4_cnn_seeds.ipynb (repeated-seed shallow-CNN baseline on Colab L4)."""

import json
from pathlib import Path

D = "/content/drive/MyDrive/GP_ECG_DATA"
cells = [
    ("md", "# Step 4 — Shallow CNN baseline, repeated seeds (IEEE Access revision)\n\n"
           "Runtime → Change runtime type → **L4 GPU** (High-RAM if offered). Then **Runtime → Run all**.\n\n"
           "Trains the same shallow CNN as the paper on raw and preprocessed PCam and CAMELYON17 (4 arms × 5 seeds = 20 runs).\n\n"
           "**Resumable:** finished runs are saved to Drive (`GP_ECG_REVISION/step4_runs`). If Colab disconnects, just Run all again; "
           "finished runs are skipped (only the run that was interrupted restarts)."),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n!nvidia-smi\n!free -g\n!df -h /content"),
    ("code", "%cd /content\nimport os\nif os.path.isdir('GP_ECG'):\n    %cd GP_ECG\n    !git pull\nelse:\n"
             "    !git clone https://github.com/LeenRayess/reliable-cross-dataset-lnm.git GP_ECG\n    %cd GP_ECG\n!git log -1 --oneline"),
    ("md", "### Copy the data from Drive to the fast local disk (~48 GB, about 10–20 min; skipped if already there)"),
    ("code", "import shutil\nSRC = {\n"
             f"  'pcam_raw': {{'train': '{D}/pcam_raw/training/camelyonpatch_level_2_split_train', "
             f"'valid': '{D}/pcam_raw/val/camelyonpatch_level_2_split_valid', 'test': '{D}/pcam_raw/test/camelyonpatch_level_2_split_test'}},\n"
             f"  'pcam_preprocessed': {{s: '{D}/preprocessed_macenko_benchmark_style/' + s for s in ['train', 'valid', 'test']}},\n"
             f"  'cam17_raw': {{s: '{D}/wilds_raw/' + s for s in ['train', 'valid', 'test']}},\n"
             f"  'cam17_preprocessed': {{s: '{D}/wilds/preprocessed_macenko_benchmark_style/' + s for s in ['train', 'valid', 'test']}},\n"
             "}\nDATA_ROOT = '/content/cnn_data'\n"
             "for surf, splits in SRC.items():\n    os.makedirs(f'{DATA_ROOT}/{surf}', exist_ok=True)\n"
             "    for split, stem in splits.items():\n        for k in ['x', 'y']:\n"
             "            src, dst = f'{stem}_{k}.h5', f'{DATA_ROOT}/{surf}/{split}_{k}.h5'\n"
             "            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):\n"
             "                print('copying', src, flush=True); shutil.copy(src, dst + '.part'); os.replace(dst + '.part', dst)\n"
             "print('data ready')\n"
             "for surf in SRC:\n    for f in sorted(os.listdir(f'{DATA_ROOT}/{surf}')):\n"
             "        print(surf, f, round(os.path.getsize(f'{DATA_ROOT}/{surf}/{f}') / 1e9, 2), 'GB')"),
    ("md", "### Train (about 6–10 h in total; progress prints one line per epoch)"),
    ("code", "OUT = '/content/drive/MyDrive/GP_ECG_REVISION'\n"
             "!python revision/scripts/step4_cnn_seeds.py --data-root /content/cnn_data "
             "--arms pcam_raw pcam_preprocessed cam17_preprocessed cam17_raw --seeds 0 1 2 3 4 "
             "--out-runs \"{OUT}/step4_runs\" --out-res \"{OUT}/step4_results\" --workers 8"),
    ("md", "### Check: every run should say DONE"),
    ("code", "for arm in ['pcam_raw', 'pcam_preprocessed', 'cam17_raw', 'cam17_preprocessed']:\n    for s in range(5):\n"
             "        p = f'{OUT}/step4_runs/{arm}_seed{s}/done.json'\n        print(arm, s, 'DONE' if os.path.exists(p) else 'MISSING')"),
]

nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": s} if k == "md" else
                {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s} for k, s in cells],
      "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"}, "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_step4_cnn_seeds.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
