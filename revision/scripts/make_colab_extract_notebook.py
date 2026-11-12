"""Generate revision/notebooks/colab_step1_extract_embeddings.ipynb."""

import json
from pathlib import Path

cells = [
    ("md", "# Step 1 — Extract foundation-model embeddings once (IEEE Access revision)\n\n"
           "Runtime → Change runtime type → **L4 GPU**. Run cells top to bottom.\n\n"
           "Everything is **resumable**: if Colab disconnects, just run all cells again — finished chunks are skipped.\n\n"
           "Expected time on L4 (fp32): roughly 1.5–3 h per model for both datasets."),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n!nvidia-smi"),
    ("code", "%cd /content\nimport os\nif os.path.isdir('GP_ECG'):\n    %cd GP_ECG\n    !git pull\nelse:\n"
             "    !git clone https://github.com/LeenRayess/GP_ECG.git GP_ECG\n    %cd GP_ECG\n!git log -1 --oneline"),
    ("code", "# ---- settings ----\n"
             "HF_TOKEN = ''  # paste from https://huggingface.co/settings/tokens (needs access to UNI2-h and Virchow2)\n"
             "MODELS = ['uni2h', 'virchow2']   # run both in one session, or one per session\n"
             "DATA = {\n"
             "    'pcam_macenko':  '/content/drive/MyDrive/GP_ECG_DATA/preprocessed_macenko_benchmark_style',\n"
             "    'cam17_macenko': '/content/drive/MyDrive/GP_ECG_DATA/wilds/preprocessed_macenko_benchmark_style',\n"
             "}\n"
             "OUT_ROOT = '/content/drive/MyDrive/GP_ECG_REVISION/embeddings'\n\n"
             "import os\nif not HF_TOKEN:\n    raise ValueError('Set HF_TOKEN first')\n"
             "os.environ['HF_TOKEN'] = HF_TOKEN\nos.environ['HUGGINGFACE_HUB_TOKEN'] = HF_TOKEN\n"
             "os.makedirs(OUT_ROOT, exist_ok=True)"),
    ("code", "!pip install -q -U timm h5py huggingface_hub\nimport torch, timm; print(torch.__version__, timm.__version__)"),
    ("code", "# Copy only the x/y .h5 files Drive -> local SSD (much faster reads)\nimport shutil, glob\nLOCAL = {}\n"
             "for tag, src in DATA.items():\n    dst = '/content/local_h5/' + tag\n    os.makedirs(dst, exist_ok=True)\n"
             "    for f in sorted(glob.glob(src + '/*_[xy].h5')):\n        out = os.path.join(dst, os.path.basename(f))\n"
             "        if not os.path.exists(out) or os.path.getsize(out) != os.path.getsize(f):\n"
             "            print('copying', f); shutil.copy(f, out + '.part'); os.replace(out + '.part', out)\n"
             "    LOCAL[tag] = dst\n    print(tag, os.listdir(dst))"),
    ("md", "### Speed / memory benchmark (for the execution-time comments). Takes ~2 min per model."),
    ("code", "for m in MODELS:\n    !python revision/scripts/extract_embeddings.py --model {m} --benchmark-only --out-root \"{OUT_ROOT}\""),
    ("md", "### Extract embeddings (train / valid / test of both datasets)"),
    ("code", "args = ' '.join('--h5-dir \"{}\" --dataset-tag {}'.format(LOCAL[t], t) for t in DATA)\n"
             "for m in MODELS:\n    !python revision/scripts/extract_embeddings.py --model {m} {args} --out-root \"{OUT_ROOT}\" --batch-size 256 --num-workers 2"),
    ("md", "### Check: every split should say DONE"),
    ("code", "import json\nfor m in MODELS:\n    for t in DATA:\n        for s in ['train', 'valid', 'test']:\n"
             "            p = f'{OUT_ROOT}/{m}/{t}/{s}/done.json'\n"
             "            print(m, t, s, 'DONE ' + str(json.load(open(p))['n_rows']) if os.path.exists(p) else 'MISSING')"),
]

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": s} if k == "md" else
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
        for k, s in cells
    ],
    "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"},
                 "kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "nbformat": 4, "nbformat_minor": 5,
}
out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_step1_extract_embeddings.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out)
