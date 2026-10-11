# Notebooks

## Manuscript figures

Notebooks named `figure*.ipynb` rebuild the panels stored in [`reports/figures/`](../reports/figures/). They assume preprocessed H5 files and the evaluation outputs from `scripts/`.

## Training sketches

| Notebook | Role |
| --- | --- |
| `pcam_baseline_training.ipynb` | Early CNN baseline path |
| `pcam_virchow2_transfer.ipynb` | Virchow2 transfer experiments |
| `stain_normalize_full_dataset.ipynb` | Full-corpus stain normalization notebook |
| `qualitative_review_virchow_c1_c4.ipynb` | Structured review of Virchow2 error buckets |

## Colab

[`colab/`](colab/) contains beginner-oriented Google Colab runbooks for training and evaluating Virchow2 and UNI2-h on the preprocessed PCam and WILDS tensors.

Do not paste Hugging Face tokens into committed cells. Use an environment variable or a private Colab secret.
