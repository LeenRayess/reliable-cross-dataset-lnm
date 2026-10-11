# Reliable Cross-Dataset Lymph Node Metastasis Detection

**Quality-aware preprocessing meets pathology foundation models**

Sameer Ahmad Hasan, [Leen Rayess](https://github.com/LeenRayess), and Zaineh Hamzeh  
Department of Biomedical Engineering, German Jordanian University

> **Paper status.** A manuscript with this title is **under peer review**. The draft in [`papers/`](papers/) is a convenience copy only. Copyright and availability may change after the review outcome, and this version may be removed from the repository.

---

Lymph-node slides are where a tiny cluster of tumor cells can change a patient’s staging. Pathologists hunt for those cells on hematoxylin and eosin (H&E) images every day. Deep learning is eager to help — until the next hospital, scanner, or stain protocol shows up, and a model that looked brilliant on one dataset quietly loses its nerve.

This repository is our answer to that very human problem, written as a locked experimental story rather than a leaderboard chase:

1. **Clean the inputs as if a second hospital will see them.** Deduplicate, block leakage, drop empty or artifact-heavy tiles, and standardize stain appearance.
2. **Train on one public lymph-node dataset, then test on another.** PatchCamelyon (PCam) and CAMELYON17-WILDS swap roles, so transfer is never a one-way compliment.
3. **Compare a small CNN with frozen pathology foundation models** (Virchow2 and UNI2-h). The question is not “can a giant encoder classify patches?” It is whether careful preprocessing is still doing real work once those encoders arrive.

Short version of the punchline: **foundation models travel well, but they are not a free pass on data hygiene.** On CAMELYON17 in-domain patches, the same quality-aware pipeline lifted a conventional CNN from an ROC-AUC of **0.72** and recall of **0.14** to **0.98** and **0.91**. PCam-trained Virchow2 and UNI2-h then transferred to CAMELYON17 with ROC-AUC **0.997** and **0.998**. The reverse direction is where the plot thickens: ranking can stay strong while a locked operating point and calibration start to wobble — which is exactly the kind of failure a clinic would feel.

<p align="center">
  <img src="reports/figures/figure11_transfer_grid_roc_auc_cnn_virchow.png" width="720" alt="Bidirectional transfer ROC-AUC grid for the CNN and Virchow2">
</p>
<p align="center"><em>Bidirectional transfer: train on one dataset, test on both. High in-domain scores are nice; the off-diagonal is the honest part.</em></p>

## What lives here

| Path | What you will find |
| --- | --- |
| [`scripts/`](scripts/) | Preprocessing, training, evaluation, and figure-generation scripts |
| [`src/`](src/) | Shared preprocessing helpers |
| [`notebooks/`](notebooks/) | Figure notebooks and Colab runbooks |
| [`reports/`](reports/) | Manuscript figures, dataset audits, qualitative error galleries |
| [`pcam_dedup/`](pcam_dedup/) | Indices of unique PCam patches after exact-content deduplication |
| [`pcam-master/`](pcam-master/) | Vendored [PatchCamelyon](https://github.com/basveeling/pcam) loader |
| [`papers/`](papers/) | Graduation report and the under-review manuscript draft |
| [`docs/`](docs/) | Qualitative error-analysis protocol |

Datasets themselves are **not** in git. They are large public corpora; you download them locally.

## How the study is put together

```
PCam  ──┐                      ┌─► in-domain test
        ├── quality filter ────┤
CAM17 ──┘   stain standardize  └─► cross-dataset test
              │
              ├─► shallow CNN (trained end-to-end)
              ├─► Virchow2 (frozen encoder + linear head)
              └─► UNI2-h  (frozen encoder + linear head)
```

Integrity comes first: split isolation, exact-hash deduplication on PCam, cross-dataset overlap checks, and tissue-quality gates. Stain handling is treated as an experimental factor, not a silent default. Model selection and temperature calibration stay on source-domain validation. Test labels from either corpus never get a vote.

<p align="center">
  <img src="reports/figures/figure1_quality_filter_panels.png" width="720" alt="Examples of patches retained or rejected by quality filtering">
</p>
<p align="center"><em>Quality control is unglamorous and slightly fussy. It is also why the CNN stops drowning in empty tiles and stain chaos.</em></p>

## Highlights

- **Preprocessing is a first-class result**, not a methods footnote. The CNN recovery on CAMELYON17 is the clearest receipt.
- **Forward transfer (PCam → CAMELYON17) is near ceiling** for both frozen foundation-model heads after the shared pipeline.
- **Reverse transfer is asymmetric.** UNI2-h kept more recall and F1 than Virchow2 when CAMELYON17-trained models were locked and sent to PCam; calibration and high-specificity operating points made the gap visible even when ROC-AUC looked polite.
- **Errors have a biology, not just a confusion matrix.** Structured review of false negatives, false positives, and high-entropy cases is stored under `reports/qualitative_error_analysis/`.

## Setup

Python 3.10+ is a comfortable starting point. Create a virtual environment and install the pinned stack:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Foundation-model runs need PyTorch, `timm`, and a Hugging Face account with access to the gated checkpoints (Virchow2, UNI2-h). Leave tokens out of the repo; set `HF_TOKEN` in the environment or in a private Colab cell.

### Data

1. **PatchCamelyon** — follow the download notes in [`pcam-master/README.md`](pcam-master/README.md) and extract with `pcam-master/keras_pcam/dataset/extract_pcam.py`.
2. **CAMELYON17-WILDS** — use the WILDS packaging and [`scripts/camelyon17/prepare_wilds_camelyon17_h5.py`](scripts/camelyon17/prepare_wilds_camelyon17_h5.py).
3. Run the quality-aware preprocessor:

```bash
python scripts/preprocess_histopath_h5.py --help
```

Training and evaluation entry points live in `scripts/` (`train_virchow_preprocessed.py`, `train_uni2h_preprocessed_colab.py`, `cnn_baseline_full_pipeline.py`, and the matching Colab notebooks). Figure notebooks in `notebooks/figure*.ipynb` rebuild the panels under `reports/figures/`.

## Citation

Until the paper is accepted, please cite this repository and name the manuscript title:

> S. A. Hasan, L. Rayess, and Z. Hamzeh, “Reliable Cross-Dataset Lymph Node Metastasis Detection Using Quality-Aware Preprocessing and Pathology Foundation Models,” *manuscript under review*, 2026.

## Acknowledgements

PCam was released by Veeling et al. CAMELYON17 / WILDS provide the multi-hospital nodal patches. Virchow2 and UNI2-h are used here as frozen public foundation backbones under their respective licenses and access terms. The PCam helper code under `pcam-master/` retains its original MIT license.

## License

Project code is released under the [MIT License](LICENSE). Dataset licenses remain with their original publishers. See [`papers/README.md`](papers/README.md) before redistributing the report or the manuscript draft.
