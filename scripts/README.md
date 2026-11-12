# Scripts

Command-line entry points for the locked experimental pipeline.

## Preprocessing and integrity

| Script | Role |
| --- | --- |
| `preprocess_histopath_h5.py` | Quality filter, stain normalization, and H5 export for PCam and CAMELYON17 |
| `preprocess_pcam_to_h5.py` | PCam-only H5 preparation |
| `dedup_pcam.py` | Exact-content deduplication of PCam patches |
| `check_overlap_preprocessed_exact.py` | Cross-dataset hash overlap after preprocessing |
| `analyze_histopath_datasets.py` | Descriptor / audit summaries |
| `prepare_stain_benchmark_h5.py` | Stain-method benchmark tensors |
| `camelyon17/` | WILDS packing, CAMELYON17 H5 prep, and PCam–CAM17 overlap checks |

## Models

| Script | Role |
| --- | --- |
| `cnn_baseline_full_pipeline.py` | End-to-end shallow CNN baseline |
| `train_virchow_preprocessed.py` | Frozen Virchow2 encoder + linear head |
| `train_virchow_preprocessed_colab.py` | Colab-oriented Virchow2 training |
| `train_uni2h_preprocessed_colab.py` | Frozen UNI2-h encoder + linear head |
| `evaluate_virchow_preprocessed_test_colab.py` | Virchow2 test evaluation |
| `evaluate_uni2h_preprocessed_test_colab.py` | UNI2-h test evaluation |
| `virchow_test_bootstrap_inference.py` | Bootstrap / paired inference helpers |

## Figures and qualitative review

| Script | Role |
| --- | --- |
| `plot_results_figures_7_10.py` | Stain benchmark, confusion, error-pool figures |
| `plot_results_figures_11_13.py` | Transfer-grid and transfer-gap figures |
| `plot_stain_norm_methods_panel.py` | Macenko / Reinhard / Vahadane panel |
| `export_qualitative_review_patches.py` | Patch export for structured error review |
| `reorder_qualitative_review_csvs.py` | Review-table housekeeping |
| `compare_preprocessed_stain_metrics.py` | Stain-metric comparisons |

Run everything from the repository root. Use `--help` on each script for the current flags.
