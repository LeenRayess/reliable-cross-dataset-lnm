# -*- coding: utf-8 -*-
"""Content of the revision work log (Word). Edit here, then run build_work_log.py.

Each entry block: list of (kind, payload)
  ("h", text)            sub-heading
  ("p", text)            paragraph  (use **bold** for emphasis)
  ("b", [items])         bullet list
  ("t", [header, rows])  table
"""

TITLE = "IEEE Access Revision — Experimental Work Log"
SUBTITLE = "Manuscript Access-2026-47700 · experiments and changes for the resubmission"

INTRO = [
    ("p", "This document records every experiment, code change and result produced for the resubmission. "
          "It is written so that the manuscript text and the response-to-reviewers letter can be drafted directly from it. "
          "Each step states **what was done**, **why** (which reviewer comment it answers), **how**, **what was found** "
          "(every number), and **where the files are**. It is a factual record; it does not contain proposed manuscript text."),
    ("p", "All code for the revision lives in the repository folder revision/ (scripts, Colab notebooks, results). "
          "Numbers in this log are copied from result files that the scripts write; they are never typed by hand."),
]

# Reviewer comment -> step -> status. Status: Planned / In progress / Done / Writing only (for the professor)
TRACKER = [
    ["Reviewer & comment", "What is needed", "Step", "Status"],
    ["R1-Q2, R3-5, R3-9", "Repeat foundation-model runs with several seeds and identical (harmonized) head settings", "1, 2, 3", "Done"],
    ["R1-Q2", "Repeat CNN arms with several seeds; confirm raw and preprocessed inputs scaled the same way", "4", "Planned"],
    ["R3-3", "Component-wise ablation of the preprocessing pipeline", "5", "Planned"],
    ["R3-4", "Quality-control filter: class-specific effect, alternative thresholds, what is removed", "6", "Planned"],
    ["R3-7", "Macenko justification: alternative reference patches and normalizers", "7", "Planned"],
    ["R3-2, R3-12", "Patient-/slide-level analysis; limits of 10 test slides", "8", "Planned"],
    ["R3-8", "Why source-domain temperature scaling does not transfer; alternative calibration", "9", "Planned"],
    ["R3-11", "Additional baseline (ImageNet-pretrained network)", "10", "Planned"],
    ["R3-13", "False-positive / false-negative patch analysis", "11", "Planned"],
    ["R2-2, R3 (additional)", "Execution time, inference time, memory, deployment cost", "1 (+12)", "Done for foundation models; CNN in Step 4"],
    ["R3-10", "Exact model versions, dependencies, code, seeds", "0, 1, 2, 3", "Done (foundation models); CNN seeds in Step 4"],
    ["R3 (additional)", "Numerical consistency between tables / supplement", "12", "Planned"],
    ["R1-Q2, R3-6", "Relabel C3 as held-out-center evaluation", "—", "Writing only (see note N1)"],
    ["R1-Q3/Q4, R2-1/3/4/5/6, R3-1/14/15, conclusion", "Text, references, novelty, limitations, independent-dataset argument", "—", "Writing only"],
]

NOTES = [
    ("h", "Notes for the manuscript (found while setting up)"),
    ("b", [
        "**N1 — C3 is a held-out-center test, and the CAMELYON17 validation split is also a different center.** "
        "The CAMELYON17-WILDS models were trained on centers 0, 3, 4; checkpoint selection and temperature fitting used the "
        "WILDS out-of-distribution validation split (center 1, 34,904 patches); testing used center 2. "
        "So C3 is trained on three hospitals, tuned on a fourth and tested on a fifth. Source: data/wilds/camelyon17_h5_full_oodval/manifest.json (valid_source = \"val\").",
        "**N2 — Number to reconcile.** The CAMELYON17 raw-CNN recall appears as 0.16 in the thesis text and 0.1381 in the manuscript. "
        "Will be resolved in Step 4 when the CNN arms are re-run (the new multi-seed numbers will replace both).",
        "**N3 — CNN input scaling (R1-Q2).** The original CNN script (scripts/cnn_baseline_full_pipeline.py) divides a patch by 255 only if its "
        "maximum value exceeds 1.5, for raw and preprocessed inputs alike. Preprocessed patches are already stored in [0,1]; raw uint8 patches are scaled, "
        "except an almost-black raw patch whose maximum pixel value is 0 or 1, which would stay unscaled. Raw and preprocessed inputs therefore reach the "
        "network on the same [0,1] scale apart from this edge case. Step 4 replaces the rule by scaling on data type (uint8 → /255) and quantifies the edge case.",
        "**N4 — Patient and slide identifiers are available** for every CAMELYON17-WILDS patch (data/wilds/camelyon17_v1.0/metadata.csv: patient, node, slide, center), "
        "and the preprocessed files store the original index of every retained patch (checkpoint_step1_<split>.npz, kept_indices). This enables the "
        "patient-/slide-level analyses in Step 8.",
        "**N5 — Data copies are identical.** The preprocessed H5 files used on Colab (Google Drive GP_ECG_DATA) and on the local PC have identical byte sizes "
        "for all 12 files (PCam and CAMELYON17, train/valid/test, x and y).",
        "**N6 — Head dropout was never active in the original foundation-model runs (affects the Methods text).** "
        "The original training scripts (scripts/train_virchow_preprocessed_colab.py, scripts/train_uni2h_preprocessed_colab.py) define a dropout layer "
        "(p = 0.2 where configured), but forward() always calls forward_logits(mc_dropout=False), which sets the dropout layer to eval mode before use, "
        "including during training. Dropout was therefore never applied. Verified in Step 3: retraining without dropout reproduces the original "
        "training-loss values (Virchow2 CAMELYON17 epoch 10: original 0.0194, repeats 0.0193–0.0196; Virchow2 PCam: 0.1043 vs 0.1049–0.1060; "
        "UNI2-h PCam: 0.0888 vs 0.0888–0.0890), whereas retraining with dropout 0.2 active gives clearly higher values (0.027, 0.134). "
        "Consequence: the manuscript statements that heads used dropout 0.2 (PCam heads, Virchow2 CAMELYON17 head) do not describe what was run; "
        "all eight original heads were effectively trained without dropout. The only real configuration difference between the Virchow2 and UNI2-h "
        "CAMELYON17 heads was the batch size (320 vs 256).",
        "**N7 — CAMELYON17 test-set composition.** The 83,181 retained CAMELYON17 test patches come from 10 slides of 9 patients (center 2). "
        "One slide (slide 28, patient 51) contributes 31,687 patches (38.1%), 85.0% of them tumour-positive. Patches per slide range from 3,373 to 31,687; "
        "positive fraction per slide ranges from 0.3% to 85.0% (slides 20, 21, 24, 27 are ≥97.5% negative). "
        "PCam test: 27,704 retained patches from 129 whole-slide images. Source: revision/results/test_clusters/.",
        "**N8 — Manuscript reference copy.** The submitted manuscript is in docs/IEEE_Access_Manuscript_SUBMISSION_READY_v2_bio.docx "
        "(byte-identical to papers/manuscript_draft_under_review.docx). All comparisons with 'the manuscript' in this log refer to this file.",
        "**N9 — Not experimental:** R3-1 (add an independent institution) is not part of the current experiment plan; "
        "it needs a decision on whether to argue and restrict the claims or to add a dataset.",
    ]),
]

STEPS = [
    {
        "title": "Step 0 — Setup and version pinning",
        "date": "2026-11-13",
        "answers": "R3-10 (reproducibility and model-version specification)",
        "blocks": [
            ("h", "What was done"),
            ("b", [
                "Created the revision/ folder for all new code, notebooks and results, and this log.",
                "Recorded the software environment and identified the exact Virchow2 weights that were used in the original experiments.",
            ]),
            ("h", "How the exact model version was identified"),
            ("p", "Each original trained model file (model_best.pt) contains a full copy of the frozen encoder weights. "
                  "A SHA-256 fingerprint of those weights was computed and compared with the fingerprint of the Hugging Face release. "
                  "For Virchow2 the fingerprints are identical, so the original experiments used Hugging Face commit "
                  "**3158645804b69e3f3bc4439d4116edddf0840a72** of paige-ai/Virchow2. "
                  "The same check for UNI2-h (Step 2) shows that the original UNI2-h models used Hugging Face commit "
                  "**d517a8dd47902dd7c308b3c36f63bce47e7b9a43** of MahmoodLab/UNI2-h."),
            ("t", [["Item", "Value"], [
                ["Virchow2", "paige-ai/Virchow2 @ 3158645804b69e3f3bc4439d4116edddf0840a72 (verified by weight hash); 631.2 M parameters; 2560-d embedding (CLS + mean patch token)"],
                ["UNI2-h", "MahmoodLab/UNI2-h @ d517a8dd47902dd7c308b3c36f63bce47e7b9a43 (verified by weight hash); 681.4 M parameters; 1536-d CLS embedding"],
                ["Local environment", "Python 3.10.7, PyTorch 2.5.1 (CUDA 12.1), timm 1.0.31, huggingface_hub 2.4.0, NumPy 2.2.6, h5py 3.16.0, scikit-learn 1.7.2"],
                ["Colab environment (embedding extraction)", "NVIDIA L4, Python 3.13.16, PyTorch 2.14.1 (CUDA 13.0, cuDNN 9.28), timm 1.0.31, huggingface_hub 2.4.0, NumPy 2.4.6, h5py 3.16.0"],
            ]]),
            ("h", "Files"),
            ("b", ["revision/scripts/fm_common.py (model loading, input transform, weight fingerprint)"]),
        ],
    },
    {
        "title": "Step 1 — Cache frozen encoder embeddings (and measure compute cost)",
        "date": "2026-11-13",
        "answers": "Enables R1-Q2, R3-5, R3-9 (multi-seed, harmonized runs); answers R2-2 and R3 (execution time, memory)",
        "blocks": [
            ("h", "Why"),
            ("p", "UNI2-h and Virchow2 are used with a frozen encoder: only the final linear layer is trained. "
                  "Because the encoder never changes and the input transform has no randomness, each patch always produces "
                  "exactly the same embedding. The original training scripts recomputed these identical embeddings in every epoch, "
                  "which made each run take days. Computing each embedding once and training the linear head on the stored "
                  "embeddings gives the same model in minutes, which makes the requested repeated-seed and harmonized experiments feasible."),
            ("h", "How"),
            ("b", [
                "revision/scripts/extract_embeddings.py runs each encoder once over train / validation / test of the preprocessed PCam and CAMELYON17 sets "
                "and saves the embeddings. The input transform is identical to the original scripts (bicubic 96→224, clamp to [0,1], ImageNet mean/std). "
                "The encoder forward pass runs in half precision (fp16 autocast) and embeddings are stored as float16; see the precision check below.",
                "It also records: Hugging Face commit, weight fingerprint, software versions, GPU, throughput (patches/s), "
                "per-patch latency at batch sizes 1, 32, 256 in fp32 and fp16, and peak GPU memory.",
                "Runs on Google Colab (L4 GPU) via revision/notebooks/colab_step1_extract_embeddings.ipynb; resumable after disconnects.",
                "Before the Colab run, the pipeline (chunked writing, resuming, loading) was tested locally with a small randomly initialised ViT "
                "on 700 PCam test patches, and with the real Virchow2 encoder on the first 512 CAMELYON17 test patches (results below).",
                "Run history: a first extraction on a Colab T4 completed UNI2-h for five of six splits before a disconnect; for consistency the whole "
                "extraction was then repeated on an L4 and the T4 output was archived (used only for the cross-GPU check in Step 2). "
                "A guard stops extraction if any embedding is non-finite (none occurred).",
            ]),
            ("h", "Equivalence check (local, completed)"),
            ("p", "To show that the cached-embedding approach reproduces the original models, the original PCam-trained Virchow2 model "
                  "(its saved linear layer) was applied to cached embeddings of the first 512 CAMELYON17 test patches and compared with the "
                  "predictions saved by the original evaluation run:"),
            ("t", [["Check", "Result"], [
                ["Labels in the same order", "Yes"],
                ["Same decision at threshold 0.5", "512 / 512 (100%)"],
                ["Largest logit difference", "0.0099 (mean 0.0017; logits range −3.4 to 10.8)"],
                ["Encoder weights identical to original run", "Yes (SHA-256 match)"],
            ]]),
            ("p", "**Precision check (fp16).** Full-precision (fp32) extraction would take roughly a day per model on a Colab T4 GPU. "
                  "Half precision (fp16) is several times faster and halves storage. The same check with fp16 embeddings stored as float16 "
                  "(first 256 CAMELYON17 test patches, original PCam-trained Virchow2 head) gave the **same decision at 0.5 for 256/256 patches**; "
                  "largest logit difference 0.072 (mean 0.010), largest probability difference 0.005. All new experiments use the same fp16 "
                  "embeddings, so they are fully consistent with one another; the original single runs are kept as the reference."),
            ("p", "The tiny differences are ordinary floating-point differences between GPUs. The full check on all test sets for both models "
                  "is Step 2. **Consequence: the original results remain valid and directly comparable to the new runs.**"),
            ("p", "For reference, the local GPU (GTX 1050 Ti, 4 GB) processes 1.9 patches/s with Virchow2 in fp32 (peak memory 3.2 GB) — "
                  "this is why extraction runs on Colab."),
            ("h", "Results (Colab, completed 2026-11-14)"),
            ("p", "All embeddings used in the experiments were extracted on one GPU type (Google Colab NVIDIA L4), fp16, stored as float16: "
                  "676,438 patches per model (PCam train 208,355 / valid 26,515 / test 27,704; CAMELYON17 train 296,294 / valid 34,389 / test 83,181), "
                  "about 2.25 h per model at 83.8 (UNI2-h) and 83.4 (Virchow2) patches/s including disk reading and resizing. Total cached size 5.2 GB."),
            ("p", "**Inference cost (encoder forward pass, synthetic 96×96 inputs upsampled to 224×224):**"),
            ("t", [["Model / GPU / precision", "Batch 1 (ms/patch)", "Batch 32 (ms/patch)", "Batch 256 (ms/patch)", "Peak GPU memory, batch 256"], [
                ["UNI2-h / L4 / fp16", "26.6", "11.5", "12.0", "6.8 GB"],
                ["UNI2-h / L4 / fp32", "55.2", "46.2", "45.0", "9.0 GB"],
                ["UNI2-h / T4 / fp16", "47.6", "23.4", "27.0", "6.8 GB"],
                ["UNI2-h / T4 / fp32", "126.2", "93.3", "122.3", "9.0 GB"],
                ["Virchow2 / L4 / fp16", "25.2", "11.7", "11.5", "5.9 GB"],
                ["Virchow2 / L4 / fp32", "55.7", "47.0", "43.7", "7.7 GB"],
                ["Virchow2 / T4 / fp16", "45.8", "24.2", "25.9", "5.9 GB"],
                ["Virchow2 / T4 / fp32", "141.9", "102.7", "117.5", "7.7 GB"],
            ]]),
            ("p", "Model sizes: UNI2-h 681.4 M parameters, Virchow2 631.2 M parameters (frozen); trainable linear head 1,537 (UNI2-h) / 2,561 (Virchow2) parameters. "
                  "Batch-1 memory (about 2.6–2.8 GB) is essentially the model weights. CPU-side resizing/normalisation costs about 1 ms per patch. "
                  "Example: the 83,181-patch CAMELYON17 test set is encoded in about 17 min on one L4 (fp16)."),
            ("p", "**Training cost after caching:** a linear head on the 208,355 cached UNI2-h PCam training embeddings trains for 10 epochs in "
                  "20 s on a GTX 1050 Ti (best validation accuracy 0.939, ROC-AUC 0.979; the original run reported 0.940 and 0.980). "
                  "In the original pipeline the encoder was re-run in every epoch (about 2 million encoder passes per training run), which took days."),
            ("h", "Files"),
            ("b", [
                "revision/scripts/extract_embeddings.py, revision/scripts/emb_io.py, revision/scripts/fm_common.py",
                "revision/notebooks/colab_step1_extract_embeddings.ipynb",
                "Google Drive: GP_ECG_REVISION/embeddings/<model>/<dataset>/<split>/ (embeddings, labels, done.json with timing), "
                "<model>/extraction_meta_<gpu>.json, <model>/benchmark_forward_<gpu>.json; local copy in data/revision_embeddings/ (not in git)",
            ]),
        ],
    },
    {
        "title": "Step 2 — Original models reproduce their results on the cached embeddings",
        "date": "2026-11-14",
        "answers": "Shows the new experiments are comparable with the submitted results; R3-10 (exact model versions)",
        "blocks": [
            ("h", "What was done"),
            ("p", "The original trained linear heads (from the original model_best.pt files) were applied to the new L4 test embeddings "
                  "for all eight model × condition cells (Virchow2 and UNI2-h, C1–C4) and compared patch-by-patch with the original "
                  "prediction files (test_predictions.npz). The encoder weights stored inside every original checkpoint were fingerprinted "
                  "and compared with the encoder used for extraction."),
            ("h", "Results"),
            ("t", [["Model", "Cond.", "n test", "Decisions changed at 0.5", "ROC-AUC original → cached", "Accuracy original → cached", "Max |Δ logit|"], [
                ["Virchow2", "C1", "27,704", "7 (0.03%)", "0.9813 → 0.9813", "0.9367 → 0.9367", "0.086"],
                ["Virchow2", "C2", "83,181", "9 (0.01%)", "0.9971 → 0.9971", "0.9840 → 0.9840", "0.110"],
                ["Virchow2", "C3", "83,181", "10 (0.01%)", "0.9966 → 0.9966", "0.9798 → 0.9798", "0.144"],
                ["Virchow2", "C4", "27,704", "15 (0.05%)", "0.9502 → 0.9503", "0.8237 → 0.8238", "0.143"],
                ["UNI2-h", "C1", "27,704", "2 (0.01%)", "0.9855 → 0.9855", "0.9502 → 0.9502", "0.029"],
                ["UNI2-h", "C2", "83,181", "2 (<0.01%)", "0.9980 → 0.9980", "0.9853 → 0.9853", "0.024"],
                ["UNI2-h", "C3", "83,181", "0", "0.9974 → 0.9974", "0.9837 → 0.9837", "0.025"],
                ["UNI2-h", "C4", "27,704", "2 (0.01%)", "0.9811 → 0.9811", "0.8963 → 0.8963", "0.031"],
            ]]),
            ("b", [
                "**Every original checkpoint contains exactly the encoder weights used for extraction** (SHA-256 match): Virchow2 = Hugging Face "
                "commit 3158645804b69e3f3bc4439d4116edddf0840a72, UNI2-h = commit d517a8dd47902dd7c308b3c36f63bce47e7b9a43. "
                "This resolves the reviewer concern that the UNI2-h version was not pinned.",
                "The checkpoint-to-training-set mapping for UNI2-h, previously marked provisional, is confirmed from the data: "
                "model_best-001/-003 and uni2h_pcam_train__model_best2.pt = PCam-trained; model_best-002/-004 and uni2h_cam17_train__model_best2.pt = CAMELYON17-trained.",
                "**Cross-GPU reproducibility:** UNI2-h embeddings extracted on a T4 and on an L4 (five splits, 593,257 patches) have a mean cosine similarity "
                "of 0.9999999 (minimum 0.99998); with the original heads, at most 3 of 593,257 decisions differ.",
            ]),
            ("h", "Interpretation"),
            ("p", "Reported metrics are reproduced to the fourth decimal place; at most 15 of 27,704 patch decisions (0.05%) differ, all from "
                  "half-precision rounding of patches lying at the decision boundary. The cached-embedding pipeline is therefore the same model as the "
                  "original, and the repeated-seed and harmonized experiments in Step 3 are directly comparable with the submitted results."),
            ("h", "Files"),
            ("b", [
                "revision/scripts/step2_equivalence.py → revision/results/step2/equivalence.json",
                "revision/scripts/step2_cross_gpu.py → revision/results/step2/cross_gpu_t4_vs_l4.json",
                "revision/results/step2/original_heads/*.npz (original linear heads + temperatures, extracted from the original checkpoints)",
            ]),
        ],
    },
    {
        "title": "Step 3 — Repeated-seed and harmonized foundation-model training",
        "date": "2026-11-14",
        "answers": "R1-Q2 (single non-harmonized runs), R3-5 (controlled comparison), R3-9 (repeated-run variability)",
        "blocks": [
            ("h", "What was done"),
            ("b", [
                "Linear heads were retrained on the cached embeddings (Step 1) for both encoders and both training sets, with **10 seeds (0–9)** "
                "per configuration. The seed sets head initialisation, dropout masks (where used) and training-batch order. 240 training runs in total.",
                "**Configuration arms:** (1) **harmonized** — identical for every encoder and training set: batch 256, no dropout, Adam lr 0.001, "
                "weight decay 0, 10 epochs, BCEWithLogits loss, checkpoint = first epoch with the highest source-validation accuracy, temperature fitted on "
                "source validation (log grid of 60 values in [0.2, 5]). (2) **original** — the effective configuration of the submitted runs: as harmonized, "
                "except the Virchow2 CAMELYON17 head uses batch 320 (see note N6: dropout was never active originally). "
                "(3) **dropout02** — sensitivity arm with dropout 0.2 actually applied where the original configs listed it "
                "(PCam heads of both encoders and the Virchow2 CAMELYON17 head; batch sizes as original).",
                "Because UNI2-h and the PCam heads have the same settings in the harmonized and original arms, those cells are identical between "
                "the two arms (same seeds give the same runs; confirmed). The two arms differ only for Virchow2 trained on CAMELYON17 (C3, C4).",
                "Test metrics use the same definitions as the submitted paper. Check: recomputing them from the original prediction files reproduces "
                "the published values exactly (e.g. Virchow2 C4 ROC-AUC 0.9502, recall 0.6304, ECE15 0.1600, sensitivity at ≥95% specificity 0.8065, "
                "specificity at ≥90% sensitivity 0.8362).",
                "Slide/patient identifiers for every retained test patch were rebuilt from the official metadata (PCam: wsi column of the test meta CSV, "
                "129 slides; CAMELYON17: slide/patient of metadata.csv, 10 slides / 9 patients), label order verified for every patch.",
                "**Two-level bootstrap** (2,000 replicates): each replicate resamples test slides with replacement (shared by both encoders, so contrasts are paired) "
                "and draws one random training seed per encoder; 95% percentile intervals. These intervals include both slide-level sampling "
                "variability and between-training-run variability.",
                "Also computed: position of each original single run within the seed distribution; fraction of the 10 × 10 seed pairs in which UNI2-h is better; "
                "sensitivity of results to the checkpoint-selection rule (validation accuracy vs validation ROC-AUC vs validation loss vs last epoch).",
            ]),
            ("h", "Results — harmonized arm, mean ± SD over 10 seeds (calibrated probabilities, threshold 0.5)"),
            ("t", [["Cond.", "Model", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1", "Brier", "Log loss", "ECE15"], [
                ["C1", "UNI2-h", "0.9849 ± 0.0007", "0.9856 ± 0.0005", "0.9490 ± 0.0023", "0.9757 ± 0.0030", "0.9136 ± 0.0061", "0.9436 ± 0.0027", "0.0405 ± 0.0015", "0.1484 ± 0.0050", "0.0188 ± 0.0035"],
                ["C1", "Virchow2", "0.9802 ± 0.0010", "0.9812 ± 0.0008", "0.9364 ± 0.0024", "0.9676 ± 0.0043", "0.8939 ± 0.0089", "0.9293 ± 0.0031", "0.0490 ± 0.0019", "0.1731 ± 0.0066", "0.0194 ± 0.0079"],
                ["C2", "UNI2-h", "0.9979 ± 0.0002", "0.9982 ± 0.0001", "0.9842 ± 0.0013", "0.9802 ± 0.0032", "0.9890 ± 0.0016", "0.9846 ± 0.0013", "0.0151 ± 0.0012", "0.0684 ± 0.0042", "0.0345 ± 0.0056"],
                ["C2", "Virchow2", "0.9967 ± 0.0003", "0.9975 ± 0.0002", "0.9818 ± 0.0012", "0.9842 ± 0.0024", "0.9801 ± 0.0034", "0.9821 ± 0.0012", "0.0151 ± 0.0010", "0.0615 ± 0.0034", "0.0113 ± 0.0014"],
                ["C3", "UNI2-h", "0.9973 ± 0.0003", "0.9979 ± 0.0002", "0.9840 ± 0.0008", "0.9946 ± 0.0006", "0.9739 ± 0.0021", "0.9841 ± 0.0008", "0.0134 ± 0.0006", "0.0564 ± 0.0028", "0.0141 ± 0.0015"],
                ["C3", "Virchow2", "0.9967 ± 0.0003", "0.9973 ± 0.0003", "0.9808 ± 0.0010", "0.9802 ± 0.0036", "0.9823 ± 0.0029", "0.9812 ± 0.0009", "0.0165 ± 0.0008", "0.0685 ± 0.0030", "0.0191 ± 0.0021"],
                ["C4", "UNI2-h", "0.9812 ± 0.0005", "0.9817 ± 0.0005", "0.8992 ± 0.0044", "0.9920 ± 0.0007", "0.7905 ± 0.0100", "0.8798 ± 0.0059", "0.0780 ± 0.0032", "0.2768 ± 0.0130", "0.1007 ± 0.0062"],
                ["C4", "Virchow2", "0.9561 ± 0.0019", "0.9585 ± 0.0017", "0.8430 ± 0.0057", "0.9872 ± 0.0021", "0.6726 ± 0.0137", "0.8000 ± 0.0090", "0.1203 ± 0.0044", "0.4095 ± 0.0151", "0.1394 ± 0.0083"],
            ]]),
            ("p", "Operating points, harmonized arm (mean ± SD): sensitivity at ≥95% specificity — C1 UNI2-h 0.9452 ± 0.0015 / Virchow2 0.9226 ± 0.0029; "
                  "C2 0.9929 ± 0.0008 / 0.9896 ± 0.0012; C3 0.9918 ± 0.0005 / 0.9900 ± 0.0008; C4 0.9309 ± 0.0029 / 0.8260 ± 0.0070. "
                  "Sensitivity at ≥98% specificity — C1 0.9145 / 0.8818; C2 0.9889 / 0.9827; C3 0.9872 / 0.9823; C4 0.8945 ± 0.0051 / 0.7530 ± 0.0084. "
                  "Specificity at ≥90% sensitivity — C1 0.9857 / 0.9704; C2 0.9989 / 0.9986; C3 0.9988 / 0.9973; C4 0.9774 ± 0.0021 / 0.8634 ± 0.0082."),
            ("p", "Original arm (only Virchow2 CAMELYON17 differs, batch 320): C3 Virchow2 ROC-AUC 0.9968 ± 0.0002, recall 0.9834 ± 0.0017, ECE15 0.0187 ± 0.0028; "
                  "C4 Virchow2 ROC-AUC 0.9578 ± 0.0026, PR-AUC 0.9602 ± 0.0025, accuracy 0.8513 ± 0.0122, recall 0.6911 ± 0.0279, F1 0.8125 ± 0.0183, "
                  "Brier 0.1132 ± 0.0100, log loss 0.3813 ± 0.0354, ECE15 0.1286 ± 0.0168, sensitivity at ≥95% specificity 0.8338 ± 0.0126."),
            ("p", "Dropout-0.2 sensitivity arm, C4 Virchow2: ROC-AUC 0.9621 ± 0.0030, recall 0.7035 ± 0.0232, F1 0.8220 ± 0.0151, ECE15 0.1224 ± 0.0140; "
                  "C1 UNI2-h ROC-AUC 0.9854 ± 0.0003, C1 Virchow2 0.9804 ± 0.0010, C2 UNI2-h 0.9980 ± 0.0001, C2 Virchow2 0.9971 ± 0.0002. "
                  "Full tables for all arms: revision/results/step3/seed_summary.csv."),
            ("p", "Selected epoch and temperature (harmonized): UNI2-h CAMELYON17 heads select epoch 1–2 (mean 1.3), T 1.146–1.210; UNI2-h PCam epoch 1–10 (mean 4.4), "
                  "T 1.085–1.350; Virchow2 CAMELYON17 epoch 3–10 (mean 6.8), T 1.350–1.679; Virchow2 PCam epoch 2–10 (mean 6.9), T 1.085–1.350."),
            ("h", "Results — UNI2-h minus Virchow2, two-level (slide × seed) bootstrap, mean [95% CI]; * = interval excludes 0"),
            ("t", [["Arm / cond.", "ROC-AUC", "Accuracy", "Recall", "F1", "Brier", "Log loss", "ECE15"], [
                ["Harmonized C1", "+0.0047 [+0.0013, +0.0086]*", "+0.0125 [+0.0052, +0.0219]*", "+0.0195 [+0.0002, +0.0457]*", "+0.0143 [+0.0056, +0.0261]*", "−0.0085 [−0.0148, −0.0034]*", "−0.0247 [−0.0463, −0.0078]*", "+0.0000 [−0.0198, +0.0142]"],
                ["Harmonized C2", "+0.0014 [+0.0002, +0.0039]*", "+0.0026 [−0.0042, +0.0107]", "+0.0124 [+0.0002, +0.0424]*", "+0.0034 [−0.0061, +0.0192]", "+0.0002 [−0.0056, +0.0066]", "+0.0081 [−0.0116, +0.0341]", "+0.0238 [+0.0095, +0.0467]*"],
                ["Harmonized C3", "+0.0007 [−0.0017, +0.0038]", "+0.0038 [−0.0052, +0.0143]", "−0.0114 [−0.0423, +0.0050]", "+0.0055 [−0.0075, +0.0349]", "−0.0037 [−0.0123, +0.0034]", "−0.0143 [−0.0460, +0.0125]", "−0.0065 [−0.0322, +0.0122]"],
                ["Harmonized C4", "+0.0253 [+0.0161, +0.0360]*", "+0.0564 [+0.0337, +0.0809]*", "+0.1190 [+0.0690, +0.1702]*", "+0.0811 [+0.0464, +0.1206]*", "−0.0425 [−0.0604, −0.0252]*", "−0.1329 [−0.1913, −0.0747]*", "−0.0387 [−0.0626, −0.0109]*"],
                ["Original C3", "+0.0005 [−0.0020, +0.0031]", "+0.0030 [−0.0069, +0.0143]", "−0.0126 [−0.0446, +0.0020]", "+0.0046 [−0.0099, +0.0358]", "−0.0032 [−0.0126, +0.0049]", "−0.0129 [−0.0481, +0.0181]", "−0.0062 [−0.0342, +0.0141]"],
                ["Original C4", "+0.0235 [+0.0142, +0.0343]*", "+0.0479 [+0.0147, +0.0762]*", "+0.0997 [+0.0231, +0.1583]*", "+0.0677 [+0.0180, +0.1097]*", "−0.0352 [−0.0562, −0.0072]*", "−0.1048 [−0.1747, −0.0068]*", "−0.0280 [−0.0592, +0.0180]"],
                ["Dropout02 C4", "+0.0189 [+0.0105, +0.0299]*", "+0.0405 [+0.0137, +0.0667]*", "+0.0858 [+0.0232, +0.1420]*", "+0.0572 [+0.0178, +0.0965]*", "−0.0281 [−0.0489, −0.0074]*", "−0.0773 [−0.1568, −0.0056]*", "−0.0208 [−0.0515, +0.0157]"],
            ]]),
            ("p", "Original-arm C1 and C2 are identical to harmonized (same configuration). Dropout02 C1/C2/C3: ROC-AUC differences +0.0051* / +0.0011* / +0.0002. "
                  "Full table: revision/results/step3/two_level_bootstrap.csv (also contains per-encoder two-level intervals)."),
            ("p", "Per-encoder two-level 95% intervals, harmonized, ROC-AUC: C1 UNI2-h 0.9850 [0.9784, 0.9904], Virchow2 0.9803 [0.9721, 0.9870]; "
                  "C2 0.9978 [0.9967, 0.9985], 0.9964 [0.9934, 0.9976]; C3 0.9968 [0.9921, 0.9991], 0.9962 [0.9919, 0.9987]; "
                  "C4 0.9812 [0.9743, 0.9876], 0.9558 [0.9410, 0.9696]. Recall C4: UNI2-h 0.7898 [0.7262, 0.8510], Virchow2 0.6708 [0.5818, 0.7566]."),
            ("h", "Results — ranking stability (fraction of the 100 seed pairs in which UNI2-h is better)"),
            ("t", [["Arm / cond.", "ROC-AUC", "PR-AUC", "Recall", "F1", "Brier", "ECE15", "Sens@95%spec", "Spec@90%sens"], [
                ["Harmonized C1", "1.00", "1.00", "1.00", "1.00", "1.00", "0.52", "1.00", "1.00"],
                ["Harmonized C2", "1.00", "1.00", "1.00", "0.91", "0.53", "0.00", "1.00", "0.92"],
                ["Harmonized C3", "0.95", "1.00", "0.01", "1.00", "1.00", "1.00", "1.00", "1.00"],
                ["Harmonized C4", "1.00", "1.00", "1.00", "1.00", "1.00", "1.00", "1.00", "1.00"],
                ["Original C3", "0.93", "1.00", "0.00", "0.92", "0.93", "0.92", "0.97", "1.00"],
                ["Original C4", "1.00", "1.00", "1.00", "1.00", "1.00", "0.90", "1.00", "1.00"],
                ["Dropout02 C4", "1.00", "1.00", "1.00", "1.00", "1.00", "0.91", "1.00", "1.00"],
            ]]),
            ("p", "In C4 the lowest UNI2-h ROC-AUC over 10 seeds (0.9807) is above the highest Virchow2 ROC-AUC in every arm (harmonized 0.9600, original 0.9624, "
                  "dropout02 0.9665); the lowest UNI2-h recall (0.7750) is above the highest Virchow2 recall in the harmonized arm (0.7048) but not in the original "
                  "(0.7534) or dropout02 (0.7526) arms. In C2, Virchow2 has lower ECE15 than UNI2-h in all 100 seed pairs. In C3, Virchow2 has higher recall than "
                  "UNI2-h in 99–100 of 100 pairs, while UNI2-h has higher precision."),
            ("h", "Results — position of the original single runs within the seed distribution (original arm)"),
            ("b", [
                "**UNI2-h:** all original values (C1–C4, ROC-AUC, recall, F1, Brier, ECE15) lie inside the range of the 10 repeated runs (|z| ≤ 1.0).",
                "**Virchow2 C1:** inside the range for all listed metrics.",
                "**Virchow2 C4:** original ROC-AUC 0.9502 vs repeated 0.9578 (range 0.9541–0.9624, z = −2.9); recall 0.6304 vs 0.6911 (0.6547–0.7534, z = −2.2); "
                "F1 0.7696 vs 0.8125 (0.7878–0.8530); Brier 0.1363 vs 0.1132 (0.0895–0.1251); ECE15 0.1600 vs 0.1286 (0.0886–0.1481). "
                "The original run is therefore below the repeated-run range on C4.",
                "Virchow2 C2/C3: values within or just outside the range (C2 ROC-AUC 0.9971 vs 0.9963–0.9970; C3 F1 0.9803 vs 0.9808–0.9847).",
                "**Explanation of the Virchow2 C4 position:** the original CAMELYON17 head was the epoch-10 checkpoint (selected by validation accuracy). "
                "Evaluating every repeated run at epoch 10 gives C4 ROC-AUC 0.9526–0.9578 (mean 0.9546) and recall 0.5623–0.6749; the original recall (0.6304) "
                "lies inside this range and its ROC-AUC is 0.0024 below the lowest. Later epochs transfer worse to PCam (see checkpoint-rule results), so the "
                "original run's low C4 values are mostly explained by its selected epoch.",
                "**Consequence for the submitted C4 contrast:** the paper reports UNI2-h − Virchow2 = +0.0309 ROC-AUC, +0.1537 recall, +0.1064 F1 from single runs. "
                "The repeated-run means are +0.0235 / +0.0997 / +0.0677 (original configuration) and +0.0253 / +0.1190 / +0.0811 (harmonized), all with two-level "
                "95% intervals excluding zero.",
            ]),
            ("h", "Results — checkpoint-selection rule (harmonized, mean over 10 seeds)"),
            ("t", [["Model / cond.", "Rule", "Mean epoch", "ROC-AUC", "Recall", "F1", "ECE15"], [
                ["UNI2-h C4", "val accuracy (primary)", "1.3", "0.9812", "0.7905", "0.8798", "0.1007"],
                ["UNI2-h C4", "val loss", "1.2", "0.9813", "0.7905", "0.8798", "0.1010"],
                ["UNI2-h C4", "val ROC-AUC", "1.0", "0.9814", "0.7858", "0.8771", "0.1040"],
                ["UNI2-h C4", "last epoch", "10", "0.9787", "0.7591", "0.8597", "0.1129"],
                ["Virchow2 C4", "val accuracy (primary)", "6.8", "0.9561", "0.6726", "0.8000", "0.1394"],
                ["Virchow2 C4", "val loss", "6.4", "0.9559", "0.6780", "0.8032", "0.1356"],
                ["Virchow2 C4", "val ROC-AUC", "4.6", "0.9575", "0.6474", "0.7824", "0.1519"],
                ["Virchow2 C4", "last epoch", "10", "0.9533", "0.6164", "0.7587", "0.1679"],
            ]]),
            ("p", "For C1–C3 the rule changes ROC-AUC by at most 0.0013 for either encoder. Under every rule, the lowest UNI2-h C4 ROC-AUC exceeds the highest "
                  "Virchow2 C4 ROC-AUC (val accuracy 0.9807 vs 0.9600; val ROC-AUC 0.9807 vs 0.9627; val loss 0.9807 vs 0.9657; last epoch 0.9776 vs 0.9571). "
                  "Full table: revision/results/step3/checkpoint_rule_sensitivity.csv."),
            ("h", "Diagnostic — training-loss comparison that revealed note N6"),
            ("p", "The original runs' logged training losses were lower than all 80 first-round repeats, which had dropout 0.2 active. On the same cached CAMELYON17 "
                  "training embeddings (eval mode), the original Virchow2 CAMELYON17 head has training loss 0.0191 and weight norm 5.70, versus 0.0214–0.0223 and "
                  "4.41–4.45 for three heads retrained with dropout 0.2 (C4 ROC-AUC original 0.9503 vs retrained 0.9645–0.9657). Inspection of the original code "
                  "then showed that dropout was never active (N6); no-dropout retraining reproduces the original loss curves. The first-round dropout runs are kept as "
                  "the dropout02 sensitivity arm. File: revision/results/step3/diag_trainloss.json."),
            ("h", "Files"),
            ("b", [
                "revision/scripts/step3_train_heads.py (training), revision/scripts/metrics_common.py (metric definitions), "
                "revision/scripts/step3_summarize.py (analysis), revision/scripts/build_test_clusters.py, revision/scripts/step3_diag_trainloss.py",
                "revision/results/step3/: runs.csv (240 runs × 2 test sets, all metrics), seed_summary.csv, ranking.csv, two_level_bootstrap.csv, "
                "original_vs_seeds.csv, checkpoint_rule_sensitivity.csv, diag_trainloss.json, summary.json",
                "revision/results/test_clusters/: pcam_test_clusters.csv, cam17_test_clusters.csv (slide/patient per test patch)",
                "data/revision_runs/step3/ (not in git): per-run validation/test logits for every epoch and training histories",
            ]),
        ],
    },
    {
        "title": "Step 3b — Reproduction of the manuscript's slide-cluster bootstrap",
        "date": "2026-11-14",
        "answers": "Reproducibility of the submitted confidence intervals (the original bootstrap code was not kept); R3-10; R3 (numerical consistency)",
        "blocks": [
            ("h", "What was done"),
            ("p", "The code that produced the submitted slide-cluster bootstrap intervals (Table III and the C4 paired contrasts) is not in the project files. "
                  "The procedure was re-implemented exactly as the manuscript describes it: 2,000 replicates per condition and model; each replicate samples test "
                  "slides with replacement (PCam: 129 whole-slide images from the official test metadata; CAMELYON17-WILDS: 10 WILDS slides) and includes all "
                  "retained patches of each sampled slide; metrics from temperature-scaled probabilities at threshold 0.5; 95% percentile intervals; the C4 "
                  "UNI2-h − Virchow2 contrast uses the same resampled slides for both models. It was applied to the original per-patch prediction files. "
                  "Because the original random seed is unknown, the whole procedure was run with 5 different bootstrap seeds (0, 1, 2, 3, 42)."),
            ("h", "Results"),
            ("b", [
                "All 37 interval end points printed in the manuscript (Table III: ROC-AUC, recall, F1, ECE15 for 8 model × condition cells; "
                "C4 contrasts for ROC-AUC, accuracy, recall, F1, ECE15) were compared with the reproduced end points.",
                "Median absolute difference 0.0018; maximum 0.0095 (C3 Virchow2 F1 lower bound: manuscript 0.9225, reproduced 0.9205–0.9320 across the 5 seeds). "
                "For every end point the manuscript value lies within or close to the spread produced by changing only the bootstrap seed, i.e. the differences "
                "are Monte-Carlo variation.",
                "Examples: C4 Virchow2 ROC-AUC manuscript [0.9340, 0.9650], reproduced lower 0.9332–0.9345, upper 0.9647–0.9657; "
                "C4 contrast ROC-AUC manuscript [0.0201, 0.0427], reproduced lower 0.0203–0.0210, upper 0.0423–0.0435; "
                "C4 contrast recall manuscript [0.1147, 0.1948], reproduced lower 0.1139–0.1162, upper 0.1935–0.1967; "
                "C2 UNI2-h ROC-AUC manuscript [0.9975, 0.9986], reproduced [0.9975, 0.9986–0.9987].",
                "Conclusion: the submitted slide-cluster intervals are reproducible from the saved predictions with the documented procedure, "
                "and a script now exists for them.",
                "Not reproduced here: the matched-index CNN intervals (no per-patch CNN prediction files were kept; they are regenerated in Step 4) and the "
                "stain-benchmark-disjoint subset analysis.",
            ]),
            ("h", "Files"),
            ("b", [
                "revision/scripts/reproduce_cluster_bootstrap.py",
                "revision/results/cluster_bootstrap_reproduction/intervals.csv (all metrics, all seeds), comparison_with_manuscript.csv",
            ]),
        ],
    },
]
