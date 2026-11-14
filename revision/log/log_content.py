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
          "Each step states **what was done**, **why** (which reviewer comment it answers), **how**, **results**, "
          "**suggested wording for the paper / response**, and **where the files are**."),
    ("p", "All code for the revision lives in the repository folder revision/ (scripts, Colab notebooks, results). "
          "Numbers in this log are copied from result files that the scripts write; they are never typed by hand."),
]

# Reviewer comment -> step -> status. Status: Planned / In progress / Done / Writing only (for the professor)
TRACKER = [
    ["Reviewer & comment", "What is needed", "Step", "Status"],
    ["R1-Q2, R3-5, R3-9", "Repeat foundation-model runs with several seeds and identical (harmonized) head settings", "1, 2, 3", "Steps 1–2 done; Step 3 next"],
    ["R1-Q2", "Repeat CNN arms with several seeds; confirm raw and preprocessed inputs scaled the same way", "4", "Planned"],
    ["R3-3", "Component-wise ablation of the preprocessing pipeline", "5", "Planned"],
    ["R3-4", "Quality-control filter: class-specific effect, alternative thresholds, what is removed", "6", "Planned"],
    ["R3-7", "Macenko justification: alternative reference patches and normalizers", "7", "Planned"],
    ["R3-2, R3-12", "Patient-/slide-level analysis; limits of 10 test slides", "8", "Planned"],
    ["R3-8", "Why source-domain temperature scaling does not transfer; alternative calibration", "9", "Planned"],
    ["R3-11", "Additional baseline (ImageNet-pretrained network)", "10", "Planned"],
    ["R3-13", "False-positive / false-negative patch analysis", "11", "Planned"],
    ["R2-2, R3 (additional)", "Execution time, inference time, memory, deployment cost", "1 (+12)", "Done for foundation models; CNN in Step 4"],
    ["R3-10", "Exact model versions, dependencies, code, seeds", "0, 1, 2", "Model versions done; seeds in Step 3"],
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
        "**N3 — Not experimental:** R3-1 (add an independent institution) is not part of the current experiment plan; "
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
            ("h", "Suggested wording"),
            ("p", "\"Encoder weights were identified by SHA-256 fingerprint of the frozen parameters stored with each trained model; "
                  "Virchow2 corresponds to Hugging Face commit 3158645 and UNI2-h to commit d517a8d. Code, configuration files, "
                  "seeds and per-patch predictions are released with the paper.\""),
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
            ("h", "Suggested wording (execution time, R2-2 / R3)"),
            ("p", "\"Encoding one 96×96 patch (upsampled to 224×224) takes about 12 ms on an NVIDIA L4 in half precision at batch size 256 "
                  "(UNI2-h 12.0 ms, Virchow2 11.5 ms; 27 ms and 26 ms on an NVIDIA T4), with a peak GPU memory of 6.8 GB and 5.9 GB respectively. "
                  "Because the encoders are frozen, embeddings are computed once (about 2.25 h per model for all 676,438 patches on one L4); "
                  "training the linear head then takes under a minute.\""),
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
            ("h", "Suggested wording"),
            ("p", "\"Because the encoders are frozen, patch embeddings were computed once (fp16, NVIDIA L4) and cached. Applying the originally "
                  "trained heads to the cached embeddings reproduced all reported test ROC-AUC and accuracy values to within 0.0001, "
                  "and embeddings were reproducible across GPU types (mean cosine similarity > 0.99999).\""),
            ("h", "Files"),
            ("b", [
                "revision/scripts/step2_equivalence.py → revision/results/step2/equivalence.json",
                "revision/scripts/step2_cross_gpu.py → revision/results/step2/cross_gpu_t4_vs_l4.json",
                "revision/results/step2/original_heads/*.npz (original linear heads + temperatures, extracted from the original checkpoints)",
            ]),
        ],
    },
]
