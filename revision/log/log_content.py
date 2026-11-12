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
    ["R1-Q2, R3-5, R3-9", "Repeat foundation-model runs with several seeds and identical (harmonized) head settings", "1, 2, 3", "In progress"],
    ["R1-Q2", "Repeat CNN arms with several seeds; confirm raw and preprocessed inputs scaled the same way", "4", "Planned"],
    ["R3-3", "Component-wise ablation of the preprocessing pipeline", "5", "Planned"],
    ["R3-4", "Quality-control filter: class-specific effect, alternative thresholds, what is removed", "6", "Planned"],
    ["R3-7", "Macenko justification: alternative reference patches and normalizers", "7", "Planned"],
    ["R3-2, R3-12", "Patient-/slide-level analysis; limits of 10 test slides", "8", "Planned"],
    ["R3-8", "Why source-domain temperature scaling does not transfer; alternative calibration", "9", "Planned"],
    ["R3-11", "Additional baseline (ImageNet-pretrained network)", "10", "Planned"],
    ["R3-13", "False-positive / false-negative patch analysis", "11", "Planned"],
    ["R2-2, R3 (additional)", "Execution time, inference time, memory, deployment cost", "1 (+12)", "In progress"],
    ["R3-10", "Exact model versions, dependencies, code, seeds", "0, 1", "In progress"],
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
                  "The same check will be done for UNI2-h during Step 1 on Colab (UNI2-h is not cached locally)."),
            ("t", [["Item", "Value"], [
                ["Virchow2", "paige-ai/Virchow2 @ 3158645804b69e3f3bc4439d4116edddf0840a72 (verified by weight hash); 631.2 M parameters; 2560-d embedding (CLS + mean patch token)"],
                ["UNI2-h", "MahmoodLab/UNI2-h — commit to be pinned in Step 1; 1536-d CLS embedding"],
                ["Local environment", "Python 3.10.7, PyTorch 2.5.1 (CUDA 12.1), timm 1.0.31, huggingface_hub 2.4.0, NumPy 2.2.6, h5py 3.16.0, scikit-learn 1.7.2"],
                ["Colab environment", "Recorded automatically by the extraction script in extraction_meta.json"],
            ]]),
            ("h", "Suggested wording"),
            ("p", "\"Encoder weights were identified by SHA-256 fingerprint of the frozen parameters stored with each trained model; "
                  "Virchow2 corresponds to Hugging Face commit 3158645 and UNI2-h to commit [Step 1]. Code, configuration files, "
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
                  "this is why extraction runs on Colab. Colab throughput and memory figures will be added here after the run."),
            ("h", "Results (Colab)"),
            ("p", "Pending — to be filled in after the Colab run (throughput, latency, memory, UNI2-h commit)."),
            ("h", "Files"),
            ("b", [
                "revision/scripts/extract_embeddings.py, revision/scripts/emb_io.py, revision/scripts/fm_common.py",
                "revision/notebooks/colab_step1_extract_embeddings.ipynb",
                "Google Drive: GP_ECG_REVISION/embeddings/<model>/<dataset>/<split>/ (embeddings, labels, done.json with timing), "
                "<model>/extraction_meta.json, <model>/benchmark_forward.json",
            ]),
        ],
    },
]
