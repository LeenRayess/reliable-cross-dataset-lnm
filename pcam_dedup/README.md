# PCam deduplication indices

Exact-content (pixel-hash) deduplication of PatchCamelyon.

| File | Role |
| --- | --- |
| `manifest.json` | Options and summary of the dedup run |
| `train_kept_indices.npy` | Training indices retained |
| `valid_kept_indices.npy` | Validation indices retained |
| `test_kept_indices.npy` | Test indices retained |

Regenerate with [`scripts/dedup_pcam.py`](../scripts/dedup_pcam.py) if you start from a fresh PCam download. Downstream preprocessing reads these indices so duplicate tiles cannot inflate in-domain or transfer scores.
