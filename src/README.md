# Source helpers

Shared Python utilities used by the preprocessing path.

| File | Role |
| --- | --- |
| `preprocess.py` | Early PCam quality-filter / Macenko helper (manifest + optional H5 export) |

The production pipeline used in the manuscript lives in [`scripts/preprocess_histopath_h5.py`](../scripts/preprocess_histopath_h5.py), which applies the same ideas to both PCam and CAMELYON17 H5 layouts.
