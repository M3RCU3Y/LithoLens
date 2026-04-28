# Data Dictionary

## Core Columns

| Standard Column | Meaning | Common Aliases |
| --- | --- | --- |
| `DEPTH_MD` | Measured depth | `DEPTH`, `DEPT`, `Depth` |
| `WELL` | Well identifier | `WELL_NAME`, `Well`, `well_id` |
| `FORCE_2020_LITHOFACIES_LITHOLOGY` | Lithology target label | `LITHOLOGY`, `FACIES`, `target` |

## Common Curves

| Curve | Meaning |
| --- | --- |
| `GR` | Gamma ray |
| `RHOB` | Bulk density |
| `NPHI` | Neutron porosity |
| `DTC` | Compressional sonic slowness |
| `RDEP` | Deep resistivity |
| `RMED` | Medium resistivity |
| `CALI` | Caliper |
| `PEF` | Photoelectric factor |
| `SP` | Spontaneous potential |

## Generated Columns

| Suffix / Column | Meaning |
| --- | --- |
| `_was_missing` | Original value was missing before imputation |
| `_imputed` | Value was filled by interpolation or median imputation |
| `_spike_flag` | Robust z-score spike flag |
| `_range_flag` | Conservative physical range violation flag |
| `qc_issue` | Any QC issue detected at that depth row |
| `confidence` | Maximum predicted class probability |
| `margin` | Difference between top two class probabilities |
| `entropy` | Shannon entropy over class probabilities |
| `uncertainty_flag` | Confidence/margin/entropy threshold triggered |
| `review_zone` | Uncertainty or QC suggests human review |
