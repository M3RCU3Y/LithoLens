# LithoLens

**Uncertainty-aware well-log lithology prediction for the Halliburton Landmark DS365.ai Hackathon 2026.**

LithoLens is a trust-first interpretation workflow. The project starts with a FORCE-style CSV baseline that predicts lithology from common well logs using well-level validation, then grows into QC, uncertainty, explanation, and dashboard layers once the baseline is stable.

The current priority is deliberately narrow: prove that the repo can load real FORCE data, infer the required schema, train a RandomForest by well, report weighted F1 and a confusion matrix, and save predictions for one held-out well.

## What Works Now

- FORCE-style semicolon or comma CSV loading
- target, well, and depth column inference
- common log curve selection from available columns
- missing-value indicators
- well-safe `GroupKFold` validation
- full per-fold metrics and out-of-fold predictions
- official FORCE penalty score support
- RandomForest MVP baseline
- weighted F1 and confusion matrix export
- confidence, margin, entropy, uncertainty flags, QC warnings, review zones, and plain-language explanations
- one held-out well prediction CSV for the demo
- official FORCE labels and penalty-matrix location confirmed from the cloned reference

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Data

The repo keeps raw data out of Git. To prepare the FORCE training file from the cloned official reference:

```powershell
python scripts/prepare_force_data.py
```

That extracts:

```text
data/raw/force_train.csv
```

You can also place a compatible CSV there manually. Required columns are flexible because LithoLens normalizes common mnemonics such as `DEPTH`, `DEPT`, `DEPTH_MD`, `WELL`, `GR`, `RHOB`, `NPHI`, `DTC`, `RDEP`, `RMED`, `CALI`, `PEF`, `SP`, `LITHOLOGY`, and `FORCE_2020_LITHOFACIES_LITHOLOGY`.

## Run Baseline

```powershell
python scripts/run_baseline.py --config configs/baseline.yaml
```

Outputs are written to:

```text
reports/mvp_baseline/random_forest_mvp.joblib
reports/mvp_baseline/metrics.json
reports/mvp_baseline/confusion_matrix.csv
reports/mvp_baseline/heldout_well_<name>_predictions.csv
```

Recent local FORCE smoke result with 50 trees:

```text
Held-out well: 15/9-13
Weighted F1: 0.6851
```

## Make Demo Outputs

```powershell
python scripts/make_demo_outputs.py --config configs/baseline.yaml
```

This trains on a synthetic mini dataset if no FORCE CSV is available, then creates a demo prediction CSV.

## Dashboard

```powershell
streamlit run app/streamlit_app.py
```

The dashboard opens the MVP held-out well artifact by default and shows log tracks, predicted lithology, confidence, uncertainty, QC review zones, explanations, and fold metrics.

## Tests

```powershell
pytest
```

## Repo Structure

```text
configs/        Reproducible YAML configs
data/           Local raw and processed data placeholders
docs/           Roadmap, attribution, data dictionary, model card
notebooks/      Exploration and demo notebooks
src/litholens/  Production Python package
app/            Streamlit dashboard
tests/          Unit tests for core behavior
scripts/        CLI entry points and reference clone helper
references/     External repos cloned for study only
reports/        Generated models, metrics, predictions
```

## Current Limitations

- CSV ingestion is implemented first; LAS parsing is not wired yet.
- FORCE penalty scoring depends on the official `12 x 12` matrix in `references/force-2020-official/lithology_competition/data/penalty_matrix.npy`.
- SHAP explanations are optional and only used if the package is installed.
- The baseline model is intentionally simple so the team can trust the data split and artifact flow before adding complexity.

## Roadmap

See [docs/ENGINEERING_ROADMAP.md](docs/ENGINEERING_ROADMAP.md) for the detailed engineering plan.
