# LithoLens

LithoLens is an uncertainty-aware well-log interpretation assistant for the Halliburton Landmark DS365.ai Hackathon 2026. It predicts lithology/facies from well-log data and keeps the geoscientist in control by showing confidence, uncertainty, QC warnings, measured-vs-imputed status, and plain-language prediction explanations.

The first scaffold targets public FORCE 2020-style CSV data, generic well-log CSVs, and notebook-friendly Python workflows. LAS loading is planned after the CSV baseline is stable.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

## Data

Place the FORCE 2020 training CSV or a compatible well-log CSV under `data/raw/`. The default config expects:

```text
data/raw/force_train.csv
```

Required columns are flexible because LithoLens normalizes common mnemonics such as `DEPTH`, `DEPT`, `DEPTH_MD`, `WELL`, `GR`, `RHOB`, `NPHI`, `DTC`, `RDEP`, `RMED`, `CALI`, `PEF`, `SP`, `LITHOLOGY`, and `FORCE_2020_LITHOFACIES_LITHOLOGY`.

## Run Baseline

```powershell
python scripts/run_baseline.py --config configs/baseline.yaml
```

Outputs are written to:

```text
reports/models/
reports/metrics/
reports/predictions/
```

## Make Demo Outputs

```powershell
python scripts/make_demo_outputs.py --config configs/baseline.yaml
```

This trains on a synthetic mini dataset if no FORCE CSV is available, then creates a demo prediction CSV.

## Launch Dashboard

```powershell
streamlit run app/streamlit_app.py
```

Upload a prediction CSV or select the latest file from `reports/predictions/`.

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

- CSV ingestion is implemented first; LAS parsing is not yet wired.
- FORCE penalty scoring is a placeholder until the exact matrix is copied from the reference material.
- SHAP explanations are optional and only used if the package is installed.
- The baseline model is intentionally simple so QC, uncertainty, and dashboard plumbing can be tested early.

## Roadmap

See [docs/ENGINEERING_ROADMAP.md](docs/ENGINEERING_ROADMAP.md) for the detailed engineering plan.
