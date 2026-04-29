# LithoLens

> Trust-aware lithology prediction for FORCE-style well logs.

LithoLens is a hackathon-ready interpretation workflow for the Halliburton Landmark DS365.ai Hackathon 2026. It predicts lithology from well-log data, then tells the interpreter how much to trust each prediction using well-level validation, confidence, uncertainty, QC warnings, review zones, and plain-language explanations.

The point is not just to classify rock type. The point is to produce a workflow a geoscientist can inspect, challenge, and safely use.

## Current Demo Snapshot

| Area | Status |
| --- | --- |
| Dataset | FORCE 2020 training CSV |
| Validation | 5-fold GroupKFold by well |
| Model | RandomForest baseline |
| Demo well | `15/9-13` |
| Mean weighted F1 | `0.6513` |
| Mean FORCE penalty | `0.8481` |
| Dashboard | Streamlit MVP |
| Large artifacts | Git LFS |

## Why This Exists

Lithology prediction from logs can move fast, but black-box predictions are hard to trust in interpretation work. LithoLens wraps the baseline model with the evidence a reviewer needs:

- What lithology was predicted?
- How confident was the model?
- Was the interval uncertain?
- Were curves missing or imputed?
- Which features broadly drove the prediction?
- Should a human review this zone?

## What Works Now

- FORCE-style semicolon or comma CSV loading
- automatic target, well, and depth column detection
- common log curve selection from available columns
- missing-value indicators for every selected curve
- well-safe `GroupKFold` validation
- full per-fold metrics and out-of-fold predictions
- official FORCE penalty score support
- RandomForest baseline model
- aggregate confusion matrix export
- confidence, probability margin, entropy, and uncertainty flags
- QC warnings and review-zone labels
- feature-importance explanation text
- Streamlit demo for the held-out well
- real FORCE data and model artifacts tracked with Git LFS

## Repository Map

```text
LithoLens/
  app/                    Streamlit demo
  configs/                Reproducible YAML configs
  data/raw/               FORCE training data, tracked with Git LFS
  docs/                   Roadmap, model card, attribution, demo notes
  notebooks/              Exploration notebooks
  references/             External repos for study only
  reports/mvp_baseline/   Metrics, predictions, model artifacts
  scripts/                CLI helpers
  src/litholens/          Reusable Python package
  tests/                  Unit and smoke tests
```

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Run the test suite:

```powershell
pytest
```

Run the baseline:

```powershell
python scripts/run_baseline.py --config configs/baseline.yaml --input data/raw/force_train.csv --output-dir reports/mvp_baseline --trees 50
```

Launch the demo:

```powershell
streamlit run app/streamlit_app.py
```

## Demo Artifacts

| Artifact | Purpose |
| --- | --- |
| `reports/mvp_baseline/metrics.json` | aggregate MVP metrics |
| `reports/mvp_baseline/fold_metrics.csv` | per-fold weighted F1 and FORCE penalty |
| `reports/mvp_baseline/calibration_metrics.csv` | confidence calibration bins |
| `reports/mvp_baseline/model_comparison.csv` | first-fold baseline model comparison |
| `reports/mvp_baseline/confusion_matrix.csv` | aggregate confusion matrix |
| `reports/mvp_baseline/all_oof_predictions.csv` | all out-of-fold predictions, tracked with Git LFS |
| `reports/mvp_baseline/heldout_well_15_9-13_predictions.csv` | Streamlit demo well |
| `reports/mvp_baseline/random_forest_mvp.joblib` | trained MVP model, tracked with Git LFS |

## Prediction Output

Each demo prediction row includes:

| Column | Meaning |
| --- | --- |
| `prediction` | predicted FORCE lithology label |
| `actual` | ground-truth FORCE lithology label |
| `confidence` | maximum predicted class probability |
| `margin` | top probability minus second probability |
| `entropy` | probability distribution uncertainty |
| `uncertainty_flag` | low-confidence or ambiguous prediction |
| `qc_warning` | missing/imputed curve warning |
| `review_zone` | interval should be reviewed by a human |
| `explanation` | short interpretation note |

## Data Notes

The FORCE training CSV and model artifacts are committed through Git LFS so teammates can clone the repo and run the demo without rebuilding everything from scratch.

If the raw CSV needs to be regenerated from the official reference clone:

```powershell
python scripts/prepare_force_data.py
```

The loader normalizes common mnemonics such as `DEPTH`, `DEPT`, `DEPTH_MD`, `WELL`, `GR`, `RHOB`, `NPHI`, `DTC`, `RDEP`, `RMED`, `CALI`, `PEF`, `SP`, `LITHOLOGY`, and `FORCE_2020_LITHOFACIES_LITHOLOGY`.

## Project Philosophy

LithoLens is intentionally simple at the model layer and careful at the workflow layer. For a hackathon demo, the strongest story is:

1. Validate by well, not by random rows.
2. Show the model's score honestly.
3. Surface uncertainty instead of hiding it.
4. Preserve QC context beside every prediction.
5. Keep the interpreter in control.

## Current Limitations

- CSV ingestion is stable first; LAS ingestion is still future work.
- Confidence is not calibrated yet.
- Explanations use global RandomForest feature importances, not SHAP.
- QC currently emphasizes missing/imputed curves in the MVP artifact.
- The baseline is not tuned for leaderboard performance yet.

## Next Build Targets

1. Add probability calibration.
2. Add spike, range, and missing-interval QC directly into the MVP artifact.
3. Add SHAP or another local explanation method.
4. Compare RandomForest with HistGradientBoosting and LightGBM.
5. Add LAS ingestion once the CSV workflow stays stable.
6. Polish the Streamlit judge path around one clean story well.

## Reference Docs

- [Engineering roadmap](docs/ENGINEERING_ROADMAP.md)
- [Judge walkthrough](docs/JUDGE_WALKTHROUGH.md)
- [Demo notes](docs/DEMO_NOTES.md)
- [Model card](docs/MODEL_CARD.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Third-party attribution](docs/THIRD_PARTY_ATTRIBUTION.md)
