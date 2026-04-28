# LithoLens Engineering Roadmap

## MVP Definition

The MVP is a reproducible Python workflow that ingests FORCE-style or generic CSV well-log data, standardizes common curve mnemonics, preserves QC and imputation flags, trains a tree-based lithology classifier with well-level validation, and exports predictions with confidence, uncertainty, review-zone flags, QC warnings, and plain-language explanations. A lightweight Streamlit dashboard must show raw logs, predicted lithology, confidence, uncertainty/review zones, QC warnings, and a selected-depth explanation panel.

## Repo Architecture

- `src/litholens/`: production package for ingestion, QC, imputation, features, splitting, training, evaluation, uncertainty, explanations, plots, and pipelines.
- `configs/`: YAML configs so DS365.ai notebooks and scripts run the same workflow.
- `scripts/`: CLI entry points for baseline training, demo outputs, and safe reference cloning.
- `notebooks/`: exploration only; reusable logic stays in `src/`.
- `app/`: Streamlit dashboard for hackathon demonstration.
- `docs/`: roadmap, attribution, data dictionary, and model card.
- `data/`: local data placeholders, ignored by Git except `.gitkeep`.
- `references/`: cloned external repos for study only; ignored by Git except `.gitkeep`.
- `reports/`: generated models, metrics, and predictions.

## Milestones

1. Bootstrap reproducible repo scaffold with config, scripts, package modules, docs, and tests.
2. Load FORCE 2020 training data and produce an EDA notebook with curve coverage and lithology distribution.
3. Train a baseline model using well-level holdout and GroupKFold validation.
4. Add QC and imputation flags without deleting suspect intervals.
5. Add uncertainty scoring, review zones, and explanation text.
6. Build Streamlit demo around a held-out well.
7. Adapt paths and execution to DS365.ai, then harden demo assets.

## Tasks By Phase

### Phase 0: Repo Setup + Reference Audit

- Create package structure, config, docs, scripts, tests, notebooks, and placeholders.
- Clone external repositories into `references/` only.
- Record third-party use in `docs/THIRD_PARTY_ATTRIBUTION.md`.
- Confirm baseline scripts run without absolute paths.

### Phase 1: FORCE Data Ingestion + EDA

- Download/place FORCE CSV under `data/raw/`.
- Validate column normalization and target detection.
- Build EDA notebook for log availability, curve ranges, missing intervals, class balance, and per-well coverage.
- Confirm lithology labels and optional penalty matrix from FORCE references.

### Phase 2: Baseline Model With Well-Level Validation

- Build feature matrix from raw curves plus simple missingness indicators.
- Split by well, not rows.
- Train RandomForest baseline and compare HistGradientBoosting.
- Export metrics, confusion matrix, per-class report, and predictions.

### Phase 3: QC + Imputation + Feature Engineering

- Add missing curve and missing interval reports.
- Detect spikes and physically implausible readings with conservative configurable ranges.
- Interpolate short within-well gaps and median-impute remaining gaps.
- Preserve measured-vs-imputed indicators.
- Add rolling, gradient, and cross-log ratio features within well boundaries.

### Phase 4: Uncertainty + Review Zones

- Compute confidence, probability margin, entropy, and uncertainty flags.
- Mark review zones when uncertainty or QC risk is present.
- Calibrate thresholds on validation wells.

### Phase 5: Explainability

- Start with feature importances and local feature values.
- Add SHAP when installed and stable.
- Generate plain-language explanations that mention confidence, dominant drivers, and missing/imputed data.

### Phase 6: Streamlit Dashboard

- Add upload/sample selector and well selector.
- Show log curves, lithology track, confidence track, uncertainty/review zones, QC warnings, and explanation panel.
- Add metrics summary when labels are available.

### Phase 7: DS365.ai Adaptation + Final Demo Hardening

- Validate scripts and notebooks in DS365.ai.
- Package demo data, outputs, and screenshots.
- Freeze config, document limitations, and rehearse judge flow.

## Assumptions

- FORCE 2020 CSV data is the initial benchmark.
- Final scoring uses well-level validation and optionally FORCE penalty scoring if the matrix is confirmed.
- LAS ingestion can be added after CSV workflow is validated.
- Tree-based models are adequate for the first demo.
- DS365.ai can run Python scripts and notebooks with standard scientific Python dependencies.

## Risks

- Curve mnemonic drift across datasets can create silent feature gaps.
- Random row splits would inflate metrics, so final reporting must stay well-based.
- Imputation can hide data problems unless flags are preserved.
- Class imbalance may make rare lithologies look worse than headline accuracy suggests.
- SHAP or heavier models may be too slow or brittle for hackathon timing.

## Dataset Plan

Use FORCE 2020 for baseline development because it provides many wells, lithology labels, and a public competition framing. Keep raw files out of Git. Produce processed feature/prediction artifacts in `reports/` or `data/processed/` only when needed. Later, test with DS365.ai-accessible data or user-provided LAS/CSV exports.

## Model Plan

Start with `RandomForestClassifier` using class weighting, then compare `HistGradientBoostingClassifier`. Use GroupKFold and holdout wells. Save models with joblib. Add optional LightGBM/XGBoost only after the baseline, QC, uncertainty, and dashboard flow are stable.

## Dashboard Plan

Streamlit remains simple and reliable: a file uploader, well selector, depth tracks, QC table, metrics table, and selected-depth explanation. Plotly is used for interactive, dashboard-friendly depth plots.

## Testing Plan

Unit tests cover spike detection, missing indicators, entropy behavior, group split isolation, and feature engineering boundaries. Script-level smoke tests use synthetic data so the repo remains testable before FORCE data is added.

## Safe Use Of External Repos

Reference repositories are cloned into `references/` only and are ignored by Git. Their code is used for study of data format, labels, scoring, validation patterns, uncertainty ideas, visualization patterns, and benchmark comparison. No files are copied directly into `src/` without attribution and review.
