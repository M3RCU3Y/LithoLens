# Hackathon Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden LithoLens from working MVP into a stronger hackathon demo by adding richer QC, calibration evidence, model comparison, judge-facing docs, and Streamlit demo polish.

**Architecture:** Keep `src/litholens/mvp_baseline.py` as the runnable baseline entry point and add small helper functions for QC, calibration, and optional model comparison. Keep generated artifacts under `reports/mvp_baseline/`, documentation under `docs/`, and dashboard changes isolated to `app/streamlit_app.py`.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, Plotly, Streamlit, pytest, Git LFS.

---

### Task 1: Rich QC Outputs

**Files:**
- Modify: `src/litholens/mvp_baseline.py`
- Test: `tests/test_mvp_baseline.py`

- [ ] Add assertions that held-out prediction CSV includes `qc_missing_curve_count`, `qc_missing_curves`, `qc_range_warning`, `qc_spike_warning`, and `qc_issue`.
- [ ] Run `pytest tests/test_mvp_baseline.py -q` and verify failure from missing columns.
- [ ] Implement conservative range flags, robust spike flags, and row-level QC summaries in the MVP output.
- [ ] Rerun test and commit.

### Task 2: Calibration Metrics

**Files:**
- Modify: `src/litholens/mvp_baseline.py`
- Test: `tests/test_calibration.py`

- [ ] Add tests for `expected_calibration_error` with a perfect-confidence case and mixed-confidence case.
- [ ] Run test and verify import failure.
- [ ] Implement ECE and Brier-style accuracy confidence summary.
- [ ] Export `calibration_metrics.csv` and include `mean_ece` in `metrics.json`.
- [ ] Rerun tests and commit.

### Task 3: Model Comparison

**Files:**
- Modify: `src/litholens/mvp_baseline.py`
- Test: `tests/test_model_comparison.py`

- [ ] Add a small synthetic test that runs `compare_models=True` and expects `model_comparison.csv` with RandomForest and HistGradientBoosting rows.
- [ ] Run test and verify failure.
- [ ] Implement optional model comparison on the first fold only for runtime control.
- [ ] Rerun tests and commit.

### Task 4: Judge Walkthrough, Notebook, And Demo Polish

**Files:**
- Modify: `app/streamlit_app.py`
- Modify: `README.md`
- Modify: `docs/MODEL_CARD.md`
- Create: `docs/JUDGE_WALKTHROUGH.md`
- Modify: `notebooks/01_data_exploration.ipynb`
- Test: `tests/test_docs_and_demo.py`

- [ ] Add tests that check judge walkthrough exists, README references it, app references calibration/model comparison artifacts, and notebook has real starter cells.
- [ ] Run test and verify failure.
- [ ] Update Streamlit with calibration/model comparison tabs and richer QC table.
- [ ] Add judge walkthrough and useful EDA notebook starter cells.
- [ ] Update README/model card to reflect new artifacts.
- [ ] Rerun tests and commit.

### Task 5: Refresh Real Artifacts And PR

**Files:**
- Modify: `reports/mvp_baseline/*`

- [ ] Run `pytest -q`.
- [ ] Run `python -m compileall -q src app scripts`.
- [ ] Run `python scripts/run_baseline.py --config configs/baseline.yaml --input data/raw/force_train.csv --output-dir reports/mvp_baseline --trees 50 --compare-models`.
- [ ] Commit refreshed artifacts.
- [ ] Push branch and open a draft PR.

## Self-Review

- Spec coverage: covers QC, calibration, model comparison, Streamlit polish, judge walkthrough, notebook cleanup, model card, refreshed artifacts, branch, commits, and PR.
- Placeholder scan: no unbounded TODOs; each task names files and expected tests.
- Type consistency: all new artifacts are stable CSV/JSON files consumed by app/docs.
