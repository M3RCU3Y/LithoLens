# Evaluation Uncertainty Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the MVP LithoLens baseline into a credible hackathon evaluation and demo workflow with FORCE penalty score, full GroupKFold metrics, uncertainty/review-zone outputs, QC flags, explanations, Streamlit demo, and final documentation.

**Architecture:** Keep the baseline centered in `src/litholens/mvp_baseline.py`, adding small helper functions rather than a second pipeline. Reuse existing modules for QC, uncertainty, explainability, and plotting where possible, and export judge-friendly artifacts under `reports/mvp_baseline/`. The Streamlit app should consume those artifacts directly.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn RandomForest, Plotly, Streamlit, joblib, pytest, Git LFS for large data/model artifacts.

---

### Task 1: FORCE Penalty Scoring

**Files:**
- Modify: `src/litholens/mvp_baseline.py`
- Test: `tests/test_force_penalty.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_force_penalty.py`:

```python
import numpy as np
import pandas as pd

from litholens.mvp_baseline import force_penalty_score


def test_force_penalty_score_uses_force_label_order():
    matrix = np.arange(144).reshape(12, 12)
    truth = pd.Series([30000, 65000, 93000])
    pred = pd.Series([65030, 65000, 30000])

    score = force_penalty_score(truth, pred, matrix)

    expected = (matrix[0, 1] + matrix[2, 2] + matrix[11, 0]) / 3
    assert score == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_force_penalty.py -q`
Expected: FAIL importing `force_penalty_score`.

- [ ] **Step 3: Implement penalty scoring**

Add a `FORCE_LABEL_ORDER` list and `force_penalty_score(y_true, y_pred, penalty_matrix)` helper in `mvp_baseline.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_force_penalty.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/litholens/mvp_baseline.py tests/test_force_penalty.py && git commit -m "add FORCE penalty scoring"`

### Task 2: Full GroupKFold Metrics

**Files:**
- Modify: `src/litholens/mvp_baseline.py`
- Test: `tests/test_mvp_baseline.py`

- [ ] **Step 1: Write the failing test**

Extend `tests/test_mvp_baseline.py` to assert that `fold_metrics_path`, `overall_predictions_path`, and `summary_path` exist, `fold_metrics.csv` contains one row per fold, and `metrics.json` includes `mean_weighted_f1` and `mean_force_penalty`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mvp_baseline.py -q`
Expected: FAIL because result fields and output files do not exist.

- [ ] **Step 3: Implement full fold loop**

Refactor `run_mvp_baseline` to iterate across all GroupKFold folds. Train one RandomForest per fold, collect per-row out-of-fold predictions, save `fold_metrics.csv`, `all_oof_predictions.csv`, `metrics.json`, and keep saving one held-out well file for demo. Save the final demo fold model to `random_forest_mvp.joblib`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mvp_baseline.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/litholens/mvp_baseline.py tests/test_mvp_baseline.py && git commit -m "report all GroupKFold metrics"`

### Task 3: Uncertainty, QC Flags, And Explanations

**Files:**
- Modify: `src/litholens/mvp_baseline.py`
- Test: `tests/test_mvp_baseline.py`

- [ ] **Step 1: Write the failing test**

Extend `tests/test_mvp_baseline.py` to assert prediction outputs include `confidence`, `margin`, `entropy`, `uncertainty_flag`, `review_zone`, `qc_warning`, and `explanation`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mvp_baseline.py -q`
Expected: FAIL due to missing columns.

- [ ] **Step 3: Implement outputs**

Use `predict_proba` to compute confidence, margin, and entropy. Mark `uncertainty_flag` from thresholds. Build `qc_warning` from missing indicators. Mark `review_zone` when uncertain or QC-warning. Add short feature-importance explanations using global top features and row-level QC context.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mvp_baseline.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add src/litholens/mvp_baseline.py tests/test_mvp_baseline.py && git commit -m "add uncertainty and explanation outputs"`

### Task 4: Streamlit Demo And Documentation

**Files:**
- Modify: `app/streamlit_app.py`
- Modify: `README.md`
- Modify: `docs/MODEL_CARD.md`
- Create: `docs/DEMO_NOTES.md`
- Test: `tests/test_streamlit_artifacts.py`

- [ ] **Step 1: Write the failing test**

Create a test that reads `app/streamlit_app.py` and asserts it points to `reports/mvp_baseline`, references `fold_metrics.csv`, and includes explanation/review-zone affordances.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streamlit_artifacts.py -q`
Expected: FAIL against current app.

- [ ] **Step 3: Update demo UI and docs**

Make Streamlit default to `reports/mvp_baseline/heldout_well_15_9-13_predictions.csv`, show fold metrics if present, and add tabs for tracks, QC/review, explanations, and metrics. Update README, model card, and demo notes with current limitations and judge flow.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streamlit_artifacts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add app/streamlit_app.py README.md docs/MODEL_CARD.md docs/DEMO_NOTES.md tests/test_streamlit_artifacts.py && git commit -m "polish MVP demo documentation"`

### Task 5: Real FORCE Artifact Refresh And PR

**Files:**
- Modify generated artifacts under `reports/mvp_baseline/`
- Modify model LFS pointer if retrained

- [ ] **Step 1: Run full validation**

Run:

```powershell
pytest -q
python -m compileall -q src app scripts
python scripts/run_baseline.py --config configs/baseline.yaml --input data/raw/force_train.csv --output-dir reports/mvp_baseline --trees 50
```

- [ ] **Step 2: Commit refreshed artifacts**

Run: `git add reports/mvp_baseline && git commit -m "refresh FORCE evaluation artifacts"`

- [ ] **Step 3: Push branch and open PR**

Run:

```powershell
git push -u origin codex/eval-uncertainty-demo
gh pr create --draft --base main --head codex/eval-uncertainty-demo --title "[codex] add FORCE evaluation and demo workflow" --body-file <body.md>
```

Expected: draft PR URL.

## Self-Review

- Spec coverage: plan covers FORCE penalty score, all GroupKFold fold metrics, uncertainty outputs, QC flags, Streamlit demo, feature-importance explanations, final model card/demo docs, branch commits, and PR.
- Placeholder scan: no `TBD`, no vague "add tests" without concrete test content.
- Type consistency: `MvpBaselineResult` will gain path fields referenced by tests; `run_mvp_baseline` remains the central entry point; artifact names are stable and consumed by Streamlit.
