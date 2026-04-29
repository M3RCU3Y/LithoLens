from pathlib import Path

import pandas as pd

from litholens.mvp_baseline import run_mvp_baseline


def test_mvp_baseline_infers_columns_and_saves_one_heldout_well(tmp_path: Path):
    rows = []
    for well_idx, well in enumerate(["A", "B", "C"]):
        for i in range(12):
            lithology = 30000 if i < 6 else 65000
            rows.append(
                {
                    "Well": well,
                    "DEPT": 1000 + i,
                    "GR": 40 + i + well_idx,
                    "RHOB": 2.2 + well_idx * 0.02,
                    "NPHI": None if i == 3 else 0.15 + i * 0.001,
                    "FORCE_2020_LITHOFACIES_LITHOLOGY": lithology,
                }
            )
    input_path = tmp_path / "force_style.csv"
    output_dir = tmp_path / "reports"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    result = run_mvp_baseline(input_path=input_path, output_dir=output_dir, n_splits=3)

    assert result.target_col == "FORCE_2020_LITHOFACIES_LITHOLOGY"
    assert result.well_col == "WELL"
    assert result.depth_col == "DEPTH_MD"
    assert result.curve_cols == ["GR", "RHOB", "NPHI"]
    assert 0 <= result.weighted_f1 <= 1
    assert result.predictions_path.exists()
    assert result.fold_metrics_path.exists()
    assert result.overall_predictions_path.exists()
    assert result.summary_path.exists()
    predictions = pd.read_csv(result.predictions_path)
    fold_metrics = pd.read_csv(result.fold_metrics_path)
    summary = pd.read_json(result.summary_path, typ="series")
    assert predictions["WELL"].nunique() == 1
    assert len(fold_metrics) == 3
    assert "weighted_f1" in fold_metrics.columns
    assert "force_penalty" in fold_metrics.columns
    assert "mean_weighted_f1" in summary.index
    assert "mean_force_penalty" in summary.index
    assert {
        "prediction",
        "actual",
        "GR_was_missing",
        "NPHI_was_missing",
        "confidence",
        "margin",
        "entropy",
        "uncertainty_flag",
        "qc_warning",
        "qc_missing_curve_count",
        "qc_missing_curves",
        "qc_range_warning",
        "qc_spike_warning",
        "qc_issue",
        "review_zone",
        "explanation",
    }.issubset(predictions.columns)
    assert predictions["confidence"].between(0, 1).all()
    assert predictions["explanation"].str.contains("Predicted").all()
    assert predictions["qc_issue"].dtype == bool
