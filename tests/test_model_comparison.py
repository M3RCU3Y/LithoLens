from pathlib import Path

import pandas as pd

from litholens.mvp_baseline import run_mvp_baseline


def test_mvp_baseline_can_export_model_comparison(tmp_path: Path):
    rows = []
    for well_idx, well in enumerate(["A", "B", "C"]):
        for i in range(10):
            lithology = 30000 if i < 5 else 65000
            rows.append(
                {
                    "WELL": well,
                    "DEPTH_MD": 1000 + i,
                    "GR": 35 + i + well_idx,
                    "RHOB": 2.2 + well_idx * 0.03,
                    "FORCE_2020_LITHOFACIES_LITHOLOGY": lithology,
                }
            )
    input_path = tmp_path / "force.csv"
    pd.DataFrame(rows).to_csv(input_path, index=False)

    result = run_mvp_baseline(
        input_path=input_path,
        output_dir=tmp_path / "reports",
        n_splits=3,
        n_estimators=5,
        compare_models=True,
    )

    assert result.model_comparison_path is not None
    comparison = pd.read_csv(result.model_comparison_path)
    assert set(comparison["model"]) == {"RandomForest", "HistGradientBoosting"}
