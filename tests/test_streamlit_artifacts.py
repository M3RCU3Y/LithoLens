from pathlib import Path


def test_streamlit_app_targets_mvp_demo_artifacts():
    source = Path("app/streamlit_app.py").read_text(encoding="utf-8")

    assert "reports/mvp_baseline" in source
    assert "fold_metrics.csv" in source
    assert "review_zone" in source
    assert "explanation" in source
    assert "heldout_well" in source
