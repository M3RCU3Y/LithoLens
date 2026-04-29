from pathlib import Path


def test_judge_docs_and_demo_references_exist():
    readme = Path("README.md").read_text(encoding="utf-8")
    app = Path("app/streamlit_app.py").read_text(encoding="utf-8")
    notebook = Path("notebooks/01_data_exploration.ipynb").read_text(encoding="utf-8")

    assert Path("docs/JUDGE_WALKTHROUGH.md").exists()
    assert "JUDGE_WALKTHROUGH.md" in readme
    assert "calibration_metrics.csv" in app
    assert "model_comparison.csv" in app
    assert "qc_range_warning" in app
    assert "FORCE Lithology Distribution" in notebook
