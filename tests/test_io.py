from pathlib import Path

from litholens.io import load_force_csv


def test_load_force_csv_auto_detects_semicolon_separator(tmp_path: Path):
    csv_path = tmp_path / "force.csv"
    csv_path.write_text(
        "Well;DEPT;GR;FORCE_2020_LITHOFACIES_LITHOLOGY\nA;1.0;80;65000\n",
        encoding="utf-8",
    )

    df = load_force_csv(csv_path)

    assert df.columns.tolist() == ["WELL", "DEPTH_MD", "GR", "FORCE_2020_LITHOFACIES_LITHOLOGY"]
    assert df.loc[0, "GR"] == 80
