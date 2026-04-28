import pandas as pd

from litholens.features import add_rolling_features
from litholens.impute import add_missing_indicators


def test_missing_indicators_are_created():
    df = pd.DataFrame({"GR": [1.0, None, 3.0]})
    out = add_missing_indicators(df, ["GR"])
    assert "GR_was_missing" in out.columns
    assert "GR_imputed" in out.columns
    assert out["GR_was_missing"].tolist() == [False, True, False]


def test_rolling_features_do_not_leak_across_wells():
    df = pd.DataFrame(
        {
            "WELL": ["A", "A", "B", "B"],
            "DEPTH_MD": [1, 2, 1, 2],
            "GR": [10.0, 10.0, 100.0, 100.0],
        }
    )
    out = add_rolling_features(df, "WELL", "DEPTH_MD", ["GR"], windows=[3])
    assert out.loc[df["WELL"] == "A", "GR_roll_mean_3"].eq(10.0).all()
    assert out.loc[df["WELL"] == "B", "GR_roll_mean_3"].eq(100.0).all()
