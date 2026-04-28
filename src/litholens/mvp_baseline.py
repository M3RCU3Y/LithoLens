"""Minimum viable FORCE-style baseline pipeline."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from litholens.impute import add_missing_indicators
from litholens.io import load_force_csv
from litholens.schema import available_curve_cols, infer_depth_col, infer_group_col, infer_target_col


@dataclass(frozen=True)
class MvpBaselineResult:
    """Artifacts and summary values from the MVP baseline."""

    target_col: str
    well_col: str
    depth_col: str
    curve_cols: list[str]
    weighted_f1: float
    confusion_matrix_path: Path
    predictions_path: Path
    model_path: Path
    heldout_well: str


def _require_column(name: str, value: str | None) -> str:
    if not value:
        raise ValueError(f"Could not infer required {name} column.")
    return value


def _feature_frame(df: pd.DataFrame, curve_cols: list[str]) -> pd.DataFrame:
    work = add_missing_indicators(df, curve_cols)
    for col in curve_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].fillna(work[col].median())
    feature_cols = curve_cols + [f"{col}_was_missing" for col in curve_cols]
    return work[feature_cols].fillna(0)


def run_mvp_baseline(
    input_path: str | Path,
    output_dir: str | Path = "reports/mvp_baseline",
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 100,
) -> MvpBaselineResult:
    """Run the smallest useful FORCE-style lithology baseline.

    This intentionally avoids dashboard, uncertainty, engineered rolling features,
    and extra model choices. It proves the core data-to-heldout-well prediction path.
    """
    df = load_force_csv(input_path)
    target_col = _require_column("target", infer_target_col(df))
    well_col = _require_column("well", infer_group_col(df))
    depth_col = _require_column("depth", infer_depth_col(df))
    curve_cols = available_curve_cols(df)
    if not curve_cols:
        raise ValueError("No supported log curves were found.")
    if df[well_col].nunique() < 2:
        raise ValueError("At least two wells are required for GroupKFold.")

    valid = df[df[target_col].notna()].copy()
    valid_with_indicators = add_missing_indicators(valid, curve_cols)
    X = _feature_frame(valid, curve_cols)
    y = valid[target_col]
    groups = valid[well_col]
    splits = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    train_idx, test_idx = next(splits.split(X, y, groups))

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predictions = model.predict(X.iloc[test_idx])
    weighted = float(f1_score(y.iloc[test_idx], predictions, average="weighted", zero_division=0))

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "random_forest_mvp.joblib"
    joblib.dump(model, model_path)

    labels = sorted(pd.unique(y))
    confusion = pd.DataFrame(
        confusion_matrix(y.iloc[test_idx], predictions, labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    confusion_path = output / "confusion_matrix.csv"
    confusion.to_csv(confusion_path)

    validation = valid_with_indicators.iloc[test_idx].copy()
    validation["actual"] = y.iloc[test_idx].to_numpy()
    validation["prediction"] = predictions
    heldout_well = str(validation[well_col].iloc[0])
    heldout = validation[validation[well_col].astype(str) == heldout_well].copy()
    predictions_path = output / f"heldout_well_{_safe_name(heldout_well)}_predictions.csv"
    heldout.to_csv(predictions_path, index=False)

    pd.Series(
        {
            "weighted_f1": weighted,
            "heldout_well": heldout_well,
            "target_col": target_col,
            "well_col": well_col,
            "depth_col": depth_col,
            "curve_cols": ",".join(curve_cols),
        }
    ).to_json(output / "metrics.json", indent=2)

    return MvpBaselineResult(
        target_col=target_col,
        well_col=well_col,
        depth_col=depth_col,
        curve_cols=curve_cols,
        weighted_f1=weighted,
        confusion_matrix_path=confusion_path,
        predictions_path=predictions_path,
        model_path=model_path,
        heldout_well=heldout_well,
    )


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
