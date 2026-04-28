"""Minimum viable FORCE-style baseline pipeline."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from litholens.impute import add_missing_indicators
from litholens.io import load_force_csv
from litholens.schema import available_curve_cols, infer_depth_col, infer_group_col, infer_target_col

FORCE_LABEL_ORDER = [30000, 65030, 65000, 80000, 74000, 70000, 70032, 88000, 86000, 99000, 90000, 93000]


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
    overall_predictions_path: Path
    fold_metrics_path: Path
    summary_path: Path
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


def force_penalty_score(y_true: pd.Series, y_pred: pd.Series, penalty_matrix: np.ndarray) -> float:
    """Compute the average FORCE penalty using the official label order."""
    label_to_index = {label: idx for idx, label in enumerate(FORCE_LABEL_ORDER)}
    penalties = []
    for truth, pred in zip(y_true, y_pred, strict=False):
        if truth not in label_to_index or pred not in label_to_index:
            continue
        penalties.append(float(penalty_matrix[label_to_index[truth], label_to_index[pred]]))
    return float(np.mean(penalties)) if penalties else float("nan")


def _load_penalty_matrix(path: str | Path | None) -> np.ndarray | None:
    if not path:
        return None
    matrix_path = Path(path)
    return np.load(matrix_path) if matrix_path.exists() else None


def run_mvp_baseline(
    input_path: str | Path,
    output_dir: str | Path = "reports/mvp_baseline",
    n_splits: int = 5,
    random_state: int = 42,
    n_estimators: int = 100,
    penalty_matrix_path: str | Path | None = "references/force-2020-official/lithology_competition/data/penalty_matrix.npy",
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
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "random_forest_mvp.joblib"
    penalty_matrix = _load_penalty_matrix(penalty_matrix_path)
    labels = sorted(pd.unique(y))

    fold_rows = []
    validation_pieces = []
    last_model = None
    splits = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    for fold, (train_idx, test_idx) in enumerate(splits.split(X, y, groups), start=1):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight="balanced_subsample",
            random_state=random_state + fold - 1,
            n_jobs=-1,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions = model.predict(X.iloc[test_idx])
        fold_true = y.iloc[test_idx]
        fold_penalty = (
            force_penalty_score(fold_true, pd.Series(predictions), penalty_matrix)
            if penalty_matrix is not None
            else float("nan")
        )
        fold_rows.append(
            {
                "fold": fold,
                "validation_rows": len(test_idx),
                "validation_wells": int(groups.iloc[test_idx].nunique()),
                "weighted_f1": float(f1_score(fold_true, predictions, average="weighted", zero_division=0)),
                "force_penalty": fold_penalty,
            }
        )
        validation = valid_with_indicators.iloc[test_idx].copy()
        validation["fold"] = fold
        validation["actual"] = fold_true.to_numpy()
        validation["prediction"] = predictions
        validation_pieces.append(validation)
        last_model = model

    if last_model is None:
        raise ValueError("No GroupKFold splits were produced.")
    joblib.dump(last_model, model_path)

    all_predictions = pd.concat(validation_pieces).sort_index()
    confusion = pd.DataFrame(
        confusion_matrix(all_predictions["actual"], all_predictions["prediction"], labels=labels),
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    confusion_path = output / "confusion_matrix.csv"
    confusion.to_csv(confusion_path)

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics_path = output / "fold_metrics.csv"
    fold_metrics.to_csv(fold_metrics_path, index=False)

    overall_predictions_path = output / "all_oof_predictions.csv"
    all_predictions.to_csv(overall_predictions_path, index=False)

    weighted = float(fold_metrics["weighted_f1"].mean())
    mean_force_penalty = float(fold_metrics["force_penalty"].mean())
    validation = all_predictions[all_predictions["fold"] == all_predictions["fold"].min()].copy()
    heldout_well = str(validation[well_col].iloc[0])
    heldout = validation[validation[well_col].astype(str) == heldout_well].copy()
    predictions_path = output / f"heldout_well_{_safe_name(heldout_well)}_predictions.csv"
    heldout.to_csv(predictions_path, index=False)

    summary_path = output / "metrics.json"
    pd.Series(
        {
            "mean_weighted_f1": weighted,
            "mean_force_penalty": mean_force_penalty,
            "heldout_well": heldout_well,
            "target_col": target_col,
            "well_col": well_col,
            "depth_col": depth_col,
            "curve_cols": ",".join(curve_cols),
        }
    ).to_json(summary_path, indent=2)

    return MvpBaselineResult(
        target_col=target_col,
        well_col=well_col,
        depth_col=depth_col,
        curve_cols=curve_cols,
        weighted_f1=weighted,
        confusion_matrix_path=confusion_path,
        predictions_path=predictions_path,
        overall_predictions_path=overall_predictions_path,
        fold_metrics_path=fold_metrics_path,
        summary_path=summary_path,
        model_path=model_path,
        heldout_well=heldout_well,
    )


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
