"""End-to-end training and prediction pipelines."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

from litholens.evaluate import accuracy, macro_f1, per_class_report, weighted_f1
from litholens.explain import generate_plain_language_explanation, top_feature_importances
from litholens.features import build_feature_matrix
from litholens.impute import interpolate_short_gaps, model_or_median_impute
from litholens.io import load_force_csv, load_generic_csv
from litholens.qc import create_qc_flags, summarize_qc
from litholens.schema import available_curve_cols, infer_target_col
from litholens.split import get_holdout_wells
from litholens.train import save_model, train_baseline_random_forest, train_gradient_boosting_model
from litholens.uncertainty import predict_with_uncertainty


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML config."""
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_dataframe(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply QC and imputation before feature building."""
    data_cfg = config.get("data", {})
    qc_cfg = config.get("qc", {})
    curve_cols = available_curve_cols(df, data_cfg.get("curve_cols"))
    group_col = data_cfg.get("group_col", "WELL")
    depth_col = data_cfg.get("depth_col", "DEPTH_MD")
    work = create_qc_flags(df)
    if group_col in work.columns and depth_col in work.columns:
        work = interpolate_short_gaps(work, group_col, depth_col, curve_cols, qc_cfg.get("max_gap", 5))
    work = model_or_median_impute(work, curve_cols, group_col if group_col in work.columns else None)
    return work


def run_training_pipeline(config_path: str | Path) -> dict[str, Path | dict[str, float]]:
    """Run training, evaluation, and artifact export."""
    config = load_config(config_path)
    data_cfg = config["data"]
    output_cfg = config["outputs"]
    raw_path = Path(data_cfg["raw_path"])
    if not raw_path.exists():
        raise FileNotFoundError(f"Place the training CSV at {raw_path}")
    df = load_force_csv(raw_path)
    target_col = data_cfg.get("target_col") or infer_target_col(df)
    group_col = data_cfg.get("group_col", "WELL")
    if not target_col or target_col not in df.columns:
        raise ValueError("No lithology target column found.")
    work = prepare_dataframe(df, config)
    X = build_feature_matrix(work, config)
    y = work[target_col]
    train_idx, test_idx = get_holdout_wells(
        work,
        group_col,
        config.get("split", {}).get("test_size", 0.2),
        config.get("split", {}).get("random_state", 42),
    )
    model_cfg = config.get("model", {})
    if model_cfg.get("type", "random_forest") == "hist_gradient_boosting":
        model = train_gradient_boosting_model(
            X.iloc[train_idx], y.iloc[train_idx], model_cfg.get("random_state", 42)
        )
    else:
        model = train_baseline_random_forest(
            X.iloc[train_idx],
            y.iloc[train_idx],
            random_state=model_cfg.get("random_state", 42),
            n_estimators=model_cfg.get("n_estimators", 300),
            max_depth=model_cfg.get("max_depth"),
            class_weight=model_cfg.get("class_weight", "balanced_subsample"),
        )
    preds = model.predict(X.iloc[test_idx])
    metrics = {
        "accuracy": accuracy(y.iloc[test_idx], preds),
        "weighted_f1": weighted_f1(y.iloc[test_idx], preds),
        "macro_f1": macro_f1(y.iloc[test_idx], preds),
    }
    model_path = save_model(model, Path(output_cfg["model_dir"]) / "baseline_model.joblib")
    metrics_dir = Path(output_cfg["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.Series(metrics).to_json(metrics_dir / "baseline_metrics.json", indent=2)
    per_class_report(y.iloc[test_idx], preds).to_csv(metrics_dir / "per_class_report.csv")
    summarize_qc(work).to_csv(metrics_dir / "qc_summary.csv", index=False)
    uncertainty = predict_with_uncertainty(
        model,
        X.iloc[test_idx],
        **config.get("uncertainty", {}),
        qc_issue=work.iloc[test_idx].get("qc_issue"),
    )
    predictions = pd.concat([work.iloc[test_idx].reset_index(drop=True), uncertainty.reset_index(drop=True)], axis=1)
    top_features = top_feature_importances(model, X.columns.tolist())
    predictions["explanation"] = predictions.apply(
        lambda row: generate_plain_language_explanation(row, row["prediction"], top_features), axis=1
    )
    prediction_dir = Path(output_cfg["prediction_dir"])
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / "baseline_holdout_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    return {"model_path": model_path, "prediction_path": prediction_path, "metrics": metrics}


def run_prediction_pipeline(input_path: str | Path, model_path: str | Path, output_path: str | Path) -> Path:
    """Run predictions for an unlabeled CSV using a saved model and default config assumptions."""
    df = load_generic_csv(input_path)
    model = joblib.load(model_path)
    config = {"data": {"curve_cols": None, "group_col": "WELL", "depth_col": "DEPTH_MD"}, "features": {}}
    work = prepare_dataframe(df, config)
    X = build_feature_matrix(work, config)
    uncertainty = predict_with_uncertainty(model, X, qc_issue=work.get("qc_issue"))
    output = pd.concat([work.reset_index(drop=True), uncertainty.reset_index(drop=True)], axis=1)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)
    return path
