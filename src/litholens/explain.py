"""Simple explainability helpers."""

import pandas as pd

from litholens.constants import FORCE_LITHOLOGY_LABELS


def top_feature_importances(model, feature_names: list[str], n: int = 5) -> list[tuple[str, float]]:
    """Return top global feature importances when the model exposes them."""
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []
    pairs = sorted(zip(feature_names, importances, strict=False), key=lambda item: item[1], reverse=True)
    return [(name, float(score)) for name, score in pairs[:n]]


def generate_plain_language_explanation(
    row: pd.Series,
    prediction,
    top_features: list[tuple[str, float]] | None = None,
) -> str:
    """Generate a compact explanation for one depth row."""
    label = FORCE_LITHOLOGY_LABELS.get(prediction, str(prediction))
    confidence = row.get("confidence")
    confidence_text = f" with {confidence:.2f} confidence" if pd.notna(confidence) else ""
    feature_names = [feature.replace("_", " ").lower() for feature, _ in (top_features or [])[:3]]
    drivers = ", ".join(feature_names) if feature_names else "the available log pattern"
    imputed = [col.replace("_imputed", "") for col in row.index if col.endswith("_imputed") and bool(row[col])]
    qc_note = ""
    if imputed:
        qc_note = f" {', '.join(imputed[:3])} was missing or imputed in this interval, so review is recommended."
    elif bool(row.get("review_zone", False)):
        qc_note = " This interval is in a review zone because uncertainty or QC risk is elevated."
    return f"Predicted {label}{confidence_text}. Main drivers were {drivers}.{qc_note}"


def try_shap_explanations(model, X: pd.DataFrame, max_rows: int = 200):
    """Return SHAP values when SHAP is installed; otherwise return None."""
    try:
        import shap  # type: ignore
    except ImportError:
        return None
    explainer = shap.Explainer(model, X.iloc[:max_rows])
    return explainer(X.iloc[:max_rows])
