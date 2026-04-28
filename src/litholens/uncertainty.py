"""Prediction uncertainty helpers."""

import numpy as np
import pandas as pd


def shannon_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Compute Shannon entropy for each probability row."""
    probs = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(probs * np.log(probs), axis=1)


def predict_with_uncertainty(
    model,
    X: pd.DataFrame,
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
    entropy_threshold: float = 1.25,
    qc_issue: pd.Series | None = None,
) -> pd.DataFrame:
    """Predict classes and add confidence, margin, entropy, and review flags."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for uncertainty scoring.")
    probabilities = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    order = np.argsort(probabilities, axis=1)
    top_idx = order[:, -1]
    second_idx = order[:, -2] if probabilities.shape[1] > 1 else order[:, -1]
    confidence = probabilities[np.arange(len(probabilities)), top_idx]
    second = probabilities[np.arange(len(probabilities)), second_idx]
    margin = confidence - second
    entropy = shannon_entropy(probabilities)
    predictions = classes[top_idx]
    uncertainty_flag = (
        (confidence < confidence_threshold)
        | (margin < margin_threshold)
        | (entropy > entropy_threshold)
    )
    qc = qc_issue.reindex(X.index).fillna(False).astype(bool) if qc_issue is not None else pd.Series(False, index=X.index)
    out = pd.DataFrame(
        {
            "prediction": predictions,
            "confidence": confidence,
            "margin": margin,
            "entropy": entropy,
            "uncertainty_flag": uncertainty_flag,
            "review_zone": uncertainty_flag | qc.to_numpy(),
        },
        index=X.index,
    )
    for i, klass in enumerate(classes):
        out[f"probability_{klass}"] = probabilities[:, i]
    return out
