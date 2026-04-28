"""Evaluation metrics and report helpers."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def weighted_f1(y_true, y_pred) -> float:
    """Compute weighted F1."""
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def macro_f1(y_true, y_pred) -> float:
    """Compute macro F1."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    """Compute accuracy."""
    return float(accuracy_score(y_true, y_pred))


def confusion_matrix_df(y_true, y_pred, labels=None) -> pd.DataFrame:
    """Return a confusion matrix as a dataframe."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels) if labels is not None else pd.DataFrame(matrix)


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    """Return sklearn per-class precision/recall/F1 report."""
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True, zero_division=0)).T


def force_penalty_score(y_true, y_pred, penalty_matrix: pd.DataFrame | None = None) -> float | None:
    """Compute FORCE penalty score when a penalty matrix is available."""
    if penalty_matrix is None:
        return None
    total = 0.0
    for truth, pred in zip(y_true, y_pred, strict=False):
        total += float(penalty_matrix.loc[truth, pred])
    return total / max(len(y_true), 1)
