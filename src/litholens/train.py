"""Model training utilities."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


def train_baseline_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int | None = None,
    class_weight: str | None = "balanced_subsample",
) -> RandomForestClassifier:
    """Train a robust RandomForest baseline."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting_model(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> HistGradientBoostingClassifier:
    """Train a scikit-learn histogram gradient boosting model."""
    model = HistGradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def save_model(model: Any, path: str | Path) -> Path:
    """Persist a trained model with joblib."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return output
