"""Well-level split helpers."""

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def get_group_kfold_splits(
    df: pd.DataFrame, group_col: str, n_splits: int = 5
) -> list[tuple[list[int], list[int]]]:
    """Return GroupKFold index splits that keep wells isolated."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    unique_groups = df[group_col].nunique()
    if unique_groups < 2:
        raise ValueError("At least two wells/groups are required for well-level validation.")
    n_splits = min(n_splits, unique_groups)
    splitter = GroupKFold(n_splits=n_splits)
    return [
        (train_idx.tolist(), val_idx.tolist())
        for train_idx, val_idx in splitter.split(df, groups=df[group_col])
    ]


def get_holdout_wells(
    df: pd.DataFrame, group_col: str, test_size: float = 0.2, random_state: int = 42
) -> tuple[list[int], list[int]]:
    """Return one train/test split using held-out wells."""
    if group_col not in df.columns:
        raise ValueError(f"Missing group column: {group_col}")
    if df[group_col].nunique() < 2:
        raise ValueError("At least two wells/groups are required for a holdout split.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df[group_col]))
    return train_idx.tolist(), test_idx.tolist()
