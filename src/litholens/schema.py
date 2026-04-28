"""Schema helpers for resolving dataset columns."""

from collections.abc import Iterable

import pandas as pd

from litholens.constants import (
    COMMON_LOG_COLUMNS,
    DEPTH_COLUMN_CANDIDATES,
    TARGET_COLUMN_CANDIDATES,
    WELL_COLUMN_CANDIDATES,
)


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    """Return the first candidate present in the dataframe."""
    columns = set(df.columns)
    return next((candidate for candidate in candidates if candidate in columns), None)


def infer_target_col(df: pd.DataFrame) -> str | None:
    """Infer a lithology target column if one exists."""
    return first_existing_column(df, TARGET_COLUMN_CANDIDATES)


def infer_depth_col(df: pd.DataFrame) -> str | None:
    """Infer the depth column if one exists."""
    return first_existing_column(df, DEPTH_COLUMN_CANDIDATES)


def infer_group_col(df: pd.DataFrame) -> str | None:
    """Infer the well/group column if one exists."""
    return first_existing_column(df, WELL_COLUMN_CANDIDATES)


def available_curve_cols(df: pd.DataFrame, configured: Iterable[str] | None = None) -> list[str]:
    """Return configured or common curve columns present in the dataframe."""
    candidates = list(configured) if configured else COMMON_LOG_COLUMNS
    return [col for col in candidates if col in df.columns]
