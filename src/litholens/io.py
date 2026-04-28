"""Data loading and mnemonic normalization."""

from pathlib import Path

import pandas as pd

from litholens.constants import MNEMONIC_ALIASES


def load_force_csv(path: str | Path) -> pd.DataFrame:
    """Load FORCE-style CSV data and standardize known column names."""
    return standardize_mnemonics(normalize_column_names(pd.read_csv(Path(path), sep=None, engine="python")))


def load_generic_csv(path: str | Path) -> pd.DataFrame:
    """Load a generic CSV and standardize known well-log mnemonics."""
    return standardize_mnemonics(normalize_column_names(pd.read_csv(Path(path))))


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to uppercase snake-ish mnemonic form."""
    renamed = {
        col: str(col).strip().replace(" ", "_").replace("-", "_").upper()
        for col in df.columns
    }
    return df.rename(columns=renamed)


def standardize_mnemonics(df: pd.DataFrame) -> pd.DataFrame:
    """Map common aliases to LithoLens canonical names."""
    rename_map = {col: MNEMONIC_ALIASES.get(col, col) for col in df.columns}
    return df.rename(columns=rename_map)
