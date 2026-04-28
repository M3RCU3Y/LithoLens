"""Missing value indicators and simple imputation strategies."""

import pandas as pd


def add_missing_indicators(df: pd.DataFrame, curve_cols: list[str]) -> pd.DataFrame:
    """Add measured-vs-missing indicators for curve columns."""
    out = df.copy()
    for col in curve_cols:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna()
            out[f"{col}_imputed"] = False
    return out


def interpolate_short_gaps(
    df: pd.DataFrame,
    group_col: str,
    depth_col: str,
    curve_cols: list[str],
    max_gap: int = 5,
) -> pd.DataFrame:
    """Interpolate short gaps within each well and mark imputed values."""
    out = add_missing_indicators(df, curve_cols)
    sort_cols = [group_col, depth_col] if group_col in out.columns else [depth_col]
    out = out.sort_values(sort_cols).copy()
    groups = out.groupby(group_col, sort=False) if group_col in out.columns else [(None, out)]
    pieces = []
    for _, group in groups:
        group = group.copy()
        for col in curve_cols:
            if col not in group.columns:
                continue
            before = group[col].isna()
            group[col] = group[col].interpolate(limit=max_gap, limit_direction="both")
            filled = before & group[col].notna()
            group[f"{col}_imputed"] = group.get(f"{col}_imputed", False) | filled
        pieces.append(group)
    return pd.concat(pieces).sort_index()


def model_or_median_impute(
    df: pd.DataFrame, curve_cols: list[str], group_col: str | None = None
) -> pd.DataFrame:
    """Fill remaining gaps with well median, then global median."""
    out = add_missing_indicators(df, curve_cols)
    for col in curve_cols:
        if col not in out.columns:
            continue
        before = out[col].isna()
        if group_col and group_col in out.columns:
            out[col] = out[col].fillna(out.groupby(group_col)[col].transform("median"))
        out[col] = out[col].fillna(out[col].median())
        filled = before & out[col].notna()
        out[f"{col}_imputed"] = out.get(f"{col}_imputed", False) | filled
    return out
