"""Feature engineering for well-log lithology models."""

import numpy as np
import pandas as pd

from litholens.schema import available_curve_cols


def make_base_features(df: pd.DataFrame, curve_cols: list[str]) -> pd.DataFrame:
    """Return numeric base curves and existing indicator columns."""
    cols = [col for col in curve_cols if col in df.columns]
    indicator_cols = [
        col for col in df.columns if col.endswith("_was_missing") or col.endswith("_imputed")
    ]
    features = df[cols + indicator_cols].copy()
    for col in features.columns:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0)
    return features


def add_rolling_features(
    df: pd.DataFrame,
    group_col: str,
    depth_col: str,
    curve_cols: list[str],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Add rolling mean features computed separately within each well."""
    windows = windows or [5, 15, 31]
    out = df.copy()
    if group_col not in out.columns:
        out[group_col] = "__single_well__"
    out = out.sort_values([group_col, depth_col]).copy()
    for col in curve_cols:
        if col not in out.columns:
            continue
        for window in windows:
            out[f"{col}_roll_mean_{window}"] = (
                out.groupby(group_col)[col]
                .transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
            )
    return out.sort_index()


def add_gradient_features(
    df: pd.DataFrame, group_col: str, depth_col: str, curve_cols: list[str]
) -> pd.DataFrame:
    """Add within-well gradients with respect to depth."""
    out = df.copy()
    if group_col not in out.columns:
        out[group_col] = "__single_well__"
    out = out.sort_values([group_col, depth_col]).copy()
    for col in curve_cols:
        if col not in out.columns:
            continue
        out[f"{col}_gradient"] = out.groupby(group_col, group_keys=False).apply(
            lambda g: g[col].diff() / g[depth_col].diff().replace(0, np.nan),
            include_groups=False,
        )
    return out.sort_index()


def add_cross_log_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple robust cross-log ratios when source curves exist."""
    out = df.copy()
    eps = 1e-6
    if {"RDEP", "RMED"}.issubset(out.columns):
        out["RDEP_RMED_RATIO"] = out["RDEP"] / (out["RMED"].abs() + eps)
    if {"NPHI", "RHOB"}.issubset(out.columns):
        out["NPHI_RHOB_RATIO"] = out["NPHI"] / (out["RHOB"].abs() + eps)
    if {"GR", "RHOB"}.issubset(out.columns):
        out["GR_RHOB_RATIO"] = out["GR"] / (out["RHOB"].abs() + eps)
    return out


def build_feature_matrix(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    """Build a numeric feature matrix using configured feature options."""
    config = config or {}
    data_cfg = config.get("data", {})
    feature_cfg = config.get("features", {})
    curve_cols = available_curve_cols(df, data_cfg.get("curve_cols"))
    group_col = data_cfg.get("group_col", "WELL")
    depth_col = data_cfg.get("depth_col", "DEPTH_MD")
    work = df.copy()
    if feature_cfg.get("include_ratios", True):
        work = add_cross_log_ratios(work)
    if depth_col in work.columns:
        work = add_rolling_features(
            work, group_col, depth_col, curve_cols, feature_cfg.get("rolling_windows", [5, 15, 31])
        )
        if feature_cfg.get("include_gradients", True):
            work = add_gradient_features(work, group_col, depth_col, curve_cols)
    base = make_base_features(work, curve_cols)
    engineered_cols = [
        col
        for col in work.columns
        if any(token in col for token in ["_roll_mean_", "_gradient", "_RATIO"])
    ]
    engineered = work[engineered_cols].apply(pd.to_numeric, errors="coerce") if engineered_cols else pd.DataFrame(index=work.index)
    return pd.concat([base, engineered], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0)
