"""Quality-control checks for well-log data."""

import numpy as np
import pandas as pd

from litholens.constants import COMMON_LOG_COLUMNS, DEFAULT_PHYSICAL_RANGES


def missing_curve_report(df: pd.DataFrame) -> pd.DataFrame:
    """Report missingness by column."""
    rows = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing,
                "missing_fraction": missing / max(len(df), 1),
                "present": missing < len(df),
            }
        )
    return pd.DataFrame(rows)


def missing_interval_report(df: pd.DataFrame, depth_col: str) -> pd.DataFrame:
    """Estimate depth gaps larger than the median sampling interval."""
    if depth_col not in df.columns or len(df) < 3:
        return pd.DataFrame(columns=["start_depth", "end_depth", "gap"])
    ordered = df.sort_values(depth_col)
    diffs = ordered[depth_col].diff()
    median_step = diffs[diffs > 0].median()
    if pd.isna(median_step) or median_step <= 0:
        return pd.DataFrame(columns=["start_depth", "end_depth", "gap"])
    gap_rows = ordered.loc[diffs > median_step * 1.5, depth_col]
    previous = ordered[depth_col].shift(1).loc[gap_rows.index]
    return pd.DataFrame({"start_depth": previous, "end_depth": gap_rows, "gap": gap_rows - previous})


def detect_spikes(
    df: pd.DataFrame, curve_cols: list[str], z_threshold: float = 6.0
) -> pd.DataFrame:
    """Detect robust z-score spikes per curve without dropping rows."""
    flags = pd.DataFrame(index=df.index)
    for col in curve_cols:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        median = values.median()
        mad = (values - median).abs().median()
        if pd.isna(mad) or mad == 0:
            flags[f"{col}_spike_flag"] = False
            continue
        robust_z = 0.6745 * (values - median).abs() / mad
        flags[f"{col}_spike_flag"] = robust_z > z_threshold
    return flags.fillna(False)


def detect_physical_range_violations(
    df: pd.DataFrame, range_config: dict[str, tuple[float, float]] | None = None
) -> pd.DataFrame:
    """Flag values outside conservative physical ranges."""
    ranges = range_config or DEFAULT_PHYSICAL_RANGES
    flags = pd.DataFrame(index=df.index)
    for col, (low, high) in ranges.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        flags[f"{col}_range_flag"] = values.notna() & ((values < low) | (values > high))
    return flags.fillna(False)


def create_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Append missing, spike, range, and aggregate QC flags."""
    curve_cols = [col for col in COMMON_LOG_COLUMNS if col in df.columns]
    out = df.copy()
    for col in curve_cols:
        out[f"{col}_missing_flag"] = out[col].isna()
    spike_flags = detect_spikes(out, curve_cols)
    range_flags = detect_physical_range_violations(out)
    out = pd.concat([out, spike_flags, range_flags], axis=1)
    flag_cols = [col for col in out.columns if col.endswith("_flag")]
    out["qc_issue"] = out[flag_cols].any(axis=1) if flag_cols else False
    return out


def summarize_qc(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize boolean QC flag counts."""
    flag_cols = [col for col in df.columns if col.endswith("_flag") or col == "qc_issue"]
    return pd.DataFrame(
        [{"flag": col, "count": int(df[col].sum()), "fraction": float(np.mean(df[col]))} for col in flag_cols]
    )
