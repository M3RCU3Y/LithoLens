"""LithoLens Streamlit dashboard for the MVP FORCE demo."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from litholens.constants import COMMON_LOG_COLUMNS
from litholens.plots import (
    plot_confidence_track,
    plot_lithology_track,
    plot_log_curves_by_depth,
    plot_qc_track,
    plot_uncertainty_track,
)

MVP_REPORT_DIR = Path("reports/mvp_baseline")
DEFAULT_PREDICTIONS = MVP_REPORT_DIR / "heldout_well_15_9-13_predictions.csv"
FOLD_METRICS = MVP_REPORT_DIR / "fold_metrics.csv"
SUMMARY_METRICS = MVP_REPORT_DIR / "metrics.json"

st.set_page_config(page_title="LithoLens", layout="wide")
st.title("LithoLens")
st.caption("FORCE 2020 lithology baseline with uncertainty, QC, and explanation")

prediction_files = sorted(MVP_REPORT_DIR.glob("heldout_well*_predictions.csv"))
uploaded = st.sidebar.file_uploader("Upload prediction CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif DEFAULT_PREDICTIONS.exists():
    df = pd.read_csv(DEFAULT_PREDICTIONS)
elif prediction_files:
    selected = st.sidebar.selectbox("MVP artifact", prediction_files, format_func=lambda path: path.name)
    df = pd.read_csv(selected)
else:
    st.info("Run `python scripts/run_baseline.py --config configs/baseline.yaml` to create MVP demo artifacts.")
    st.stop()

depth_col = "DEPTH_MD" if "DEPTH_MD" in df.columns else st.sidebar.selectbox("Depth column", df.columns)
well_col = "WELL" if "WELL" in df.columns else None
if well_col:
    wells = sorted(df[well_col].dropna().astype(str).unique())
    heldout_well = wells[0] if wells else "unknown"
    selected_well = st.sidebar.selectbox("Held-out well", wells, index=0)
    view = df[df[well_col].astype(str) == selected_well].copy()
else:
    heldout_well = "uploaded"
    view = df.copy()

curve_cols = [col for col in COMMON_LOG_COLUMNS if col in view.columns]
review_fraction = float(view["review_zone"].mean()) if "review_zone" in view.columns and len(view) else 0.0
mean_confidence = float(view["confidence"].mean()) if "confidence" in view.columns and len(view) else 0.0

top = st.columns(4)
top[0].metric("Held-out well", heldout_well)
top[1].metric("Rows", f"{len(view):,}")
top[2].metric("Mean confidence", f"{mean_confidence:.2f}")
top[3].metric("Review zones", f"{review_fraction:.1%}")

tabs = st.tabs(["Tracks", "Review Zones", "Explanations", "Fold Metrics"])

with tabs[0]:
    left, middle, right = st.columns([2, 1, 1])
    with left:
        st.plotly_chart(plot_log_curves_by_depth(view, depth_col, curve_cols), use_container_width=True)
    with middle:
        if "prediction" in view.columns:
            st.plotly_chart(plot_lithology_track(view, depth_col), use_container_width=True)
        if "confidence" in view.columns:
            st.plotly_chart(plot_confidence_track(view, depth_col), use_container_width=True)
    with right:
        if {"entropy", "review_zone"}.issubset(view.columns):
            st.plotly_chart(plot_uncertainty_track(view, depth_col), use_container_width=True)
        if "qc_warning" in view.columns:
            qc_view = view.assign(qc_issue=view["qc_warning"].fillna("").ne(""))
            st.plotly_chart(plot_qc_track(qc_view, depth_col), use_container_width=True)

with tabs[1]:
    columns = [
        col
        for col in [depth_col, "prediction", "actual", "confidence", "margin", "entropy", "uncertainty_flag", "qc_warning", "review_zone"]
        if col in view.columns
    ]
    review = view[view["review_zone"].astype(bool)] if "review_zone" in view.columns else view.head(0)
    st.dataframe(review[columns].head(500), use_container_width=True)

with tabs[2]:
    if "explanation" not in view.columns:
        st.info("No explanation column found in this prediction file.")
    else:
        depth = st.slider(
            "Depth explanation",
            float(view[depth_col].min()),
            float(view[depth_col].max()),
            float(view[depth_col].median()),
        )
        nearest = view.iloc[(view[depth_col] - depth).abs().argsort().iloc[0]]
        st.subheader(f"Depth {nearest[depth_col]:.2f}")
        st.write(nearest["explanation"])
        detail_cols = [
            col
            for col in ["prediction", "actual", "confidence", "margin", "entropy", "qc_warning", "review_zone"]
            if col in nearest.index
        ]
        st.dataframe(nearest[detail_cols].to_frame("value"), use_container_width=True)

with tabs[3]:
    if FOLD_METRICS.exists():
        st.dataframe(pd.read_csv(FOLD_METRICS), use_container_width=True)
    else:
        st.info("No fold_metrics.csv found yet.")
    if SUMMARY_METRICS.exists():
        st.json(pd.read_json(SUMMARY_METRICS, typ="series").to_dict())
