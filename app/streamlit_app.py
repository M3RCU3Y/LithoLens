"""LithoLens Streamlit dashboard."""

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

st.set_page_config(page_title="LithoLens", layout="wide")
st.title("LithoLens")
st.caption("Uncertainty-aware well-log lithology prediction")

prediction_files = sorted(Path("reports/predictions").glob("*.csv"))
uploaded = st.sidebar.file_uploader("Upload prediction CSV", type=["csv"])
selected_file = None
if uploaded is None and prediction_files:
    selected_file = st.sidebar.selectbox("Sample output", prediction_files, format_func=lambda p: p.name)

if uploaded is not None:
    df = pd.read_csv(uploaded)
elif selected_file is not None:
    df = pd.read_csv(selected_file)
else:
    st.info("Upload a prediction CSV or run `python scripts/make_demo_outputs.py` first.")
    st.stop()

depth_col = "DEPTH_MD" if "DEPTH_MD" in df.columns else st.sidebar.selectbox("Depth column", df.columns)
well_col = "WELL" if "WELL" in df.columns else None
if well_col:
    well = st.sidebar.selectbox("Well", sorted(df[well_col].dropna().unique()))
    view = df[df[well_col] == well].copy()
else:
    view = df.copy()

curve_cols = [col for col in COMMON_LOG_COLUMNS if col in view.columns]
tabs = st.tabs(["Tracks", "QC", "Metrics / Rows"])

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
        if "qc_issue" in view.columns:
            st.plotly_chart(plot_qc_track(view, depth_col), use_container_width=True)
    if "explanation" in view.columns:
        depth = st.slider(
            "Depth explanation",
            float(view[depth_col].min()),
            float(view[depth_col].max()),
            float(view[depth_col].median()),
        )
        nearest = view.iloc[(view[depth_col] - depth).abs().argsort().iloc[0]]
        st.subheader(f"Explanation at {nearest[depth_col]:.2f}")
        st.write(nearest["explanation"])

with tabs[1]:
    flag_cols = [col for col in view.columns if col.endswith("_flag") or col.endswith("_imputed")]
    if flag_cols:
        st.dataframe(view[[depth_col] + flag_cols].head(500), use_container_width=True)
    else:
        st.info("No QC flag columns found.")

with tabs[2]:
    st.metric("Rows", len(view))
    if "review_zone" in view.columns:
        st.metric("Review-zone fraction", f"{view['review_zone'].mean():.1%}")
    st.dataframe(view.head(500), use_container_width=True)
