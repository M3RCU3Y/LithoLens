"""Plotly visualizations for logs, lithology, confidence, uncertainty, and QC."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _reverse_depth(fig: go.Figure) -> go.Figure:
    fig.update_yaxes(autorange="reversed", title="Depth")
    return fig


def plot_log_curves_by_depth(df: pd.DataFrame, depth_col: str, curve_cols: list[str]) -> go.Figure:
    """Plot selected log curves against depth."""
    fig = go.Figure()
    for col in curve_cols:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df[col], y=df[depth_col], mode="lines", name=col))
    fig.update_layout(xaxis_title="Curve value", legend_title="Curve", height=650)
    return _reverse_depth(fig)


def plot_lithology_track(df: pd.DataFrame, depth_col: str, lithology_col: str = "prediction") -> go.Figure:
    """Plot predicted lithology as a depth-colored strip."""
    return _reverse_depth(px.scatter(df, x=lithology_col, y=depth_col, color=lithology_col, height=650))


def plot_confidence_track(df: pd.DataFrame, depth_col: str, confidence_col: str = "confidence") -> go.Figure:
    """Plot confidence against depth."""
    fig = px.line(df, x=confidence_col, y=depth_col, height=650)
    fig.update_xaxes(range=[0, 1])
    return _reverse_depth(fig)


def plot_uncertainty_track(df: pd.DataFrame, depth_col: str) -> go.Figure:
    """Plot uncertainty entropy and review zones."""
    fig = px.scatter(df, x="entropy", y=depth_col, color="review_zone", height=650)
    return _reverse_depth(fig)


def plot_qc_track(df: pd.DataFrame, depth_col: str) -> go.Figure:
    """Plot QC issue flags by depth."""
    fig = px.scatter(df, x="qc_issue", y=depth_col, color="qc_issue", height=650)
    return _reverse_depth(fig)
