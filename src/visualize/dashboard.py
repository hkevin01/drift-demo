"""
# ============================================================
# FILE: src/visualize/dashboard.py
# ID: DSH-001
# Purpose: Lightweight Plotly Dash dashboard for drift metrics.
# ============================================================
"""

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html

from utils.config import LOG_PATH


def build_dashboard(log_path: str = LOG_PATH) -> Dash:
    """
    ID: DSH-002
    Purpose: Build and return a Dash app from the drift log CSV.
    Inputs: log_path - CSV produced by the main pipeline.
    Outputs: configured Dash app instance.
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path)
    if df.empty:
        raise ValueError("Log file is empty. Run pipeline before dashboard.")

    fig_acc = px.line(df, x="batch", y="accuracy", title="Accuracy Over Time")
    fig_sig = px.line(df, x="batch", y=["psi", "kl_div", "ks_pvalue"], title="Drift Signals")

    retrain_points = df[df["retrain_triggered"] == True]
    if not retrain_points.empty:
        fig_acc.add_scatter(
            x=retrain_points["batch"],
            y=retrain_points["accuracy"],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Retrain",
        )

    app = Dash(__name__)
    app.layout = html.Div(
        style={"fontFamily": "Arial", "padding": "20px"},
        children=[
            html.H1("Drift Monitoring Dashboard"),
            dcc.Graph(figure=fig_acc),
            dcc.Graph(figure=fig_sig),
        ],
    )
    return app


def run_dashboard() -> None:
    """
    ID: DSH-003
    Purpose: Launch the Dash server at http://127.0.0.1:8050.
    """
    app = build_dashboard()
    app.run(debug=False)


if __name__ == "__main__":
    run_dashboard()
