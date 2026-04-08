"""
Visualization tools.
Plotly interactive charts and Seaborn/Matplotlib static plots.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from tools.registry import tool

logger = logging.getLogger(__name__)

# Use non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


@tool(name="plotly_chart", description="Generate an interactive Plotly chart", category="viz")
def plotly_chart(data: dict, chart_type: str = "bar",
                 title: str = "", output_path: str = "",
                 **kwargs) -> dict:
    """
    Generate an interactive Plotly chart and save as HTML + JSON.

    Supported chart_types: bar, scatter, histogram, heatmap, box, line, pie
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly.io as pio
    except ImportError:
        logger.warning("Plotly not installed, skipping chart generation")
        return {"error": "plotly not installed"}

    fig = None

    if chart_type == "bar":
        fig = go.Figure(data=[
            go.Bar(x=data.get("x", []), y=data.get("y", []),
                   marker_color=data.get("color", "#667eea"))
        ])
    elif chart_type == "scatter":
        fig = go.Figure(data=[
            go.Scatter(x=data.get("x", []), y=data.get("y", []),
                       mode="markers",
                       marker=dict(color=data.get("color", "#667eea"),
                                   size=data.get("size", 6)))
        ])
    elif chart_type == "histogram":
        fig = go.Figure(data=[
            go.Histogram(x=data.get("x", []),
                         marker_color=data.get("color", "#667eea"),
                         nbinsx=data.get("bins", 30))
        ])
    elif chart_type == "heatmap":
        fig = go.Figure(data=[
            go.Heatmap(z=data.get("z", [[]]),
                       x=data.get("x", []),
                       y=data.get("y", []),
                       colorscale=data.get("colorscale", "RdBu_r"))
        ])
    elif chart_type == "box":
        fig = go.Figure()
        for name, values in data.get("groups", {}).items():
            fig.add_trace(go.Box(y=values, name=str(name)))
    elif chart_type == "line":
        fig = go.Figure(data=[
            go.Scatter(x=data.get("x", []), y=data.get("y", []),
                       mode="lines+markers",
                       line=dict(color=data.get("color", "#667eea")))
        ])
    elif chart_type == "pie":
        fig = go.Figure(data=[
            go.Pie(labels=data.get("labels", []),
                   values=data.get("values", []))
        ])
    else:
        fig = go.Figure()

    if fig and title:
        fig.update_layout(
            title=title,
            template="plotly_dark",
            font=dict(family="Inter, sans-serif"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

    result = {"chart_type": chart_type, "title": title}

    if fig and output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        # Save HTML
        html_path = output_path if output_path.endswith(".html") else output_path + ".html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        result["html_path"] = html_path

        # Save JSON for embedding
        json_path = output_path.replace(".html", ".json") if output_path.endswith(".html") else output_path + ".json"
        result["json_data"] = fig.to_json()
        result["json_path"] = json_path
        with open(json_path, "w") as f:
            f.write(fig.to_json())

    return result


@tool(name="seaborn_plot", description="Generate a static Seaborn/Matplotlib plot", category="viz")
def seaborn_plot(df: pd.DataFrame, plot_type: str = "histogram",
                 column: str = "", hue: str = "",
                 title: str = "", output_path: str = "",
                 figsize: tuple = (10, 6), **kwargs) -> dict:
    """
    Generate a static plot with Seaborn and save as PNG.

    Supported plot_types: histogram, correlation_matrix, boxplot, pairplot, countplot
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Apply dark style
    plt.style.use("dark_background")
    sns.set_theme(style="darkgrid")

    try:
        if plot_type == "histogram":
            if column and column in df.columns:
                sns.histplot(df[column].dropna(), bins=30, ax=ax,
                             color="#667eea", kde=True)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center")

        elif plot_type == "correlation_matrix":
            numeric_df = df.select_dtypes(include="number")
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                sns.heatmap(corr, annot=len(corr) <= 12, fmt=".2f",
                            cmap="RdBu_r", center=0, ax=ax,
                            square=True, linewidths=0.5)
            else:
                ax.text(0.5, 0.5, "Insufficient numeric columns",
                        ha="center", va="center")

        elif plot_type == "boxplot":
            if column and column in df.columns:
                if hue and hue in df.columns:
                    sns.boxplot(data=df, x=hue, y=column, ax=ax,
                                palette="viridis")
                else:
                    sns.boxplot(data=df, y=column, ax=ax, color="#667eea")
            else:
                numeric_cols = df.select_dtypes(include="number").columns[:10]
                if len(numeric_cols) > 0:
                    df[numeric_cols].boxplot(ax=ax)

        elif plot_type == "countplot":
            if column and column in df.columns:
                order = df[column].value_counts().head(15).index
                sns.countplot(data=df, x=column, ax=ax, order=order,
                              palette="viridis")
                plt.xticks(rotation=45, ha="right")

        elif plot_type == "pairplot":
            # pairplot creates its own figure
            plt.close(fig)
            numeric_cols = df.select_dtypes(include="number").columns[:5]
            if len(numeric_cols) >= 2:
                plot_df = df[numeric_cols].dropna().head(1000)
                g = sns.pairplot(plot_df, diag_kind="kde",
                                 plot_kws={"alpha": 0.5, "s": 15})
                if title:
                    g.figure.suptitle(title, y=1.02)
                if output_path:
                    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                    g.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()
                return {"plot_type": plot_type, "path": output_path}
            return {"plot_type": plot_type, "error": "Not enough numeric columns"}

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        result = {"plot_type": plot_type, "title": title}

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight",
                        facecolor="#1a1a2e")
            result["path"] = output_path

        plt.close(fig)
        return result

    except Exception as e:
        plt.close(fig)
        logger.error(f"Plot generation failed: {e}")
        return {"plot_type": plot_type, "error": str(e)}


@tool(name="generate_feature_importance_chart", description="Generate feature importance bar chart", category="viz")
def generate_feature_importance_chart(importance: dict[str, float],
                                       title: str = "Feature Importance",
                                       output_path: str = "output/charts/feature_importance.html") -> dict:
    """Generate an interactive feature importance bar chart."""
    try:
        import plotly.graph_objects as go

        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])

        fig = go.Figure(data=[
            go.Bar(
                x=list(sorted_imp.values()),
                y=list(sorted_imp.keys()),
                orientation="h",
                marker=dict(
                    color=list(sorted_imp.values()),
                    colorscale="Viridis",
                )
            )
        ])

        fig.update_layout(
            title=title,
            template="plotly_dark",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=max(400, len(sorted_imp) * 30),
            font=dict(family="Inter, sans-serif"),
            yaxis=dict(autorange="reversed"),
        )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.write_html(output_path, include_plotlyjs="cdn")

        json_path = output_path.replace(".html", ".json")
        with open(json_path, "w") as f:
            f.write(fig.to_json())

        return {"html_path": output_path, "json_path": json_path}

    except ImportError:
        logger.warning("Plotly not available")
        return {"error": "plotly not installed"}
