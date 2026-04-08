"""
Visualization tools.
Plotly interactive charts and Seaborn/Matplotlib static plots.
HIGH-CONTRAST DARK THEME — all colors tuned for visibility on dark backgrounds.
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


# ─── High-contrast color palette for dark backgrounds ───
CHART_PALETTE = [
    "#66d9ef",  # Bright cyan
    "#f9a825",  # Vivid amber
    "#ef5350",  # Bright red
    "#66bb6a",  # Bright green
    "#ab47bc",  # Purple
    "#ff7043",  # Deep orange
    "#29b6f6",  # Light blue
    "#fdd835",  # Yellow
    "#ec407a",  # Pink
    "#26a69a",  # Teal
]


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
            go.Bar(
                x=data.get("x", []),
                y=data.get("y", []),
                marker_color=data.get("color", "#66d9ef"),
                text=data.get("y", []),
                textposition="outside",
                textfont=dict(color="#e0e8e4", size=11),
            )
        ])
    elif chart_type == "scatter":
        fig = go.Figure(data=[
            go.Scatter(
                x=data.get("x", []),
                y=data.get("y", []),
                mode="markers",
                marker=dict(
                    color=data.get("color", "#66d9ef"),
                    size=data.get("size", 8),
                    line=dict(width=1, color="#ffffff40"),
                ),
            )
        ])
    elif chart_type == "histogram":
        fig = go.Figure(data=[
            go.Histogram(
                x=data.get("x", []),
                marker_color=data.get("color", "#66d9ef"),
                marker_line=dict(color="#ffffff30", width=1),
                nbinsx=data.get("bins", 30),
                opacity=0.9,
            )
        ])
    elif chart_type == "heatmap":
        fig = go.Figure(data=[
            go.Heatmap(
                z=data.get("z", [[]]),
                x=data.get("x", []),
                y=data.get("y", []),
                colorscale=data.get("colorscale", "RdBu_r"),
                texttemplate="%{z:.2f}",
                textfont=dict(size=10, color="#e0e8e4"),
            )
        ])
    elif chart_type == "box":
        fig = go.Figure()
        for i, (name, values) in enumerate(data.get("groups", {}).items()):
            fig.add_trace(go.Box(
                y=values, name=str(name),
                marker_color=CHART_PALETTE[i % len(CHART_PALETTE)],
            ))
    elif chart_type == "line":
        fig = go.Figure(data=[
            go.Scatter(
                x=data.get("x", []),
                y=data.get("y", []),
                mode="lines+markers",
                line=dict(color=data.get("color", "#66d9ef"), width=2),
                marker=dict(size=6, color=data.get("color", "#66d9ef")),
            )
        ])
    elif chart_type == "pie":
        fig = go.Figure(data=[
            go.Pie(
                labels=data.get("labels", []),
                values=data.get("values", []),
                textinfo="label+percent+value",
                textfont=dict(size=12, color="#ffffff"),
                marker=dict(colors=CHART_PALETTE[:len(data.get("labels", []))]),
            )
        ])
    else:
        fig = go.Figure()

    if fig and title:
        fig.update_layout(
            title=dict(text=title, font=dict(color="#e0e8e4", size=14)),
            template="plotly_dark",
            font=dict(family="Inter, sans-serif", color="#c8d8cc", size=11),
            plot_bgcolor="#131e16",
            paper_bgcolor="#111a14",
            xaxis=dict(
                gridcolor="rgba(100,140,110,0.25)",
                linecolor="rgba(100,140,110,0.4)",
                tickfont=dict(color="#a0b8a8"),
                title_font=dict(color="#c8d8cc"),
            ),
            yaxis=dict(
                gridcolor="rgba(100,140,110,0.25)",
                linecolor="rgba(100,140,110,0.4)",
                tickfont=dict(color="#a0b8a8"),
                title_font=dict(color="#c8d8cc"),
            ),
        )

    result = {"chart_type": chart_type, "title": title}

    if fig and output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        html_path = output_path if output_path.endswith(".html") else output_path + ".html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        result["html_path"] = html_path

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
    HIGH-CONTRAST colors for dark backgrounds.
    """
    # Set up dark theme with visible colors
    plt.rcParams.update({
        'figure.facecolor': '#111a14',
        'axes.facecolor': '#131e16',
        'axes.edgecolor': '#4a6a54',
        'axes.labelcolor': '#c8d8cc',
        'text.color': '#e0e8e4',
        'xtick.color': '#a0b8a8',
        'ytick.color': '#a0b8a8',
        'grid.color': 'rgba(100,140,110,0.2)',
        'grid.alpha': 0.3,
        'legend.facecolor': '#1a2a1e',
        'legend.edgecolor': '#4a6a54',
    })

    fig, ax = plt.subplots(figsize=figsize, facecolor='#111a14')
    ax.set_facecolor('#131e16')

    try:
        if plot_type == "histogram":
            if column and column in df.columns:
                sns.histplot(
                    df[column].dropna(), bins=30, ax=ax,
                    color="#66d9ef", kde=True, edgecolor="#ffffff30",
                    alpha=0.8, linewidth=0.5,
                )
                ax.axvline(df[column].median(), color="#f9a825", linestyle="--",
                           linewidth=1.5, label=f"Median: {df[column].median():.2f}")
                ax.legend(fontsize=9, facecolor='#1a2a1e', edgecolor='#4a6a54',
                          labelcolor='#c8d8cc')
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        color="#c8d8cc", fontsize=14)

        elif plot_type == "correlation_matrix":
            numeric_df = df.select_dtypes(include="number")
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(
                    corr, annot=len(corr) <= 15, fmt=".2f",
                    cmap="RdYlBu_r", center=0, ax=ax,
                    square=True, linewidths=0.5,
                    linecolor="#2a3a2e",
                    annot_kws={"size": 8, "color": "#e0e8e4"},
                    cbar_kws={"shrink": 0.8},
                )
                # Make colorbar text visible
                cbar = ax.collections[0].colorbar
                cbar.ax.tick_params(colors='#c8d8cc')
            else:
                ax.text(0.5, 0.5, "Insufficient numeric columns",
                        ha="center", va="center", color="#c8d8cc", fontsize=12)

        elif plot_type == "boxplot":
            if column and column in df.columns:
                if hue and hue in df.columns:
                    sns.boxplot(data=df, x=hue, y=column, ax=ax,
                                palette=CHART_PALETTE[:df[hue].nunique()])
                else:
                    sns.boxplot(data=df, y=column, ax=ax, color="#66d9ef",
                                flierprops=dict(markerfacecolor='#ef5350', markersize=4))
            else:
                numeric_cols = df.select_dtypes(include="number").columns[:10]
                if len(numeric_cols) > 0:
                    df[numeric_cols].boxplot(ax=ax, patch_artist=True,
                                             boxprops=dict(facecolor='#66d9ef40', edgecolor='#66d9ef'),
                                             whiskerprops=dict(color='#a0b8a8'),
                                             capprops=dict(color='#a0b8a8'),
                                             medianprops=dict(color='#f9a825', linewidth=2))

        elif plot_type == "countplot":
            if column and column in df.columns:
                order = df[column].value_counts().head(15).index
                palette = CHART_PALETTE[:len(order)]
                sns.countplot(data=df, x=column, hue=column, ax=ax, order=order,
                              palette=palette, legend=False, edgecolor="#ffffff20")
                # Add value labels on top of bars
                for container in ax.containers:
                    ax.bar_label(container, fontsize=9, color='#e0e8e4',
                                 padding=3, fmt='%d')
                plt.xticks(rotation=45, ha="right")

        elif plot_type == "pairplot":
            plt.close(fig)
            numeric_cols = df.select_dtypes(include="number").columns[:5]
            if len(numeric_cols) >= 2:
                plot_df = df[numeric_cols].dropna().head(1000)
                with plt.rc_context({
                    'figure.facecolor': '#111a14',
                    'axes.facecolor': '#131e16',
                    'text.color': '#e0e8e4',
                    'axes.labelcolor': '#c8d8cc',
                    'xtick.color': '#a0b8a8',
                    'ytick.color': '#a0b8a8',
                }):
                    g = sns.pairplot(plot_df, diag_kind="kde",
                                     plot_kws={"alpha": 0.6, "s": 15, "color": "#66d9ef"},
                                     diag_kws={"color": "#66d9ef", "alpha": 0.7})
                    if title:
                        g.figure.suptitle(title, y=1.02, color="#e0e8e4")
                    g.figure.set_facecolor('#111a14')
                    if output_path:
                        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
                        g.savefig(output_path, dpi=150, bbox_inches="tight",
                                  facecolor="#111a14")
                    plt.close()
                return {"plot_type": plot_type, "path": output_path}
            return {"plot_type": plot_type, "error": "Not enough numeric columns"}

        if title:
            ax.set_title(title, fontsize=14, fontweight="bold", color="#e0e8e4", pad=12)

        # Style axes
        ax.tick_params(colors='#a0b8a8')
        ax.spines['bottom'].set_color('#4a6a54')
        ax.spines['left'].set_color('#4a6a54')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()

        result = {"plot_type": plot_type, "title": title}

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight",
                        facecolor="#111a14", edgecolor="none")
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
    """Generate an interactive feature importance bar chart with high-contrast colors."""
    try:
        import plotly.graph_objects as go

        sorted_imp = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])

        # Generate a gradient from cyan to amber for importance bars
        n = len(sorted_imp)
        colors = []
        for i in range(n):
            ratio = i / max(n - 1, 1)
            # Cyan (#66d9ef) to Amber (#f9a825)
            r = int(0x66 + (0xf9 - 0x66) * ratio)
            g = int(0xd9 + (0xa8 - 0xd9) * ratio)
            b = int(0xef + (0x25 - 0xef) * ratio)
            colors.append(f"rgb({r},{g},{b})")

        fig = go.Figure(data=[
            go.Bar(
                x=list(sorted_imp.values()),
                y=list(sorted_imp.keys()),
                orientation="h",
                marker=dict(
                    color=colors,
                    line=dict(color="#ffffff20", width=1),
                ),
                text=[f"{v:.4f}" for v in sorted_imp.values()],
                textposition="outside",
                textfont=dict(color="#e0e8e4", size=10),
            )
        ])

        fig.update_layout(
            title=dict(text=title, font=dict(color="#e0e8e4", size=14)),
            template="plotly_dark",
            font=dict(family="Inter, sans-serif", color="#c8d8cc", size=11),
            plot_bgcolor="#131e16",
            paper_bgcolor="#111a14",
            xaxis=dict(
                title=dict(text="Importance Score", font=dict(color="#c8d8cc")),
                gridcolor="rgba(100,140,110,0.25)",
                linecolor="rgba(100,140,110,0.4)",
                tickfont=dict(color="#a0b8a8"),
            ),
            yaxis=dict(
                title=dict(text="Feature", font=dict(color="#c8d8cc")),
                autorange="reversed",
                tickfont=dict(color="#c8d8cc"),
            ),
            height=max(400, len(sorted_imp) * 35),
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
