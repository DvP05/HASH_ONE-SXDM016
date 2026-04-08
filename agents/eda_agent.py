"""
Stage 3 — EDA Agent
Conducts comprehensive Exploratory Data Analysis, surfaces
meaningful patterns, generates visualizations, and produces
a structured discovery report.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.models import EDAReport, Finding, Hypothesis

logger = logging.getLogger(__name__)


class EDAAgent(BaseAgent):
    """
    Performs structured Exploratory Data Analysis and emits a rich
    report with auto-generated visualizations and LLM commentary.
    """

    def run(self, hints: str = "") -> EDAReport:
        self.start_timer()
        self.log("═══ Stage 3: Exploratory Data Analysis ═══")

        df = self.memory.retrieve("clean_dataset")
        if df is None:
            raise ValueError("No clean_dataset in memory. Run DataCleaningAgent first.")

        target = self.memory.retrieve("target_column", "")
        output_dir = self.memory.retrieve("output_dir", "output")
        charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        findings: list[Finding] = []

        # ── Univariate Analysis ──
        self.log("📊 Running univariate analysis...")
        for col in df.columns:
            finding = self._analyze_univariate(df, col, charts_dir)
            findings.append(finding)

        # ── Bivariate Analysis (vs target) ──
        if target and target in df.columns:
            self.log(f"📊 Running bivariate analysis vs target '{target}'...")
            for col in df.columns:
                if col != target:
                    finding = self._analyze_bivariate(df, col, target, charts_dir)
                    if finding:
                        findings.append(finding)

        # ── Correlation Analysis ──
        self.log("📊 Analyzing correlations...")
        corr_findings = self._analyze_correlations(df, charts_dir)
        findings.extend(corr_findings)

        # ── Data Traits ──
        data_traits = self._compute_data_traits(df, target)

        # ── Target Distribution ──
        if target and target in df.columns:
            self._plot_target_distribution(df, target, charts_dir)

        # ── LLM Narrative Synthesis ──
        narrative = self._synthesize_narrative(findings, data_traits)

        # ── Hypothesis Generation ──
        hypotheses = self._generate_hypotheses(findings)

        # ── Feature Recommendations ──
        recommended_features = self._recommend_features(findings, target, df)

        report = EDAReport(
            findings=findings,
            narrative=narrative,
            hypotheses=hypotheses,
            recommended_features=recommended_features,
            correlation_matrix_path=os.path.join(charts_dir, "correlation_matrix.png"),
            data_traits=data_traits,
        )

        self.memory.store("eda_report", report, source="eda")
        self.log(f"✅ EDA complete: {len(findings)} findings, {len(hypotheses)} hypotheses")

        self.stop_timer()
        return report

    def _analyze_univariate(self, df: pd.DataFrame, col: str,
                             charts_dir: str) -> Finding:
        """Analyze a single column."""
        series = df[col]
        stats: dict = {
            "count": int(series.count()),
            "missing": int(series.isnull().sum()),
            "missing_pct": round(float(series.isnull().mean()), 4),
            "unique": int(series.nunique()),
            "dtype": str(series.dtype),
        }

        skewness = None
        kurtosis = None
        dist_type = "unknown"

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            stats.update({
                "mean": round(float(desc.get("mean", 0)), 4),
                "std": round(float(desc.get("std", 0)), 4),
                "min": float(desc.get("min", 0)),
                "max": float(desc.get("max", 0)),
                "median": float(desc.get("50%", 0)),
            })
            if len(series.dropna()) > 2:
                skewness = round(float(series.skew()), 4)
            if len(series.dropna()) > 3:
                kurtosis = round(float(series.kurtosis()), 4)

            # Distribution type inference
            if skewness is not None:
                if abs(skewness) < 0.5:
                    dist_type = "approximately_normal"
                elif skewness > 1.0:
                    dist_type = "right_skewed"
                elif skewness < -1.0:
                    dist_type = "left_skewed"
                else:
                    dist_type = "moderately_skewed"

            # Generate histogram
            try:
                self.execute_tool("seaborn_plot", df=df, plot_type="histogram",
                                  column=col, title=f"Distribution: {col}",
                                  output_path=os.path.join(charts_dir, f"dist_{col}.png"))
            except Exception:
                pass
        else:
            top_vals = series.value_counts().head(10).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in top_vals.items()}
            dist_type = "categorical"

        return Finding(
            column=col,
            analysis_type="univariate",
            stats=stats,
            skewness=skewness,
            kurtosis=kurtosis,
            distribution_type=dist_type,
        )

    def _analyze_bivariate(self, df: pd.DataFrame, col: str,
                            target: str, charts_dir: str) -> Finding | None:
        """Analyze relationship between a column and the target."""
        if col == target:
            return None

        stats: dict = {"column": col, "target": target}

        try:
            if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(df[target]):
                corr = df[[col, target]].corr().iloc[0, 1]
                if np.isnan(corr):
                    return None
                stats["correlation"] = round(float(corr), 4)
                stats["abs_correlation"] = round(abs(float(corr)), 4)

                # Only report significant correlations
                if abs(corr) > 0.1:
                    return Finding(
                        column=col,
                        analysis_type="bivariate",
                        stats=stats,
                    )
            elif pd.api.types.is_numeric_dtype(df[target]):
                # Categorical vs numeric target
                group_means = df.groupby(col)[target].mean()
                if len(group_means) > 1 and len(group_means) <= 20:
                    stats["group_means"] = {str(k): round(float(v), 4)
                                            for k, v in group_means.items()}
                    return Finding(
                        column=col,
                        analysis_type="bivariate_cat_num",
                        stats=stats,
                    )
        except Exception:
            pass

        return None

    def _analyze_correlations(self, df: pd.DataFrame,
                               charts_dir: str) -> list[Finding]:
        """Analyze correlation structure."""
        numeric_df = df.select_dtypes(include="number")
        if len(numeric_df.columns) < 2:
            return []

        corr_matrix = numeric_df.corr()

        # Generate correlation heatmap
        try:
            self.execute_tool("seaborn_plot", df=numeric_df,
                              plot_type="correlation_matrix",
                              title="Correlation Matrix",
                              output_path=os.path.join(charts_dir, "correlation_matrix.png"))
        except Exception:
            pass

        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.5:
                    high_corr.append({
                        "col_a": str(corr_matrix.columns[i]),
                        "col_b": str(corr_matrix.columns[j]),
                        "correlation": round(float(val), 4),
                    })

        high_corr.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        if high_corr:
            return [Finding(
                analysis_type="correlation",
                stats={"high_correlations": high_corr[:20]},
                commentary=f"Found {len(high_corr)} high-correlation pairs (|r| > 0.5)",
            )]

        return []

    def _compute_data_traits(self, df: pd.DataFrame, target: str) -> dict:
        """Compute high-level data traits."""
        traits = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_numeric": len(df.select_dtypes(include="number").columns),
            "n_categorical": len(df.select_dtypes(include=["object", "category"]).columns),
            "avg_missing_pct": round(float(df.isnull().mean().mean()), 4),
            "has_duplicates": bool(df.duplicated().any()),
        }

        if target and target in df.columns:
            if pd.api.types.is_numeric_dtype(df[target]):
                if df[target].nunique() == 2:
                    traits["task_type"] = "binary_classification"
                    value_counts = df[target].value_counts(normalize=True)
                    traits["class_balance"] = {str(k): round(float(v), 4)
                                               for k, v in value_counts.items()}
                elif df[target].nunique() <= 20:
                    traits["task_type"] = "multiclass_classification"
                else:
                    traits["task_type"] = "regression"
            else:
                traits["task_type"] = "classification"
                value_counts = df[target].value_counts(normalize=True)
                traits["class_balance"] = {str(k): round(float(v), 4)
                                           for k, v in value_counts.items()}

        return traits

    def _plot_target_distribution(self, df: pd.DataFrame, target: str,
                                   charts_dir: str) -> None:
        """Plot target variable distribution."""
        try:
            if pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10:
                self.execute_tool("seaborn_plot", df=df, plot_type="histogram",
                                  column=target, title=f"Target Distribution: {target}",
                                  output_path=os.path.join(charts_dir, "target_distribution.png"))
            else:
                self.execute_tool("seaborn_plot", df=df, plot_type="countplot",
                                  column=target, title=f"Target Distribution: {target}",
                                  output_path=os.path.join(charts_dir, "target_distribution.png"))
        except Exception as e:
            self.log(f"⚠️ Target plot failed: {e}", level="warning")

    def _synthesize_narrative(self, findings: list[Finding],
                               data_traits: dict) -> str:
        """Use LLM to synthesize findings into a narrative."""
        try:
            findings_summary = []
            for f in findings[:30]:  # Limit to avoid token overflow
                if f.analysis_type == "univariate" and f.skewness is not None:
                    findings_summary.append(
                        f"- {f.column}: {f.distribution_type}, skew={f.skewness}, "
                        f"missing={f.stats.get('missing_pct', 0):.1%}"
                    )
                elif f.analysis_type == "correlation":
                    for hc in f.stats.get("high_correlations", [])[:5]:
                        findings_summary.append(
                            f"- Correlation: {hc['col_a']} ↔ {hc['col_b']} = {hc['correlation']:.3f}"
                        )
                elif f.analysis_type == "bivariate":
                    corr = f.stats.get("correlation", 0)
                    if abs(corr) > 0.2:
                        findings_summary.append(
                            f"- {f.column} vs target: r={corr:.3f}"
                        )

            domain = self.memory.retrieve("domain_context", "")
            narrative = self.llm_chat(
                system=(
                    "You are a data scientist writing an EDA summary. "
                    "Be specific with numbers. Keep it under 300 words."
                ),
                user=(
                    f"Data traits: {json.dumps(data_traits, indent=2)}\n\n"
                    f"Key findings:\n" + "\n".join(findings_summary[:20]) + "\n\n"
                    f"Domain: {domain}\n\n"
                    "Write a concise EDA narrative."
                ),
            )
            return narrative

        except Exception as e:
            self.log(f"⚠️ Narrative synthesis failed: {e}", level="warning")
            return (
                f"Dataset contains {data_traits.get('n_rows', 0)} rows and "
                f"{data_traits.get('n_columns', 0)} columns. "
                f"Average missing rate is {data_traits.get('avg_missing_pct', 0):.1%}."
            )

    def _generate_hypotheses(self, findings: list[Finding]) -> list[Hypothesis]:
        """Generate data-driven hypotheses from findings."""
        try:
            corr_findings = [f for f in findings if f.analysis_type in ("bivariate", "correlation")]
            if not corr_findings:
                return []

            corr_summary = []
            for f in corr_findings[:10]:
                if f.analysis_type == "bivariate":
                    corr_summary.append(f"{f.column}: r={f.stats.get('correlation', 0):.3f}")
                elif f.analysis_type == "correlation":
                    for hc in f.stats.get("high_correlations", [])[:3]:
                        corr_summary.append(f"{hc['col_a']} ↔ {hc['col_b']}: r={hc['correlation']:.3f}")

            result = self.llm_json(
                system=(
                    "You are a data scientist. Generate 2-4 testable hypotheses from the data. "
                    "Return JSON with a 'hypotheses' array, each with: id, statement, "
                    "supporting_evidence (array), suggested_validation."
                ),
                user=f"Correlations found:\n" + "\n".join(corr_summary),
            )

            hypotheses = []
            for h in result.get("hypotheses", []):
                hypotheses.append(Hypothesis(
                    id=h.get("id", f"H{len(hypotheses)+1}"),
                    statement=h.get("statement", ""),
                    supporting_evidence=h.get("supporting_evidence", []),
                    suggested_validation=h.get("suggested_validation", ""),
                ))
            return hypotheses

        except Exception as e:
            self.log(f"⚠️ Hypothesis generation failed: {e}", level="warning")
            return []

    def _recommend_features(self, findings: list[Finding],
                             target: str, df: pd.DataFrame) -> list[str]:
        """Recommend features for modeling."""
        recommended = []

        # Recommend features with high correlation to target
        for f in findings:
            if f.analysis_type == "bivariate":
                corr = abs(f.stats.get("correlation", 0))
                if corr > 0.1:
                    recommended.append(f.column)

        # Include all numeric columns not yet included
        for col in df.select_dtypes(include="number").columns:
            if col != target and col not in recommended:
                recommended.append(col)

        # Include low-cardinality categorical columns
        for col in df.select_dtypes(include="object").columns:
            if col != target and df[col].nunique() <= 20:
                recommended.append(col)

        return recommended
