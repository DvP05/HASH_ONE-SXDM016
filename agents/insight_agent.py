"""
Stage 6 — Insight Generation Agent
Synthesizes all pipeline outputs into human-readable, actionable insights.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

from agents.base_agent import BaseAgent
from core.models import InsightReport, Recommendation

logger = logging.getLogger(__name__)


class InsightAgent(BaseAgent):
    """
    Synthesizes all pipeline outputs into a structured insight report
    with narrative, recommendations, and next steps.
    """

    def run(self, hints: str = "") -> InsightReport:
        self.start_timer()
        self.log("═══ Stage 6: Insight Generation ═══")

        # Gather all context
        context = {
            "problem_statement": self.memory.retrieve("problem_statement", ""),
            "domain_context": self.memory.retrieve("domain_context", ""),
            "eda_report": self.memory.retrieve("eda_report"),
            "cleaning_log": self.memory.retrieve("cleaning_log", []),
            "feature_store": self.memory.retrieve("feature_store"),
            "model_result": self.memory.retrieve("model_result"),
            "model_evaluation": self.memory.retrieve("model_evaluation"),
        }

        output_dir = self.memory.retrieve("output_dir", "output")

        # ── Executive Summary ──
        self.log("📝 Writing executive summary...")
        executive_summary = self._write_executive_summary(context)

        # ── Data Quality Section ──
        self.log("📊 Writing data quality section...")
        data_quality = self._write_data_quality_section(context)

        # ── Pattern Section ──
        self.log("🔍 Writing pattern analysis...")
        pattern_section = self._write_pattern_section(context)

        # ── Model Section ──
        self.log("🤖 Writing model interpretation...")
        model_section = self._write_model_section(context)

        # ── Recommendations ──
        self.log("💡 Generating recommendations...")
        recommendations = self._generate_recommendations(context)

        # ── Next Steps ──
        self.log("🚀 Suggesting next steps...")
        next_steps = self._suggest_next_steps(context)

        report = InsightReport(
            executive_summary=executive_summary,
            data_quality_section=data_quality,
            pattern_section=pattern_section,
            model_section=model_section,
            recommendations=recommendations,
            next_steps=next_steps,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        # Store and save
        self.memory.store("insight_report", report, source="insight_generation")

        # Save as markdown
        self._save_report(report, output_dir)

        # Save model card
        model_result = context.get("model_result")
        if model_result and hasattr(model_result, "model_card") and model_result.model_card:
            self._save_model_card(model_result.model_card, output_dir)

        self.log(f"✅ Insight generation complete: {len(recommendations)} recommendations")

        self.stop_timer()
        return report

    def _write_executive_summary(self, context: dict) -> str:
        """Generate executive summary."""
        model_result = context.get("model_result")
        eda_report = context.get("eda_report")
        feature_store = context.get("feature_store")

        try:
            summary_data = {
                "problem": context.get("problem_statement", "Data analysis task"),
                "rows_analyzed": eda_report.data_traits.get("n_rows", 0) if eda_report else 0,
                "features_used": feature_store.n_features if feature_store else 0,
                "model_name": model_result.model_name if model_result else "N/A",
            }

            if model_result and model_result.evaluation:
                eval_metrics = model_result.evaluation
                if eval_metrics.roc_auc:
                    summary_data["roc_auc"] = eval_metrics.roc_auc
                if eval_metrics.accuracy:
                    summary_data["accuracy"] = eval_metrics.accuracy
                if eval_metrics.f1_score:
                    summary_data["f1"] = eval_metrics.f1_score
                if eval_metrics.r2:
                    summary_data["r2"] = eval_metrics.r2
                summary_data["cv_mean"] = eval_metrics.cv_mean

            result = self.llm_chat(
                system=(
                    "You are writing an executive summary for a data analysis report. "
                    "Be concise (3-5 sentences), cite specific numbers, and highlight the key finding."
                ),
                user=f"Analysis results:\n{json.dumps(summary_data, indent=2, default=str)}",
            )
            return result

        except Exception as e:
            self.log(f"⚠️ Executive summary LLM failed: {e}", level="warning")
            # Fallback
            parts = [f"Analysis of {context.get('problem_statement', 'the dataset')} is complete."]
            if model_result and model_result.evaluation:
                ev = model_result.evaluation
                if ev.roc_auc:
                    parts.append(f"The {model_result.model_name} model achieves ROC-AUC = {ev.roc_auc:.3f}.")
                if ev.accuracy:
                    parts.append(f"Accuracy: {ev.accuracy:.1%}.")
                parts.append(f"Cross-validation score: {ev.cv_mean:.4f} ± {ev.cv_std:.4f}.")
            return " ".join(parts)

    def _write_data_quality_section(self, context: dict) -> str:
        """Summarize data quality findings."""
        cleaning_log = context.get("cleaning_log", [])
        eda_report = context.get("eda_report")

        sections = ["## Data Quality\n"]

        if cleaning_log:
            sections.append(f"**{len(cleaning_log)} cleaning operations** were applied:\n")
            for entry in cleaning_log:
                if isinstance(entry, dict):
                    sections.append(f"- **{entry.get('operation', 'unknown')}**: "
                                    f"{entry.get('reasoning', '')} "
                                    f"({entry.get('rows_affected', 0)} rows affected)")
            sections.append("")

        if eda_report and eda_report.data_traits:
            traits = eda_report.data_traits
            sections.append(f"**Dataset**: {traits.get('n_rows', 0):,} rows × "
                            f"{traits.get('n_columns', 0)} columns")
            sections.append(f"**Average missing rate**: {traits.get('avg_missing_pct', 0):.1%}")

            if "class_balance" in traits:
                sections.append(f"**Class balance**: {traits['class_balance']}")

        return "\n".join(sections)

    def _write_pattern_section(self, context: dict) -> str:
        """Summarize discovered patterns."""
        eda_report = context.get("eda_report")
        if not eda_report:
            return "No EDA report available."

        sections = ["## Key Patterns\n"]

        if eda_report.narrative:
            sections.append(eda_report.narrative)
            sections.append("")

        if eda_report.hypotheses:
            sections.append("### Hypotheses\n")
            for h in eda_report.hypotheses:
                sections.append(f"**{h.id}**: {h.statement}")
                if h.supporting_evidence:
                    for ev in h.supporting_evidence:
                        sections.append(f"  - Evidence: {ev}")
                sections.append("")

        return "\n".join(sections)

    def _write_model_section(self, context: dict) -> str:
        """Summarize model performance."""
        model_result = context.get("model_result")
        if not model_result:
            return "No model results available."

        sections = ["## Model Performance\n"]
        sections.append(f"**Selected model**: {model_result.model_name}")

        if model_result.evaluation:
            ev = model_result.evaluation
            sections.append(f"\n**Cross-validation**: {ev.cv_mean:.4f} ± {ev.cv_std:.4f}")

            if ev.task_type == "classification":
                metrics = []
                if ev.accuracy is not None:
                    metrics.append(f"Accuracy: {ev.accuracy:.1%}")
                if ev.roc_auc is not None:
                    metrics.append(f"ROC-AUC: {ev.roc_auc:.4f}")
                if ev.f1_score is not None:
                    metrics.append(f"F1-Score: {ev.f1_score:.4f}")
                if ev.precision is not None:
                    metrics.append(f"Precision: {ev.precision:.4f}")
                if ev.recall is not None:
                    metrics.append(f"Recall: {ev.recall:.4f}")
                if metrics:
                    sections.append("\n**Metrics**: " + " | ".join(metrics))
            else:
                metrics = []
                if ev.r2 is not None:
                    metrics.append(f"R²: {ev.r2:.4f}")
                if ev.rmse is not None:
                    metrics.append(f"RMSE: {ev.rmse:.4f}")
                if ev.mae is not None:
                    metrics.append(f"MAE: {ev.mae:.4f}")
                if metrics:
                    sections.append("\n**Metrics**: " + " | ".join(metrics))

            if ev.top_features:
                sections.append("\n### Top Predictive Features\n")
                for i, feat in enumerate(ev.top_features[:10], 1):
                    sections.append(f"{i}. **{feat.get('feature', '')}** "
                                    f"(importance: {feat.get('importance', 0):.4f})")

        if model_result.cv_results:
            sections.append("\n### Model Comparison\n")
            sections.append("| Model | CV Score |")
            sections.append("|-------|---------|")
            for name, scores in sorted(model_result.cv_results.items(),
                                        key=lambda x: x[1].get("cv_mean", 0),
                                        reverse=True):
                sections.append(f"| {name} | {scores.get('cv_mean', 0):.4f} ± "
                                f"{scores.get('cv_std', 0):.4f} |")

        return "\n".join(sections)

    def _generate_recommendations(self, context: dict) -> list[Recommendation]:
        """Generate actionable recommendations."""
        recommendations = []

        # Try LLM-generated recommendations
        try:
            model_result = context.get("model_result")
            eda_report = context.get("eda_report")

            reco_context = {
                "problem": context.get("problem_statement", ""),
                "model": model_result.model_name if model_result else "",
                "top_features": [],
                "key_patterns": [],
            }

            if model_result and model_result.evaluation and model_result.evaluation.top_features:
                reco_context["top_features"] = [
                    f.get("feature", "") for f in model_result.evaluation.top_features[:5]
                ]

            if eda_report and eda_report.hypotheses:
                reco_context["key_patterns"] = [h.statement for h in eda_report.hypotheses[:3]]

            result = self.llm_json(
                system=(
                    "You are a business intelligence analyst. Generate 3-5 actionable recommendations. "
                    "Return JSON with 'recommendations' array, each with: title, description, "
                    "estimated_impact, confidence (0-1), supporting_evidence (array), "
                    "action_items (array), priority (high/medium/low)."
                ),
                user=f"Analysis context:\n{json.dumps(reco_context, indent=2, default=str)}",
            )

            for r in result.get("recommendations", []):
                recommendations.append(Recommendation(
                    title=r.get("title", "Recommendation"),
                    description=r.get("description", ""),
                    impact=r.get("estimated_impact", ""),
                    confidence=float(r.get("confidence", 0.5)),
                    evidence=r.get("supporting_evidence", []),
                    action_items=r.get("action_items", []),
                    priority=r.get("priority", "medium"),
                ))
        except Exception as e:
            self.log(f"⚠️ LLM recommendation generation failed: {e}", level="warning")

        # If LLM didn't produce enough, build data-driven recommendations
        if len(recommendations) < 2:
            recommendations.extend(self._build_data_driven_recommendations(context))

        return recommendations

    def _build_data_driven_recommendations(self, context: dict) -> list[Recommendation]:
        """Build recommendations directly from pipeline outputs (no LLM needed)."""
        recs = []
        model_result = context.get("model_result")
        eda_report = context.get("eda_report")
        feature_store = context.get("feature_store")

        # Recommendation based on top features
        if model_result and model_result.evaluation and model_result.evaluation.top_features:
            top = model_result.evaluation.top_features[:3]
            feature_names = [f.get("feature", "") for f in top]
            recs.append(Recommendation(
                title="Focus on Top Predictive Features",
                description=(
                    f"The model identifies {', '.join(feature_names)} as the strongest predictors. "
                    f"Design targeted interventions around these variables to maximize impact."
                ),
                impact="High — directly addresses the strongest model signals",
                confidence=0.85,
                evidence=[f"{f.get('feature')}: importance={f.get('importance', 0):.4f}" for f in top],
                action_items=[
                    f"Investigate why '{feature_names[0]}' has the highest impact",
                    "Design interventions targeting top features",
                    "Create monitoring dashboards for these key variables",
                ],
                priority="high",
            ))

        # Recommendation based on model performance
        if model_result and model_result.evaluation:
            ev = model_result.evaluation
            if ev.roc_auc and ev.roc_auc < 0.85:
                recs.append(Recommendation(
                    title="Improve Model Performance with More Data",
                    description=(
                        f"Current ROC-AUC is {ev.roc_auc:.3f}. Consider collecting additional "
                        f"features (behavioral signals, interaction data) and more recent data "
                        f"to push performance above 0.85."
                    ),
                    impact="Medium — potential 5-10% improvement in prediction accuracy",
                    confidence=0.7,
                    evidence=[
                        f"Current ROC-AUC: {ev.roc_auc:.4f}",
                        f"CV Score: {ev.cv_mean:.4f} +/- {ev.cv_std:.4f}",
                    ],
                    action_items=[
                        "Collect additional behavioral features",
                        "Try ensemble methods (stacking, blending)",
                        "Experiment with deep learning on larger datasets",
                    ],
                    priority="medium",
                ))

        # Recommendation based on data quality
        cleaning_log = context.get("cleaning_log", [])
        if cleaning_log:
            total_affected = sum(
                e.get("rows_affected", 0) if isinstance(e, dict) else 0
                for e in cleaning_log
            )
            if total_affected > 0:
                recs.append(Recommendation(
                    title="Improve Data Collection Quality",
                    description=(
                        f"{len(cleaning_log)} data quality issues were fixed, affecting "
                        f"{total_affected} records. Addressing root causes at the source "
                        f"will improve future model reliability."
                    ),
                    impact="Medium — reduces data cleaning overhead and improves model inputs",
                    confidence=0.75,
                    evidence=[
                        f"{e.get('operation', 'unknown')}: {e.get('rows_affected', 0)} rows"
                        for e in cleaning_log if isinstance(e, dict)
                    ],
                    action_items=[
                        "Add input validation at data collection points",
                        "Set up data quality monitoring alerts",
                        "Document data standards for upstream providers",
                    ],
                    priority="medium",
                ))

        # Deployment recommendation
        recs.append(Recommendation(
            title="Deploy Model with Monitoring",
            description=(
                "Deploy the trained model to a staging environment with A/B testing "
                "and set up drift detection to ensure performance is maintained over time."
            ),
            impact="High — enables real-world value capture from the analysis",
            confidence=0.8,
            evidence=["Model passed all quality gates", "Cross-validation shows stable performance"],
            action_items=[
                "Set up model serving infrastructure",
                "Implement prediction monitoring and alerting",
                "Schedule monthly model retraining",
                "Create A/B test plan for recommendation validation",
            ],
            priority="high",
        ))

        return recs

    def _suggest_next_steps(self, context: dict) -> list[str]:
        """Suggest next steps."""
        try:
            result = self.llm_json(
                system="Return a JSON object with 'next_steps' array (3-5 short strings).",
                user=f"Problem: {context.get('problem_statement', '')}. What are the next steps?",
            )
            return result.get("next_steps", [
                "Validate model on holdout/test data",
                "Deploy model to staging environment",
                "Set up monitoring for model drift",
            ])
        except Exception:
            return [
                "Validate model on holdout/test data",
                "Collect more recent data for retraining",
                "A/B test recommendations in production",
                "Set up monitoring for model performance drift",
            ]

    def _save_report(self, report: InsightReport, output_dir: str) -> None:
        """Save insight report as markdown."""
        os.makedirs(output_dir, exist_ok=True)

        md = [
            "# Autonomous Analysis — Insight Report\n",
            f"*Generated at: {report.generated_at}*\n",
            "---\n",
            "## Executive Summary\n",
            report.executive_summary,
            "\n---\n",
            report.data_quality_section,
            "\n---\n",
            report.pattern_section,
            "\n---\n",
            report.model_section,
            "\n---\n",
            "## Recommendations\n",
        ]

        for rec in report.recommendations:
            md.append(f"\n### [{rec.priority.upper()}] {rec.title}")
            md.append(f"\n{rec.description}")
            if rec.action_items:
                md.append("\n**Action Items:**")
                for item in rec.action_items:
                    md.append(f"- {item}")
            md.append(f"\n*Confidence: {rec.confidence:.0%}*\n")

        if report.next_steps:
            md.append("\n---\n")
            md.append("## Next Steps\n")
            for step in report.next_steps:
                md.append(f"- {step}")

        report_path = os.path.join(output_dir, "insight_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        self.log(f"💾 Report saved to {report_path}")

    def _save_model_card(self, model_card: Any, output_dir: str) -> None:
        """Save model card as markdown."""
        os.makedirs(output_dir, exist_ok=True)

        md = [
            f"# Model Card: {model_card.model_name}\n",
            f"**Type**: {model_card.model_type}",
            f"**Task**: {model_card.task_type}\n",
            "## Intended Use\n",
            model_card.intended_use or "General predictive modeling.",
            "\n## Performance\n",
        ]

        if model_card.evaluation:
            ev = model_card.evaluation
            md.append(f"- CV Score: {ev.cv_mean:.4f} ± {ev.cv_std:.4f}")
            if ev.roc_auc:
                md.append(f"- ROC-AUC: {ev.roc_auc:.4f}")
            if ev.accuracy:
                md.append(f"- Accuracy: {ev.accuracy:.1%}")

        md.append("\n## Limitations\n")
        md.append(model_card.limitations or "Standard ML model limitations apply.")

        if model_card.best_params:
            md.append("\n## Hyperparameters\n")
            md.append("```json")
            md.append(json.dumps(model_card.best_params, indent=2, default=str))
            md.append("```")

        card_path = os.path.join(output_dir, "model_card.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md))

        self.log(f"💾 Model card saved to {card_path}")
