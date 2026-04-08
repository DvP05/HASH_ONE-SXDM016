"""
Stage 5 — Predictive Modeling Agent
Selects, trains, tunes, and evaluates the best predictive model(s).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.models import ModelCard, ModelEvaluation, ModelingResult

logger = logging.getLogger(__name__)


class ModelingAgent(BaseAgent):
    """
    Trains and selects the best model via multi-algorithm
    competition with optional Bayesian hyperparameter optimization.
    """

    def run(self, hints: str = "") -> ModelingResult:
        self.start_timer()
        self.log("═══ Stage 5: Predictive Modeling ═══")

        X = self.memory.retrieve("X_features")
        y = self.memory.retrieve("y_target")
        task_type = self.memory.retrieve("task_type", "classification")
        target = self.memory.retrieve("target_column", "")
        output_dir = self.memory.retrieve("output_dir", "output")
        charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)

        if X is None or y is None:
            raise ValueError("No features in memory. Run FeatureEngineeringAgent first.")

        # ── Select Candidate Models ──
        candidates = self._get_candidate_models(X, y, task_type)
        self.log(f"🏆 Candidate models: {candidates}")

        # ── Train All Candidates ──
        results = {}
        for model_name in candidates:
            self.log(f"🚀 Training: {model_name}...")
            try:
                result = self.execute_tool(
                    "train_model", X=X, y=y,
                    model_name=model_name, task_type=task_type,
                )
                results[model_name] = result
                cv_score = result.get("cv_mean", 0)
                self.log(f"   📈 {model_name}: CV={cv_score:.4f} ± {result.get('cv_std', 0):.4f}")
            except Exception as e:
                self.log(f"   ❌ {model_name} failed: {e}", level="warning")

        if not results:
            raise ValueError("All model candidates failed to train!")

        # ── Select Best Model ──
        best_name = max(results, key=lambda k: results[k].get("cv_mean", 0))
        best_result = results[best_name]
        self.log(f"🥇 Best model: {best_name} (CV={best_result['cv_mean']:.4f})")

        # ── Hyperparameter Optimization ──
        self.log(f"🔧 Optimizing hyperparameters for {best_name}...")
        try:
            optimized = self.execute_tool(
                "optimize_hyperparams", X=X, y=y,
                model_name=best_name, n_trials=30, task_type=task_type,
            )
            if optimized.get("cv_mean", 0) > best_result.get("cv_mean", 0):
                self.log(f"   📈 Improved: {best_result['cv_mean']:.4f} → {optimized['cv_mean']:.4f}")
                best_result = optimized
            else:
                self.log(f"   ↔️ No improvement from optimization")
        except Exception as e:
            self.log(f"   ⚠️ Optimization skipped: {e}", level="warning")

        # ── SHAP Explainability ──
        self.log("🔍 Computing feature importance (SHAP)...")
        shap_result = {}
        model_obj = best_result.get("model")
        if model_obj is not None:
            try:
                shap_result = self.execute_tool(
                    "shap_explain", model=model_obj, X=X, output_dir=charts_dir,
                )
            except Exception as e:
                self.log(f"   ⚠️ SHAP failed: {e}", level="warning")

        # Use model's feature importance as fallback
        if not shap_result.get("top_features") and best_result.get("feature_importance"):
            shap_result["top_features"] = [
                {"feature": k, "importance": v}
                for k, v in list(best_result["feature_importance"].items())[:20]
            ]

        # ── Feature Importance Chart ──
        if best_result.get("feature_importance"):
            try:
                self.execute_tool(
                    "generate_feature_importance_chart",
                    importance=best_result["feature_importance"],
                    title=f"Feature Importance ({best_name})",
                    output_path=os.path.join(charts_dir, "feature_importance.html"),
                )
            except Exception as e:
                self.log(f"   ⚠️ Feature importance chart failed: {e}", level="warning")

        # ── Build Evaluation ──
        evaluation = self._build_evaluation(best_result, task_type, shap_result)

        # ── Model Card ──
        model_card = self._generate_model_card(best_name, best_result, evaluation)

        # ── Build Result ──
        modeling_result = ModelingResult(
            model_name=best_name,
            evaluation=evaluation,
            cv_results={k: {"cv_mean": v.get("cv_mean", 0), "cv_std": v.get("cv_std", 0)}
                        for k, v in results.items()},
            model_card=model_card,
            shap_summary_path=shap_result.get("plot_path", ""),
            feature_importance_path=os.path.join(charts_dir, "feature_importance.html"),
        )

        # Store in memory
        self.memory.store("model_result", modeling_result, source="modeling")
        self.memory.store("trained_model", model_obj, source="modeling")
        self.memory.store("model_evaluation", evaluation, source="modeling")

        # Save feature importance
        if best_result.get("feature_importance"):
            fi_path = os.path.join(output_dir, "feature_importance.json")
            with open(fi_path, "w") as f:
                json.dump(best_result["feature_importance"], f, indent=2)

        self.log(f"✅ Modeling complete: {best_name} (CV={best_result['cv_mean']:.4f})")

        self.stop_timer()
        return modeling_result

    def _get_candidate_models(self, X: pd.DataFrame, y: pd.Series,
                               task_type: str) -> list[str]:
        """Get candidate models based on data characteristics."""
        if task_type == "classification":
            candidates = ["random_forest", "logistic_regression"]
            # Add XGBoost if available
            try:
                import xgboost
                candidates.append("xgboost")
            except ImportError:
                pass
            # Add LightGBM if available
            try:
                import lightgbm
                candidates.append("lightgbm")
            except ImportError:
                pass
        else:
            candidates = ["random_forest", "ridge"]
            try:
                import xgboost
                candidates.append("xgboost")
            except ImportError:
                pass

        return candidates

    def _build_evaluation(self, result: dict, task_type: str,
                           shap_result: dict) -> ModelEvaluation:
        """Build a ModelEvaluation from training results."""
        eval_data = {
            "task_type": task_type,
            "cv_mean": result.get("cv_mean", 0),
            "cv_std": result.get("cv_std", 0),
            "top_features": shap_result.get("top_features", []),
        }

        if task_type == "classification":
            eval_data.update({
                "accuracy": result.get("accuracy"),
                "roc_auc": result.get("roc_auc"),
                "f1_score": result.get("f1_score"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "confusion_matrix": result.get("confusion_matrix"),
            })
        else:
            eval_data.update({
                "r2": result.get("r2"),
                "rmse": result.get("rmse"),
                "mae": result.get("mae"),
            })

        return ModelEvaluation(**eval_data)

    def _generate_model_card(self, model_name: str, result: dict,
                              evaluation: ModelEvaluation) -> ModelCard:
        """Generate a model card."""
        problem = self.memory.retrieve("problem_statement", "")
        domain = self.memory.retrieve("domain_context", "")

        # LLM-generated limitations
        try:
            limitations = self.llm_chat(
                system="You are a ML engineer. List 2-3 key limitations of this model in 2-3 sentences.",
                user=(
                    f"Model: {model_name}\n"
                    f"Task: {evaluation.task_type}\n"
                    f"CV Score: {result.get('cv_mean', 0):.4f}\n"
                    f"Domain: {domain}"
                ),
            )
        except Exception:
            limitations = "Model trained on historical data; may not generalize to distribution shifts."

        return ModelCard(
            model_name=model_name,
            model_type=model_name.split("_")[0] if "_" in model_name else model_name,
            task_type=evaluation.task_type,
            evaluation=evaluation,
            training_data_summary=f"{result.get('cv_mean', 0):.4f} mean CV score",
            intended_use=problem,
            limitations=limitations,
            best_params=result.get("params", {}),
        )
