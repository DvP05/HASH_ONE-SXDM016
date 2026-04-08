"""
Machine learning tools.
Model training, hyperparameter optimization, SHAP explainability.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, mean_absolute_error, mean_squared_error,
    precision_score, recall_score, r2_score, roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tools.registry import tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Model Factories
# ─────────────────────────────────────────────

MODEL_REGISTRY = {}


def _register_models():
    """Register available model factories."""
    global MODEL_REGISTRY

    MODEL_REGISTRY["logistic_regression"] = {
        "class": LogisticRegression,
        "default_params": {"max_iter": 1000, "random_state": 42},
        "task": "classification",
    }
    MODEL_REGISTRY["random_forest_clf"] = {
        "class": RandomForestClassifier,
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "task": "classification",
    }
    MODEL_REGISTRY["random_forest_reg"] = {
        "class": RandomForestRegressor,
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "task": "regression",
    }
    MODEL_REGISTRY["ridge"] = {
        "class": Ridge,
        "default_params": {"alpha": 1.0},
        "task": "regression",
    }

    # Try to import XGBoost
    try:
        from xgboost import XGBClassifier, XGBRegressor
        MODEL_REGISTRY["xgboost_clf"] = {
            "class": XGBClassifier,
            "default_params": {
                "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                "random_state": 42,
                "eval_metric": "logloss"
            },
            "task": "classification",
        }
        MODEL_REGISTRY["xgboost_reg"] = {
            "class": XGBRegressor,
            "default_params": {
                "n_estimators": 100, "max_depth": 6, "learning_rate": 0.1,
                "random_state": 42,
            },
            "task": "regression",
        }
    except ImportError:
        logger.info("XGBoost not available, skipping")

    # Try to import LightGBM
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
        MODEL_REGISTRY["lightgbm_clf"] = {
            "class": LGBMClassifier,
            "default_params": {
                "n_estimators": 100, "max_depth": -1, "learning_rate": 0.1,
                "random_state": 42, "verbose": -1,
            },
            "task": "classification",
        }
        MODEL_REGISTRY["lightgbm_reg"] = {
            "class": LGBMRegressor,
            "default_params": {
                "n_estimators": 100, "max_depth": -1, "learning_rate": 0.1,
                "random_state": 42, "verbose": -1,
            },
            "task": "regression",
        }
    except ImportError:
        logger.info("LightGBM not available, skipping")


_register_models()


# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────

@tool(name="train_model", description="Train a scikit-learn compatible model", category="ml")
def train_model(X: pd.DataFrame, y: pd.Series, model_name: str = "random_forest_clf",
                params: dict | None = None, task_type: str = "classification") -> dict:
    """
    Train a model and return evaluation metrics.
    """
    # Auto-select model variant
    if model_name in ("random_forest", "xgboost", "lightgbm"):
        suffix = "_clf" if task_type == "classification" else "_reg"
        model_name = model_name + suffix

    if model_name not in MODEL_REGISTRY:
        # Fallback
        model_name = "random_forest_clf" if task_type == "classification" else "random_forest_reg"

    model_info = MODEL_REGISTRY[model_name]
    model_params = {**model_info["default_params"]}
    if params:
        model_params.update(params)

    model_class = model_info["class"]
    model = model_class(**model_params)

    # Prepare data
    X_clean = X.copy()
    y_clean = y.copy()
    # Handle categorical columns
    label_encoders = {}
    for col in X_clean.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_clean[col] = le.fit_transform(X_clean[col].astype(str))
        label_encoders[col] = le

    # Encode classification target if needed (required by some estimators like XGBoost)
    target_encoder = None
    y_clean = y.copy()
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y_clean = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)

    # Fill remaining NaNs
    X_clean = X_clean.fillna(0)

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = X_clean.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        X_clean[numeric_cols] = scaler.fit_transform(X_clean[numeric_cols])

    # Cross-validation
    if task_type == "classification":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        n_classes = int(pd.Series(y_clean).nunique())
        scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scoring = "r2"

    cv_scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring=scoring)

    # Train on full dataset
    model.fit(X_clean, y_clean)
    y_pred = model.predict(X_clean)

    result = {
        "model_name": model_name,
        "model": model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "target_encoder_classes": target_encoder.classes_.tolist() if target_encoder else [],
        "cv_mean": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "cv_scores": [round(float(s), 4) for s in cv_scores],
        "params": model_params,
    }

    if task_type == "classification":
        result.update({
            "accuracy": round(float(accuracy_score(y_clean, y_pred)), 4),
            "f1_score": round(float(f1_score(y_clean, y_pred, average="weighted")), 4),
            "precision": round(float(precision_score(y_clean, y_pred, average="weighted", zero_division=0)), 4),
            "recall": round(float(recall_score(y_clean, y_pred, average="weighted", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_clean, y_pred).tolist(),
        })
        # ROC-AUC
        n_classes = int(pd.Series(y_clean).nunique())
        if n_classes == 2:
            try:
                y_proba = model.predict_proba(X_clean)[:, 1]
                result["roc_auc"] = round(float(roc_auc_score(y_clean, y_proba)), 4)
            except Exception:
                result["roc_auc"] = result["cv_mean"]
        elif n_classes > 2:
            try:
                y_proba = model.predict_proba(X_clean)
                result["roc_auc"] = round(float(
                    roc_auc_score(y_clean, y_proba, multi_class="ovr", average="weighted")
                ), 4)
            except Exception:
                result["roc_auc"] = result["cv_mean"]
    else:
        result.update({
            "r2": round(float(r2_score(y_clean, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_clean, y_pred))), 4),
            "mae": round(float(mean_absolute_error(y_clean, y_pred)), 4),
        })

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=X_clean.columns)
        importance = importance.sort_values(ascending=False)
        result["feature_importance"] = {
            str(k): round(float(v), 4) for k, v in importance.head(20).items()
        }

    logger.info(f"Trained {model_name}: CV={result['cv_mean']:.4f}±{result['cv_std']:.4f}")
    return result


@tool(name="optimize_hyperparams", description="Bayesian hyperparameter optimization with Optuna", category="ml")
def optimize_hyperparams(X: pd.DataFrame, y: pd.Series,
                         model_name: str = "random_forest_clf",
                         n_trials: int = 30,
                         task_type: str = "classification") -> dict:
    """
    Run Optuna hyperparameter optimization.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        logger.warning("Optuna not installed, using default hyperparameters")
        return train_model(X, y, model_name, task_type=task_type)

    # Prepare data
    X_clean = X.copy()
    y_clean = y.copy()
    for col in X_clean.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_clean[col] = le.fit_transform(X_clean[col].astype(str))
    X_clean = X_clean.fillna(0)

    if task_type == "classification":
        y_clean = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)

    def objective(trial):
        if "random_forest" in model_name:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42,
                "n_jobs": -1,
            }
            if task_type == "classification":
                model = RandomForestClassifier(**params)
            else:
                model = RandomForestRegressor(**params)
        elif "xgboost" in model_name:
            try:
                from xgboost import XGBClassifier, XGBRegressor
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "random_state": 42,
                }
                if task_type == "classification":
                    params["eval_metric"] = "logloss"
                    model = XGBClassifier(**params)
                else:
                    model = XGBRegressor(**params)
            except ImportError:
                model = RandomForestClassifier(random_state=42) if task_type == "classification" else RandomForestRegressor(random_state=42)
        else:
            if task_type == "classification":
                params = {
                    "C": trial.suggest_float("C", 0.001, 100, log=True),
                    "max_iter": 1000,
                    "random_state": 42,
                }
                model = LogisticRegression(**params)
            else:
                params = {"alpha": trial.suggest_float("alpha", 0.001, 100, log=True)}
                model = Ridge(**params)

        if task_type == "classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            n_classes = int(pd.Series(y_clean).nunique())
            scoring = "roc_auc" if n_classes == 2 else "roc_auc_ovr_weighted"
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "r2"

        scores = cross_val_score(model, X_clean, y_clean, cv=cv, scoring=scoring)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"Optuna best params: {best_params} (score: {study.best_value:.4f})")

    # Retrain with best params
    return train_model(X, y, model_name, params=best_params, task_type=task_type)


@tool(name="shap_explain", description="Compute SHAP feature importance", category="ml")
def shap_explain(model: Any, X: pd.DataFrame,
                 output_dir: str = "output/charts") -> dict:
    """
    Compute SHAP values and save a summary plot.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed, computing feature importance from model")
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=X.columns)
            importance = importance.sort_values(ascending=False)
            return {
                "top_features": [
                    {"feature": str(k), "importance": round(float(v), 4)}
                    for k, v in importance.head(20).items()
                ],
                "method": "feature_importances_",
            }
        return {"top_features": [], "method": "unavailable"}

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    X_clean = X.copy()
    for col in X_clean.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_clean[col] = le.fit_transform(X_clean[col].astype(str))
    X_clean = X_clean.fillna(0)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_clean.head(500))

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        # Mean absolute SHAP values
        mean_shap = np.abs(shap_values).mean(axis=0)
        importance = pd.Series(mean_shap, index=X_clean.columns)
        importance = importance.sort_values(ascending=False)

        # Save SHAP summary plot
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            shap.summary_plot(shap_values, X_clean.head(500), show=False,
                              max_display=15)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "shap_summary.png"), dpi=150,
                        bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not save SHAP plot: {e}")

        return {
            "top_features": [
                {"feature": str(k), "importance": round(float(v), 4)}
                for k, v in importance.head(20).items()
            ],
            "method": "shap_tree_explainer",
            "plot_path": os.path.join(output_dir, "shap_summary.png"),
        }

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        if hasattr(model, "feature_importances_"):
            importance = pd.Series(model.feature_importances_, index=X_clean.columns)
            importance = importance.sort_values(ascending=False)
            return {
                "top_features": [
                    {"feature": str(k), "importance": round(float(v), 4)}
                    for k, v in importance.head(20).items()
                ],
                "method": "feature_importances_fallback",
            }
        return {"top_features": [], "method": "failed"}
