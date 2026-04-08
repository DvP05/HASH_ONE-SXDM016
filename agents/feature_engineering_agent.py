"""
Stage 4 — Feature Engineering Agent
Automatically generates, selects, and validates features
for the predictive model.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from agents.base_agent import BaseAgent
from core.models import FeatureEngineeringPlan, FeatureStore, FeatureTransform

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseAgent):
    """
    Generates and selects features using a combination of
    automated transforms and LLM-guided domain reasoning.
    """

    def run(self, hints: str = "") -> FeatureStore:
        self.start_timer()
        self.log("═══ Stage 4: Feature Engineering ═══")

        df = self.memory.retrieve("clean_dataset")
        if df is None:
            raise ValueError("No clean_dataset in memory. Run DataCleaningAgent first.")

        target = self.memory.retrieve("target_column", "")
        task_type = self.memory.retrieve("task_type", "classification")
        eda_report = self.memory.retrieve("eda_report")

        if not target or target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset.")

        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]

        # ── Apply Transforms ──
        self.log("⚙️ Applying feature transforms...")
        X_transformed = self._apply_transforms(X, y, task_type, eda_report)

        # ── Feature Selection ──
        self.log("🔍 Selecting best features...")
        selected_features, importance_scores = self._select_features(
            X_transformed, y, task_type
        )

        # Build feature set
        X_final = X_transformed[selected_features]

        feature_store = FeatureStore(
            feature_names=selected_features,
            lineage={col: self._get_lineage(col, X.columns.tolist()) for col in selected_features},
            importance_scores=importance_scores,
            n_features=len(selected_features),
            n_samples=len(X_final),
        )

        # Store in memory
        self.memory.store("feature_store", feature_store, source="feature_engineering")
        self.memory.store("X_features", X_final, source="feature_engineering")
        self.memory.store("y_target", y, source="feature_engineering")

        self.log(f"✅ Feature engineering complete: {len(selected_features)} features selected")

        self.stop_timer()
        return feature_store

    def _apply_transforms(self, X: pd.DataFrame, y: pd.Series,
                           task_type: str,
                           eda_report: Any = None) -> pd.DataFrame:
        """Apply feature transforms."""
        X_out = X.copy()

        # 1. Encode categorical variables
        self.log("  📝 Encoding categorical variables...")
        for col in X_out.select_dtypes(include=["object", "category"]).columns:
            n_unique = X_out[col].nunique()
            if n_unique == 2:
                # Binary encoding
                le = LabelEncoder()
                X_out[col] = le.fit_transform(X_out[col].astype(str))
            elif n_unique <= 10:
                # One-hot encoding
                dummies = pd.get_dummies(X_out[col], prefix=col, drop_first=True)
                X_out = X_out.drop(columns=[col])
                X_out = pd.concat([X_out, dummies], axis=1)
            else:
                # Label encoding for high cardinality
                le = LabelEncoder()
                X_out[col] = le.fit_transform(X_out[col].astype(str))

        # 2. Log-transform highly skewed numeric features
        self.log("  📐 Log-transforming skewed features...")
        for col in X_out.select_dtypes(include="number").columns:
            if len(X_out[col].dropna()) > 2:
                skew = X_out[col].skew()
                if abs(skew) > 2.0 and X_out[col].min() >= 0:
                    new_col = f"log_{col}"
                    X_out[new_col] = np.log1p(X_out[col])
                    self.log(f"    ✅ Created {new_col} (skew: {skew:.2f})")

        # 3. Create interaction features for highly correlated pairs
        if eda_report and hasattr(eda_report, 'findings'):
            self.log("  🔗 Creating interaction features...")
            corr_findings = [f for f in eda_report.findings if f.analysis_type == "correlation"]
            for f in corr_findings:
                for pair in f.stats.get("high_correlations", [])[:3]:
                    col_a = pair.get("col_a", "")
                    col_b = pair.get("col_b", "")
                    if col_a in X_out.columns and col_b in X_out.columns:
                        if pd.api.types.is_numeric_dtype(X_out[col_a]) and \
                           pd.api.types.is_numeric_dtype(X_out[col_b]):
                            new_col = f"{col_a}_x_{col_b}"
                            X_out[new_col] = X_out[col_a] * X_out[col_b]
                            self.log(f"    ✅ Created interaction: {new_col}")

        # 4. Fill remaining NaN
        X_out = X_out.fillna(0)

        self.log(f"  📦 Transformed: {X.shape[1]} → {X_out.shape[1]} features")
        return X_out

    def _select_features(self, X: pd.DataFrame, y: pd.Series,
                          task_type: str) -> tuple[list[str], dict[str, float]]:
        """
        Multi-method feature selection.
        Uses mutual information + correlation filtering.
        """
        X_numeric = X.select_dtypes(include="number").fillna(0)

        if len(X_numeric.columns) == 0:
            return list(X.columns), {}

        # 1. Mutual Information
        try:
            y_encoded = y.copy()
            if not pd.api.types.is_numeric_dtype(y):
                le = LabelEncoder()
                y_encoded = pd.Series(le.fit_transform(y.astype(str)))

            if task_type == "classification":
                mi_scores = mutual_info_classif(X_numeric, y_encoded, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_numeric, y_encoded, random_state=42)

            mi_ranking = pd.Series(mi_scores, index=X_numeric.columns).sort_values(ascending=False)

        except Exception as e:
            self.log(f"⚠️ Mutual information failed: {e}", level="warning")
            mi_ranking = pd.Series(1.0, index=X_numeric.columns)

        # 2. Correlation-based filtering (remove redundant)
        corr_matrix = X_numeric.corr().abs()
        redundant = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    # Remove the one with lower MI
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    if mi_ranking.get(col_j, 0) < mi_ranking.get(col_i, 0):
                        redundant.add(col_j)
                    else:
                        redundant.add(col_i)

        if redundant:
            self.log(f"  🗑️ Removed {len(redundant)} redundant features (r > 0.95)")

        # 3. Select top features (max 50, or all if fewer)
        candidates = [col for col in mi_ranking.index if col not in redundant]
        max_features = min(50, len(candidates))
        selected = candidates[:max_features]

        # Filter out features with zero MI
        selected = [col for col in selected if mi_ranking.get(col, 0) > 0.001] or selected[:10]

        importance_scores = {
            str(col): round(float(mi_ranking.get(col, 0)), 4) for col in selected
        }

        self.log(f"  ✅ Selected {len(selected)}/{len(X_numeric.columns)} features")

        return selected, importance_scores

    def _get_lineage(self, col: str, original_columns: list[str]) -> str:
        """Track where a feature came from."""
        if col in original_columns:
            return f"original:{col}"
        if col.startswith("log_"):
            return f"log_transform:{col[4:]}"
        if "_x_" in col:
            parts = col.split("_x_")
            return f"interaction:{parts[0]}*{parts[1]}"
        if "__" in col or col.count("_") > 1:
            return f"encoded:{col}"
        return f"derived:{col}"
