"""
Stage 2 — Data Cleaning Agent
Detects and remediates data quality issues autonomously.
Every action is justified, reversible, and auditable.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from core.models import (
    CleaningLogEntry, CleaningOperation, CleaningPlan,
    CleaningResult, DataProfile,
)
from core.sandbox import safe_exec_simple

logger = logging.getLogger(__name__)


class DataCleaningAgent(BaseAgent):
    """
    Autonomously cleans data using a combination of statistical
    methods and LLM-guided decision making.
    """

    def run(self, hints: str = "") -> CleaningResult:
        self.start_timer()
        self.log("═══ Stage 2: Data Cleaning ═══")

        df = self.memory.retrieve("raw_dataset")
        if df is None:
            raise ValueError("No raw_dataset in memory. Run DataCollectionAgent first.")

        # Profile the data
        profile_before = self._profile_data(df)
        self.memory.store("data_profile_before", profile_before, source="data_cleaning")
        self.log(f"📊 Data profile: {profile_before.shape[0]} rows × {profile_before.shape[1]} cols")
        self.log(f"   Missing: {sum(1 for v in profile_before.missing_pct.values() if v > 0)} columns have missing values")
        self.log(f"   Duplicates: {profile_before.duplicate_count}")
        self.log(f"   Outliers: {sum(profile_before.outlier_flags.values())} total outlier values")

        # Get cleaning plan from LLM
        cleaning_plan = self._get_cleaning_plan(df, profile_before)

        # Execute cleaning operations
        df_clean = df.copy()
        change_log: list[CleaningLogEntry] = []

        for op in cleaning_plan.operations:
            self.log(f"🧹 Applying: {op.name} ({op.strategy})")
            try:
                rows_before = len(df_clean)
                df_clean = self._apply_operation(df_clean, op)
                rows_after = len(df_clean)
                rows_affected = abs(rows_before - rows_after)

                # Count changed values if rows didn't change
                if rows_affected == 0 and op.target_column and op.target_column in df.columns:
                    rows_affected = int((df_clean[op.target_column] != df[op.target_column]).sum()) if op.target_column in df_clean.columns else 0

                change_log.append(CleaningLogEntry(
                    operation=op.name,
                    reasoning=op.llm_justification,
                    rows_affected=rows_affected,
                    reversible=True,
                ))
                self.log(f"   ✅ {op.name}: {rows_affected} rows affected")

            except Exception as e:
                self.log(f"   ⚠️ {op.name} failed: {e}", level="warning")
                change_log.append(CleaningLogEntry(
                    operation=op.name,
                    reasoning=f"Failed: {e}",
                    rows_affected=0,
                    reversible=True,
                ))

        # Profile after cleaning
        profile_after = self._profile_data(df_clean)

        # Store results
        self.memory.store("clean_dataset", df_clean, source="data_cleaning")
        self.memory.store("cleaning_log", [entry.model_dump() for entry in change_log],
                          source="data_cleaning")
        self.memory.store("data_profile_after", profile_after, source="data_cleaning")

        self.log(f"✅ Cleaning complete: {df.shape[0]}→{df_clean.shape[0]} rows, "
                 f"{len(change_log)} operations applied")

        self.stop_timer()
        return CleaningResult(
            log=change_log,
            profile_before=profile_before,
            profile_after=profile_after,
        )

    def _profile_data(self, df: pd.DataFrame) -> DataProfile:
        """Generate a comprehensive data quality profile."""
        outlier_flags = {}
        for col in df.select_dtypes(include="number").columns:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
                if n_outliers > 0:
                    outlier_flags[col] = n_outliers

        numeric_stats = {}
        for col in df.select_dtypes(include="number").columns:
            desc = df[col].describe()
            numeric_stats[col] = {
                "mean": round(float(desc.get("mean", 0)), 4),
                "std": round(float(desc.get("std", 0)), 4),
                "min": float(desc.get("min", 0)),
                "max": float(desc.get("max", 0)),
                "skewness": round(float(df[col].skew()), 4) if len(df[col].dropna()) > 2 else 0,
            }

        return DataProfile(
            shape=(df.shape[0], df.shape[1]),
            dtypes={col: str(dt) for col, dt in df.dtypes.items()},
            missing_pct={col: round(float(df[col].isnull().mean()), 4) for col in df.columns},
            cardinality={col: int(df[col].nunique()) for col in df.columns},
            outlier_flags=outlier_flags,
            duplicate_count=int(df.duplicated().sum()),
            sample={col: [str(v) for v in df[col].dropna().head(3).tolist()] for col in df.columns},
            numeric_stats=numeric_stats,
        )

    def _get_cleaning_plan(self, df: pd.DataFrame, profile: DataProfile) -> CleaningPlan:
        """Get cleaning plan via LLM or rule-based fallback."""
        try:
            profile_summary = (
                f"Shape: {profile.shape}\n"
                f"Missing: {dict(sorted(profile.missing_pct.items(), key=lambda x: x[1], reverse=True)[:10])}\n"
                f"Duplicates: {profile.duplicate_count}\n"
                f"Outliers: {dict(list(profile.outlier_flags.items())[:10])}\n"
                f"Dtypes: {profile.dtypes}\n"
                f"Cardinality: {dict(sorted(profile.cardinality.items(), key=lambda x: x[1])[:10])}"
            )

            domain_context = self.memory.retrieve("domain_context", "")
            result = self.llm_json(
                system=(
                    "You are a data cleaning expert. Generate a cleaning plan as JSON with an 'operations' array. "
                    "Each operation should have: name, target_column, strategy, llm_justification, and code. "
                    "The code should use pandas on a DataFrame called 'df' and store the result in 'df_result'. "
                    "Only use pandas (pd) and numpy (np) in the code. Imports are already available."
                ),
                user=(
                    f"Data profile:\n{profile_summary}\n\n"
                    f"Domain context: {domain_context}\n\n"
                    "Generate a cleaning plan with these operations (in order):\n"
                    "1. Remove exact duplicates\n"
                    "2. Handle missing values (median for numeric, mode for categorical)\n"
                    "3. Cap outliers with IQR method (3x multiplier)\n"
                    "Return ONLY the JSON."
                ),
            )

            operations = []
            for op_data in result.get("operations", []):
                operations.append(CleaningOperation(
                    name=op_data.get("name", "unknown"),
                    target_column=op_data.get("target_column", ""),
                    strategy=op_data.get("strategy", ""),
                    llm_justification=op_data.get("llm_justification", ""),
                    code=op_data.get("code", ""),
                ))

            if operations:
                return CleaningPlan(operations=operations)

        except Exception as e:
            self.log(f"⚠️ LLM cleaning plan failed, using defaults: {e}", level="warning")

        # Rule-based fallback plan
        return self._default_cleaning_plan(df, profile)

    def _default_cleaning_plan(self, df: pd.DataFrame,
                                profile: DataProfile) -> CleaningPlan:
        """Default rule-based cleaning plan."""
        operations = []

        # 1. Drop duplicates
        if profile.duplicate_count > 0:
            operations.append(CleaningOperation(
                name="drop_duplicates",
                target_column="__all__",
                strategy="drop_exact_duplicates",
                llm_justification="Remove exact duplicate rows.",
                code="df_result = df.drop_duplicates()",
            ))

        # 2. Fill missing numeric with median
        numeric_missing = [col for col, pct in profile.missing_pct.items()
                           if pct > 0 and profile.dtypes.get(col, "").startswith(("int", "float"))]
        if numeric_missing:
            operations.append(CleaningOperation(
                name="fill_numeric_missing",
                target_column="__numeric__",
                strategy="median_imputation",
                llm_justification="Fill missing numeric values with column median.",
                code=(
                    "df_result = df.copy()\n"
                    "for col in df_result.select_dtypes(include='number').columns:\n"
                    "    df_result[col] = df_result[col].fillna(df_result[col].median())"
                ),
            ))

        # 3. Fill missing categorical with mode
        cat_missing = [col for col, pct in profile.missing_pct.items()
                       if pct > 0 and profile.dtypes.get(col, "") == "object"]
        if cat_missing:
            operations.append(CleaningOperation(
                name="fill_categorical_missing",
                target_column="__categorical__",
                strategy="mode_imputation",
                llm_justification="Fill missing categorical values with mode.",
                code=(
                    "df_result = df.copy()\n"
                    "for col in df_result.select_dtypes(include='object').columns:\n"
                    "    if df_result[col].isnull().any():\n"
                    "        mode_val = df_result[col].mode()\n"
                    "        df_result[col] = df_result[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')"
                ),
            ))

        # 4. Cap outliers
        if profile.outlier_flags:
            operations.append(CleaningOperation(
                name="cap_outliers_iqr",
                target_column="__numeric__",
                strategy="iqr_cap",
                llm_justification="Cap extreme outliers using IQR method.",
                code=(
                    "df_result = df.copy()\n"
                    "for col in df_result.select_dtypes(include='number').columns:\n"
                    "    Q1, Q3 = df_result[col].quantile(0.25), df_result[col].quantile(0.75)\n"
                    "    IQR = Q3 - Q1\n"
                    "    if IQR > 0:\n"
                    "        lower, upper = Q1 - 3*IQR, Q3 + 3*IQR\n"
                    "        df_result[col] = df_result[col].clip(lower, upper)"
                ),
            ))

        return CleaningPlan(operations=operations)

    def _apply_operation(self, df: pd.DataFrame,
                          op: CleaningOperation) -> pd.DataFrame:
        """Apply a single cleaning operation."""
        if op.code:
            try:
                return safe_exec_simple(op.code, df)
            except Exception as e:
                self.log(f"   Sandbox exec failed for {op.name}: {e}, trying direct", level="warning")

        # Direct fallback operations
        if "duplicate" in op.strategy.lower():
            return df.drop_duplicates()

        if "median" in op.strategy.lower():
            df_result = df.copy()
            for col in df_result.select_dtypes(include="number").columns:
                df_result[col] = df_result[col].fillna(df_result[col].median())
            return df_result

        if "mode" in op.strategy.lower():
            df_result = df.copy()
            for col in df_result.select_dtypes(include="object").columns:
                if df_result[col].isnull().any():
                    mode_val = df_result[col].mode()
                    fill = mode_val[0] if len(mode_val) > 0 else "Unknown"
                    df_result[col] = df_result[col].fillna(fill)
            return df_result

        if "iqr" in op.strategy.lower() or "outlier" in op.strategy.lower():
            df_result = df.copy()
            for col in df_result.select_dtypes(include="number").columns:
                Q1, Q3 = df_result[col].quantile(0.25), df_result[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
                    df_result[col] = df_result[col].clip(lower, upper)
            return df_result

        return df
