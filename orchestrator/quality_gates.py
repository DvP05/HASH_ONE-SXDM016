"""
Quality gates — validates stage outputs against configured thresholds.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_GATES = {
    "data_collection": {
        "min_rows": 10,
        "min_columns": 2,
    },
    "data_cleaning": {
        "max_missing_pct_after": 0.05,
        "min_rows_retained_pct": 0.80,
    },
    "eda": {
        "min_findings": 1,
    },
    "feature_engineering": {
        "min_features": 3,
        "max_features": 200,
    },
    "modeling": {
        "classification": {
            "min_roc_auc": 0.55,
            "max_train_test_gap": 0.20,
        },
        "regression": {
            "min_r2": 0.10,
        },
    },
}


class QualityGateResult:
    def __init__(self, passed: bool, issues: list[str] | None = None):
        self.passed = passed
        self.issues = issues or []


class QualityGates:
    """Validates stage outputs against quality thresholds."""

    def __init__(self, config_path: str = ""):
        self.gates = DEFAULT_GATES.copy()

        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    custom = yaml.safe_load(f) or {}
                self.gates.update(custom)
                logger.info(f"Loaded quality gates from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load quality gates: {e}")

    def validate(self, stage_name: str, result: Any,
                 memory: Any = None) -> QualityGateResult:
        """Validate a stage's output against its quality gate."""
        gate = self.gates.get(stage_name)
        if not gate:
            return QualityGateResult(passed=True)

        issues = []

        if stage_name == "data_collection":
            issues = self._validate_collection(result, gate)
        elif stage_name == "data_cleaning":
            issues = self._validate_cleaning(result, gate, memory)
        elif stage_name == "eda":
            issues = self._validate_eda(result, gate)
        elif stage_name == "feature_engineering":
            issues = self._validate_features(result, gate)
        elif stage_name == "modeling":
            issues = self._validate_modeling(result, gate, memory)

        passed = len(issues) == 0
        if not passed:
            logger.warning(f"Quality gate FAILED for {stage_name}: {issues}")
        else:
            logger.info(f"Quality gate PASSED for {stage_name}")

        return QualityGateResult(passed=passed, issues=issues)

    def _validate_collection(self, result: dict, gate: dict) -> list[str]:
        issues = []
        if result.get("rows", 0) < gate.get("min_rows", 10):
            issues.append(f"Too few rows: {result.get('rows')} < {gate['min_rows']}")
        if result.get("columns", 0) < gate.get("min_columns", 2):
            issues.append(f"Too few columns: {result.get('columns')} < {gate['min_columns']}")
        return issues

    def _validate_cleaning(self, result: Any, gate: dict,
                            memory: Any = None) -> list[str]:
        issues = []
        if hasattr(result, "profile_after") and result.profile_after:
            max_missing = max(result.profile_after.missing_pct.values()) if result.profile_after.missing_pct else 0
            threshold = gate.get("max_missing_pct_after", 0.05)
            if max_missing > threshold:
                issues.append(f"Max missing {max_missing:.1%} > {threshold:.1%}")
        return issues

    def _validate_eda(self, result: Any, gate: dict) -> list[str]:
        issues = []
        if hasattr(result, "findings"):
            if len(result.findings) < gate.get("min_findings", 1):
                issues.append(f"Too few findings: {len(result.findings)}")
        return issues

    def _validate_features(self, result: Any, gate: dict) -> list[str]:
        issues = []
        if hasattr(result, "n_features"):
            if result.n_features < gate.get("min_features", 3):
                issues.append(f"Too few features: {result.n_features}")
            if result.n_features > gate.get("max_features", 200):
                issues.append(f"Too many features: {result.n_features}")
        return issues

    def _validate_modeling(self, result: Any, gate: dict,
                            memory: Any = None) -> list[str]:
        issues = []
        if hasattr(result, "evaluation") and result.evaluation:
            ev = result.evaluation
            task_type = ev.task_type
            task_gate = gate.get(task_type, {})

            if task_type == "classification":
                min_auc = task_gate.get("min_roc_auc", 0.55)
                if ev.roc_auc is not None and ev.roc_auc < min_auc:
                    issues.append(f"ROC-AUC {ev.roc_auc:.4f} < {min_auc}")
            elif task_type == "regression":
                min_r2 = task_gate.get("min_r2", 0.10)
                if ev.r2 is not None and ev.r2 < min_r2:
                    issues.append(f"R² {ev.r2:.4f} < {min_r2}")

        return issues
