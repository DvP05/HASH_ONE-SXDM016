"""
Pydantic data models for the Autonomous Analysis System.
All structured data flowing through the pipeline is defined here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"


class SourceType(str, Enum):
    FILE = "file"
    API = "api"
    NASA_API = "nasa_api"
    DATABASE = "database"
    STREAM = "stream"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ─────────────────────────────────────────────
# Configuration Models
# ─────────────────────────────────────────────


class LLMConfig(BaseModel):
    provider: str = "auto"
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 4096
    base_url: str = ""


class DataSource(BaseModel):
    source_type: SourceType = SourceType.FILE
    uri: str = ""
    path: str = ""
    auth: dict = Field(default_factory=dict)
    fetch_params: dict = Field(default_factory=dict)
    query: str = ""

    class Config:
        extra = "allow"


class PipelineConfig(BaseModel):
    problem_statement: str = ""
    domain_context: str = ""
    sources: list[DataSource] = Field(default_factory=list)
    target_column: str = ""
    task_type: TaskType = TaskType.CLASSIFICATION
    human_checkpoints: list[str] = Field(default_factory=list)
    quality_gates_path: str = "config/quality_gates.yaml"
    llm: LLMConfig = Field(default_factory=LLMConfig)
    output_dir: str = "output"


# ─────────────────────────────────────────────
# Data Processing Models
# ─────────────────────────────────────────────


class DataProfile(BaseModel):
    """Profile of a dataset's quality and characteristics."""
    shape: tuple[int, int] = (0, 0)
    dtypes: dict[str, str] = Field(default_factory=dict)
    missing_pct: dict[str, float] = Field(default_factory=dict)
    cardinality: dict[str, int] = Field(default_factory=dict)
    outlier_flags: dict[str, int] = Field(default_factory=dict)
    duplicate_count: int = 0
    sample: dict = Field(default_factory=dict)
    numeric_stats: dict = Field(default_factory=dict)


class CleaningOperation(BaseModel):
    """A single data cleaning operation with LLM justification."""
    name: str
    target_column: str = ""
    strategy: str = ""
    params: dict = Field(default_factory=dict)
    llm_justification: str = ""
    code: str = ""


class CleaningPlan(BaseModel):
    """Ordered list of cleaning operations."""
    operations: list[CleaningOperation] = Field(default_factory=list)


class CleaningLogEntry(BaseModel):
    operation: str
    reasoning: str = ""
    rows_affected: int = 0
    reversible: bool = True


class CleaningResult(BaseModel):
    log: list[CleaningLogEntry] = Field(default_factory=list)
    profile_before: Optional[DataProfile] = None
    profile_after: Optional[DataProfile] = None


# ─────────────────────────────────────────────
# EDA Models
# ─────────────────────────────────────────────


class Finding(BaseModel):
    """A single EDA finding."""
    column: str = ""
    analysis_type: str = ""
    stats: dict = Field(default_factory=dict)
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    distribution_type: str = ""
    commentary: str = ""
    visualization_path: str = ""
    data: Any = None


class Hypothesis(BaseModel):
    id: str
    statement: str
    supporting_evidence: list[str] = Field(default_factory=list)
    suggested_validation: str = ""


class EDAReport(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    narrative: str = ""
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    recommended_features: list[str] = Field(default_factory=list)
    correlation_matrix_path: str = ""
    data_traits: dict = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Feature Engineering Models
# ─────────────────────────────────────────────


class FeatureTransform(BaseModel):
    name: str
    transform_type: str = ""
    input_columns: list[str] = Field(default_factory=list)
    operation: str = ""
    output_columns: list[str] = Field(default_factory=list)
    rationale: str = ""
    code: str = ""


class FeatureEngineeringPlan(BaseModel):
    transforms: list[FeatureTransform] = Field(default_factory=list)


class FeatureStore(BaseModel):
    feature_names: list[str] = Field(default_factory=list)
    lineage: dict[str, str] = Field(default_factory=dict)
    importance_scores: dict[str, float] = Field(default_factory=dict)
    n_features: int = 0
    n_samples: int = 0


# ─────────────────────────────────────────────
# Modeling Models
# ─────────────────────────────────────────────


class ModelEvaluation(BaseModel):
    task_type: str = ""
    # Classification metrics
    accuracy: Optional[float] = None
    roc_auc: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    confusion_matrix: Optional[list] = None
    # Regression metrics
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    # Robustness
    cv_mean: float = 0.0
    cv_std: float = 0.0
    train_test_gap: float = 0.0
    # SHAP
    top_features: list[dict] = Field(default_factory=list)
    global_explanation: str = ""


class ModelCard(BaseModel):
    model_name: str = ""
    model_type: str = ""
    task_type: str = ""
    evaluation: Optional[ModelEvaluation] = None
    training_data_summary: str = ""
    intended_use: str = ""
    limitations: str = ""
    best_params: dict = Field(default_factory=dict)


class ModelingResult(BaseModel):
    model_name: str = ""
    evaluation: Optional[ModelEvaluation] = None
    cv_results: dict = Field(default_factory=dict)
    model_card: Optional[ModelCard] = None
    shap_summary_path: str = ""
    feature_importance_path: str = ""


# ─────────────────────────────────────────────
# Insight Models
# ─────────────────────────────────────────────


class Recommendation(BaseModel):
    title: str
    description: str = ""
    impact: str = ""
    confidence: float = 0.0
    evidence: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    priority: Priority = Priority.MEDIUM


class InsightReport(BaseModel):
    executive_summary: str = ""
    data_quality_section: str = ""
    pattern_section: str = ""
    model_section: str = ""
    recommendations: list[Recommendation] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    generated_at: str = ""


# ─────────────────────────────────────────────
# Agent/Pipeline Trace Models
# ─────────────────────────────────────────────


class ToolCall(BaseModel):
    tool_name: str
    args: dict = Field(default_factory=dict)
    result_summary: str = ""
    duration_seconds: float = 0.0


class AgentTrace(BaseModel):
    agent_name: str = ""
    stage: str = ""
    start_time: str = ""
    end_time: str = ""
    duration_seconds: float = 0.0
    llm_calls: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tokens_used: int = 0
    cost_usd: float = 0.0
    reasoning_steps: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StageResult(BaseModel):
    stage_name: str
    status: str = "success"
    trace: Optional[AgentTrace] = None
    output_summary: str = ""
    quality_gate_passed: bool = True


class PipelineResult(BaseModel):
    stages: dict[str, StageResult] = Field(default_factory=dict)
    total_duration_seconds: float = 0.0
    total_llm_calls: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    insight_report: Optional[InsightReport] = None
    completed_at: str = ""
