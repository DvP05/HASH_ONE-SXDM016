# End-to-End Agentic AI Pipeline
## From Raw Data Collection to Insight Generation

> A comprehensive blueprint for building intelligent, autonomous data science pipelines using AI agents — covering data collection, cleaning, EDA, feature engineering, predictive modeling, and automated insight generation.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Agent Design Philosophy](#agent-design-philosophy)
3. [Stage 1 — Data Collection Agent](#stage-1--data-collection-agent)
4. [Stage 2 — Data Cleaning Agent](#stage-2--data-cleaning-agent)
5. [Stage 3 — EDA Agent](#stage-3--eda-agent)
6. [Stage 4 — Feature Engineering Agent](#stage-4--feature-engineering-agent)
7. [Stage 5 — Predictive Modeling Agent](#stage-5--predictive-modeling-agent)
8. [Stage 6 — Insight Generation Agent](#stage-6--insight-generation-agent)
9. [Orchestrator Layer](#orchestrator-layer)
10. [Memory & State Management](#memory--state-management)
11. [Tool Registry](#tool-registry)
12. [Evaluation & Observability](#evaluation--observability)
13. [Full Stack Implementation](#full-stack-implementation)
14. [Best Practices & Anti-Patterns](#best-practices--anti-patterns)

---

## Architecture Overview

The pipeline is composed of **six specialized agents** coordinated by a central **Orchestrator**. Each agent is a self-contained reasoning unit with access to tools, memory, and a well-defined input/output contract.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR AGENT                           │
│         (Plans, routes, retries, and validates stage outputs)       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────────────▼─────────────────────────┐
        │              SHARED MEMORY STORE               │
        │   (Schema registry · Data lineage · State)     │
        └──────────────────────┬─────────────────────────┘
                               │
   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐   ┌───────┐
   │ Data  │──▶│Clean  │──▶│  EDA  │──▶│Feature│──▶│Model  │──▶│Insight│
   │Collect│   │ Agent │   │ Agent │   │ Eng.  │   │ Agent │   │ Agent │
   └───────┘   └───────┘   └───────┘   └───────┘   └───────┘   └───────┘
```

### Core Principles

- **Autonomy with guardrails** — agents make decisions within bounded action spaces
- **Human-in-the-loop checkpoints** — configurable approval gates between stages
- **Reproducibility** — every agent action is logged with its reasoning trace
- **Composability** — each stage is independently testable and replaceable
- **Self-healing** — agents can detect failures and reroute or retry with modified strategies

---

## Agent Design Philosophy

Each agent follows the **ReAct (Reasoning + Acting)** pattern extended with memory:

```
Observe → Think → Plan → Act → Reflect → Store
```

### Agent Anatomy

```python
class BaseAgent:
    """
    Every agent in the pipeline inherits from this base class.
    """
    def __init__(self, name: str, tools: list[Tool], memory: MemoryStore):
        self.name = name
        self.tools = tools          # Callable tools this agent can invoke
        self.memory = memory        # Shared or scoped memory
        self.llm = LLMClient()      # Underlying language model
        self.trace = []             # Audit trail of all reasoning steps

    def run(self, task: AgentTask) -> AgentResult:
        context = self.memory.retrieve(task.context_keys)
        plan = self.llm.plan(task=task, context=context, tools=self.tools)

        for step in plan.steps:
            obs = self.execute_tool(step.tool, step.args)
            self.trace.append({"step": step, "observation": obs})
            if self.should_replan(obs):
                plan = self.llm.replan(plan, obs)

        result = self.llm.synthesize(self.trace)
        self.memory.store(task.output_keys, result)
        return result
```

### LLM Backbone Choices

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Orchestration & planning | GPT-4o / Claude Opus | Strongest reasoning, handles ambiguity |
| Code generation | Claude Sonnet / GPT-4o | Fast, reliable code output |
| Schema inference | Claude Haiku / GPT-4o-mini | Low-latency, cost-efficient |
| Insight narration | Claude Sonnet | Strong prose generation |

---

## Stage 1 — Data Collection Agent

### Responsibility

Autonomously discover, ingest, validate schema, and normalize raw data from heterogeneous sources.

### Supported Sources

- REST APIs (paginated, OAuth, API key)
- Databases (PostgreSQL, MySQL, MongoDB, BigQuery)
- File systems (CSV, Parquet, JSON, Excel, XML)
- Web scraping (static HTML, dynamic JS via Playwright)
- Streaming (Kafka, Kinesis, WebSockets)
- Cloud storage (S3, GCS, Azure Blob)

### Agent Workflow

```
1. DISCOVER    → Identify available data sources from config or user prompt
2. AUTHENTICATE → Resolve credentials from vault/env
3. INGEST      → Pull data using appropriate connector
4. VALIDATE    → Check schema, row counts, freshness
5. NORMALIZE   → Convert to canonical Parquet format
6. REGISTER    → Store schema + metadata in memory store
```

### Implementation

```python
from dataclasses import dataclass
from typing import Any
import pandas as pd

@dataclass
class DataSource:
    source_type: str          # "api" | "db" | "file" | "stream"
    uri: str
    auth: dict
    fetch_params: dict        # pagination config, filters, etc.

class DataCollectionAgent(BaseAgent):
    """
    Ingests raw data from any configured source and emits
    a normalized DataFrame with schema metadata.
    """

    TOOL_REGISTRY = ["http_fetch", "db_query", "file_reader",
                     "s3_download", "kafka_consume", "playwright_scrape"]

    def run(self, sources: list[DataSource]) -> CollectionResult:
        frames = []

        for source in sources:
            self.log(f"Ingesting from {source.source_type}: {source.uri}")

            # LLM selects the right tool based on source metadata
            tool_name = self.llm.select_tool(source, self.TOOL_REGISTRY)
            raw = self.execute_tool(tool_name, source)

            schema = self.infer_schema(raw)
            validated = self.validate(raw, schema)
            normalized = self.normalize(validated)

            frames.append(normalized)
            self.memory.store(f"schema:{source.uri}", schema)

        merged = self.llm.decide_merge_strategy(frames)
        self.memory.store("raw_dataset", merged)
        return CollectionResult(data=merged, sources=sources)

    def validate(self, df: pd.DataFrame, schema: dict) -> pd.DataFrame:
        """
        LLM-assisted validation: generates and executes
        validation rules based on inferred schema.
        """
        rules = self.llm.generate_validation_rules(schema)
        for rule in rules:
            result = rule.evaluate(df)
            if result.failed:
                self.log(f"⚠️  Validation failed: {rule.description}")
                df = self.llm.suggest_fix(df, rule, result)
        return df
```

### Configuration Example

```yaml
# pipeline_config.yaml
data_collection:
  sources:
    - type: api
      uri: "https://api.example.com/v2/sales"
      auth:
        type: bearer
        token_env: SALES_API_TOKEN
      pagination:
        strategy: cursor
        param: next_cursor
        page_size: 1000

    - type: database
      uri: "postgresql://host/analytics"
      query: "SELECT * FROM events WHERE created_at > '2024-01-01'"
      auth:
        type: env
        var: DB_CONNECTION_STRING

    - type: file
      path: "s3://data-lake/raw/customers/*.parquet"
```

---

## Stage 2 — Data Cleaning Agent

### Responsibility

Detect and remediate data quality issues autonomously. Every action is justified, reversible, and auditable.

### Issue Detection Matrix

| Issue Type | Detection Method | Remediation Strategy |
|------------|-----------------|----------------------|
| Missing values | Statistical analysis | Mean/median/mode imputation, KNN, MICE, flag & drop |
| Duplicates | Hash + fuzzy match | Dedup with confidence score |
| Outliers | IQR, Z-score, Isolation Forest | Cap, transform, or flag |
| Type mismatches | Schema inference | Cast or parse with regex |
| Inconsistent categories | Fuzzy string match | Canonical mapping |
| Date/time issues | Format detection | Normalize to ISO 8601 |
| PII detection | Named entity recognition | Hash, mask, or tokenize |

### Agent Workflow

```
1. PROFILE     → Generate full data quality report
2. PRIORITIZE  → LLM ranks issues by severity and downstream impact
3. PLAN        → Generate cleaning operations in dependency order
4. EXECUTE     → Apply transforms with rollback support
5. VERIFY      → Rerun quality checks post-cleaning
6. DOCUMENT    → Emit cleaning report with before/after stats
```

### Implementation

```python
class DataCleaningAgent(BaseAgent):
    """
    Autonomously cleans data using a combination of statistical
    methods and LLM-guided decision making.
    """

    def run(self) -> CleaningResult:
        df = self.memory.retrieve("raw_dataset")
        profile = self.profile_data(df)

        # LLM reasons about quality issues and produces a cleaning plan
        cleaning_plan = self.llm.generate_cleaning_plan(
            profile=profile,
            domain_context=self.memory.retrieve("domain_context"),
            constraints=["preserve_row_count_if_possible",
                         "flag_rather_than_drop_ambiguous"]
        )

        df_clean = df.copy()
        change_log = []

        for operation in cleaning_plan.operations:
            df_before = df_clean.copy()
            df_clean = self.apply_operation(df_clean, operation)
            change_log.append({
                "operation": operation.name,
                "reasoning": operation.llm_justification,
                "rows_affected": self.diff_count(df_before, df_clean),
                "reversible": True
            })

        self.memory.store("clean_dataset", df_clean)
        self.memory.store("cleaning_log", change_log)
        return CleaningResult(data=df_clean, log=change_log)

    def profile_data(self, df: pd.DataFrame) -> DataProfile:
        return DataProfile(
            shape=df.shape,
            dtypes=df.dtypes.to_dict(),
            missing_pct=df.isnull().mean().to_dict(),
            cardinality={col: df[col].nunique() for col in df.columns},
            outlier_flags=self.detect_outliers(df),
            duplicate_count=df.duplicated().sum(),
            sample=df.sample(min(5, len(df))).to_dict()
        )

    def apply_operation(self, df, op: CleaningOperation) -> pd.DataFrame:
        """
        Executes a cleaning operation. Operations are code-generated
        by the LLM, sandboxed, and applied safely.
        """
        code = self.llm.generate_pandas_code(op)
        return self.safe_exec(code, df)   # Executes in restricted env
```

### Cleaning Operations DSL

```python
# The LLM produces cleaning plans in this structured format
cleaning_plan = CleaningPlan(operations=[
    CleaningOperation(
        name="impute_age_missing",
        target_column="age",
        strategy="median_by_group",
        group_by=["gender", "region"],
        llm_justification="Age likely varies by demographic group; "
                          "group-wise median reduces bias vs. global median."
    ),
    CleaningOperation(
        name="standardize_country_codes",
        target_column="country",
        strategy="fuzzy_canonical_mapping",
        canonical_source="iso_3166",
        threshold=0.85,
        llm_justification="Detected 23 country name variants mapping "
                          "to 3 distinct ISO codes."
    ),
    CleaningOperation(
        name="cap_revenue_outliers",
        target_column="revenue",
        strategy="iqr_cap",
        multiplier=3.0,
        llm_justification="Revenue outliers at 99.8th percentile appear "
                          "to be data entry errors, not genuine extremes."
    )
])
```

---

## Stage 3 — EDA Agent

### Responsibility

Conduct comprehensive Exploratory Data Analysis, surface meaningful patterns, generate visualizations, and produce a structured discovery report.

### Analysis Modules

```
┌─────────────────────────────────────────────┐
│              EDA AGENT MODULES              │
├─────────────────┬───────────────────────────┤
│ Univariate      │ Distribution, skewness,   │
│ Analysis        │ kurtosis, percentiles      │
├─────────────────┼───────────────────────────┤
│ Bivariate       │ Correlation, scatter,      │
│ Analysis        │ cross-tabs, chi-square     │
├─────────────────┼───────────────────────────┤
│ Multivariate    │ PCA, cluster structure,    │
│ Analysis        │ mutual information         │
├─────────────────┼───────────────────────────┤
│ Temporal        │ Trend, seasonality,        │
│ Analysis        │ autocorrelation            │
├─────────────────┼───────────────────────────┤
│ Target Analysis │ Class balance, leakage     │
│                 │ detection, baseline stats  │
└─────────────────┴───────────────────────────┘
```

### Implementation

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class EDAAgent(BaseAgent):
    """
    Performs structured EDA and emits a rich analysis report
    with auto-generated visualizations and LLM commentary.
    """

    def run(self) -> EDAReport:
        df = self.memory.retrieve("clean_dataset")
        target = self.memory.retrieve("target_column")

        findings = []

        # --- Univariate ---
        for col in df.columns:
            finding = self.analyze_univariate(df[col])
            findings.append(finding)

        # --- Bivariate (vs target) ---
        if target:
            for col in df.columns:
                if col != target:
                    finding = self.analyze_bivariate(df[col], df[target])
                    findings.append(finding)

        # --- Correlation Structure ---
        corr_findings = self.analyze_correlations(df)
        findings.extend(corr_findings)

        # --- LLM synthesizes findings into narrative ---
        narrative = self.llm.synthesize_eda(
            findings=findings,
            domain_context=self.memory.retrieve("domain_context")
        )

        # --- Hypothesis generation ---
        hypotheses = self.llm.generate_hypotheses(findings)

        report = EDAReport(
            findings=findings,
            narrative=narrative,
            hypotheses=hypotheses,
            recommended_features=self.llm.recommend_features(findings)
        )

        self.memory.store("eda_report", report)
        return report

    def analyze_univariate(self, series: pd.Series) -> Finding:
        desc = series.describe()
        skewness = series.skew() if pd.api.types.is_numeric_dtype(series) else None
        kurt = series.kurtosis() if pd.api.types.is_numeric_dtype(series) else None

        finding = Finding(
            column=series.name,
            analysis_type="univariate",
            stats=desc.to_dict(),
            skewness=skewness,
            kurtosis=kurt,
            distribution_type=self.infer_distribution(series),
            visualization=self.generate_distribution_plot(series)
        )

        # LLM interprets the statistics
        finding.commentary = self.llm.interpret_univariate(finding)
        return finding

    def analyze_correlations(self, df: pd.DataFrame) -> list[Finding]:
        numeric_df = df.select_dtypes(include="number")
        corr_matrix = numeric_df.corr()

        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.7:
                    high_corr.append({
                        "col_a": corr_matrix.columns[i],
                        "col_b": corr_matrix.columns[j],
                        "correlation": val
                    })

        commentary = self.llm.interpret_correlations(high_corr)
        return [Finding(analysis_type="correlation",
                        data=high_corr,
                        commentary=commentary)]
```

### LLM-Generated EDA Hypotheses (Example Output)

```json
{
  "hypotheses": [
    {
      "id": "H1",
      "statement": "Customer tenure is positively correlated with average order value.",
      "supporting_evidence": ["pearson_r=0.62 between tenure_days and avg_order_value"],
      "suggested_validation": "Partial correlation controlling for product_category"
    },
    {
      "id": "H2",
      "statement": "Churn rate follows a seasonal pattern with peaks in Q1.",
      "supporting_evidence": ["autocorrelation spike at lag=12 in monthly churn series"],
      "suggested_validation": "STL decomposition + Mann-Kendall trend test"
    }
  ]
}
```

---

## Stage 4 — Feature Engineering Agent

### Responsibility

Automatically generate, select, and validate features for the predictive model. Guided by EDA findings and domain knowledge.

### Feature Generation Strategies

```
Raw Features
    │
    ├── Numeric Transforms
    │       ├── Log, sqrt, Box-Cox, Yeo-Johnson
    │       ├── Polynomial interactions (degree 2)
    │       ├── Binning & discretization
    │       └── Rank transforms
    │
    ├── Categorical Encoding
    │       ├── One-hot, ordinal, binary
    │       ├── Target encoding (with CV leak prevention)
    │       └── Embedding (for high-cardinality)
    │
    ├── Temporal Features
    │       ├── Hour, day, weekday, month, quarter
    │       ├── Lag features (t-1, t-7, t-30)
    │       ├── Rolling statistics (mean, std, min, max)
    │       └── Time-since features
    │
    ├── Aggregation Features
    │       ├── Group statistics by key columns
    │       └── Window aggregations
    │
    └── Domain-Specific Features
            └── LLM-generated from domain context
```

### Implementation

```python
from sklearn.preprocessing import PowerTransformer, TargetEncoder
from sklearn.feature_selection import mutual_info_classif

class FeatureEngineeringAgent(BaseAgent):
    """
    Generates and selects features using a combination of
    automated transforms and LLM-guided domain reasoning.
    """

    def run(self) -> FeatureSet:
        df = self.memory.retrieve("clean_dataset")
        eda_report = self.memory.retrieve("eda_report")
        target = self.memory.retrieve("target_column")
        task_type = self.memory.retrieve("task_type")  # classification / regression

        # LLM proposes feature engineering operations based on EDA
        fe_plan = self.llm.create_feature_plan(
            schema=self.memory.retrieve("schema"),
            eda_findings=eda_report.findings,
            hypotheses=eda_report.hypotheses,
            task_type=task_type
        )

        df_features = df.copy()

        # Apply each proposed transform
        for transform in fe_plan.transforms:
            try:
                df_features = self.apply_transform(df_features, transform)
                self.log(f"✅ Applied: {transform.name} → {transform.output_columns}")
            except Exception as e:
                self.log(f"⚠️  Failed: {transform.name} — {e}")

        # Feature selection
        selected = self.select_features(df_features, target, task_type)

        # Store feature store with lineage
        feature_store = FeatureStore(
            features=df_features[selected],
            lineage={col: self.get_lineage(col, fe_plan) for col in selected},
            importance_scores=self.compute_importance(df_features[selected], df[target])
        )

        self.memory.store("feature_store", feature_store)
        return feature_store

    def select_features(self, df, target, task_type) -> list[str]:
        """
        Multi-method feature selection with LLM as tie-breaker.
        """
        X = df.drop(columns=[target])
        y = df[target]

        # Method 1: Mutual Information
        mi_scores = mutual_info_classif(X.fillna(0), y)
        mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        # Method 2: Correlation-based filtering (remove redundant)
        corr_matrix = X.corr().abs()
        redundant = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > 0.95:
                    redundant.add(corr_matrix.columns[j])

        candidates = [col for col in mi_ranking.index[:50] if col not in redundant]

        # LLM makes final selection with business reasoning
        final_selection = self.llm.select_features(
            candidates=candidates,
            mi_scores=mi_ranking[candidates].to_dict(),
            domain_context=self.memory.retrieve("domain_context"),
            max_features=30
        )

        return final_selection
```

### Feature Engineering Plan (LLM Output Format)

```json
{
  "transforms": [
    {
      "name": "log_transform_revenue",
      "type": "numeric_transform",
      "input_columns": ["revenue"],
      "operation": "log1p",
      "output_columns": ["log_revenue"],
      "rationale": "Revenue shows right-skew (skewness=4.2); log transform will normalize distribution and stabilize variance for linear models."
    },
    {
      "name": "tenure_x_segment",
      "type": "interaction",
      "input_columns": ["tenure_days", "segment_encoded"],
      "operation": "multiply",
      "output_columns": ["tenure_segment_interaction"],
      "rationale": "EDA hypothesis H1 suggests tenure effect may differ by segment; interaction term captures this conditional relationship."
    },
    {
      "name": "rolling_7d_purchase_rate",
      "type": "temporal_aggregate",
      "input_columns": ["purchase_flag"],
      "time_column": "event_date",
      "group_by": ["customer_id"],
      "window": "7D",
      "operation": "mean",
      "output_columns": ["purchase_rate_7d"],
      "rationale": "Short-term purchase velocity is a strong behavioral signal for churn prediction."
    }
  ]
}
```

---

## Stage 5 — Predictive Modeling Agent

### Responsibility

Select, train, tune, and evaluate the best predictive model(s) for the task. Applies AutoML principles with LLM-guided meta-learning.

### Model Selection Framework

```
Task Detection
    │
    ├── Classification → LogReg, Random Forest, XGBoost,
    │                    LightGBM, CatBoost, Neural Net
    │
    ├── Regression    → Ridge, Lasso, ElasticNet, XGBoost,
    │                   LightGBM, SVR, Neural Net
    │
    └── Time Series   → ARIMA, Prophet, LSTM, Temporal Fusion
                        Transformer, N-BEATS
```

### Implementation

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
import optuna

class PredictiveModelingAgent(BaseAgent):
    """
    Trains and selects the best model via multi-algorithm
    competition with Bayesian hyperparameter optimization.
    """

    CANDIDATE_MODELS = {
        "logistic_regression": LogisticRegressionFactory,
        "random_forest": RandomForestFactory,
        "xgboost": XGBoostFactory,
        "lightgbm": LightGBMFactory,
        "neural_net": NeuralNetFactory,
    }

    def run(self) -> ModelingResult:
        feature_store = self.memory.retrieve("feature_store")
        target = self.memory.retrieve("target_column")
        task_type = self.memory.retrieve("task_type")

        X = feature_store.features
        y = self.memory.retrieve("clean_dataset")[target]

        # LLM recommends model candidates based on data characteristics
        candidates = self.llm.recommend_models(
            n_rows=len(X),
            n_features=len(X.columns),
            task_type=task_type,
            class_balance=y.value_counts(normalize=True).to_dict(),
            data_traits=self.memory.retrieve("eda_report").data_traits
        )

        results = {}

        # Train all candidates with cross-validation
        for model_name in candidates:
            self.log(f"Training: {model_name}")
            pipeline = self.build_pipeline(model_name, feature_store)
            cv_scores = self.cross_validate(pipeline, X, y, task_type)
            results[model_name] = cv_scores

        # Select best model
        best_model_name = self.llm.select_best_model(results)
        self.log(f"Selected: {best_model_name}")

        # Hyperparameter optimization with Optuna
        best_pipeline = self.optimize_hyperparameters(
            model_name=best_model_name,
            X=X, y=y,
            n_trials=100,
            task_type=task_type
        )

        # Final evaluation
        evaluation = self.evaluate(best_pipeline, X, y, task_type)

        # Explainability (SHAP)
        shap_values = self.compute_shap(best_pipeline, X)

        result = ModelingResult(
            model=best_pipeline,
            evaluation=evaluation,
            cv_results=results,
            shap_values=shap_values,
            model_card=self.generate_model_card(best_pipeline, evaluation)
        )

        self.memory.store("model_result", result)
        return result

    def optimize_hyperparameters(self, model_name, X, y, n_trials, task_type):
        """
        Bayesian optimization via Optuna. Search space is
        generated by LLM based on model type and data size.
        """
        search_space = self.llm.generate_search_space(model_name, X.shape)

        def objective(trial):
            params = {k: trial.suggest(*v) for k, v in search_space.items()}
            pipeline = self.build_pipeline(model_name, params=params)
            scores = cross_val_score(pipeline, X, y, cv=5,
                                     scoring=self.get_metric(task_type))
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        best_params = study.best_params
        return self.build_pipeline(model_name, params=best_params)

    def generate_model_card(self, pipeline, evaluation) -> ModelCard:
        """LLM generates a structured model card (ala Model Cards for Model Reporting)."""
        return self.llm.write_model_card(
            pipeline=pipeline,
            evaluation=evaluation,
            training_data_summary=self.memory.retrieve("eda_report").summary,
            intended_use=self.memory.retrieve("problem_statement"),
            limitations=self.llm.identify_limitations(pipeline, evaluation)
        )
```

### Evaluation Report Schema

```python
@dataclass
class ModelEvaluation:
    task_type: str
    # Classification metrics
    accuracy: float | None = None
    roc_auc: float | None = None
    f1_score: float | None = None
    precision: float | None = None
    recall: float | None = None
    confusion_matrix: list | None = None

    # Regression metrics
    rmse: float | None = None
    mae: float | None = None
    r2: float | None = None
    mape: float | None = None

    # Model robustness
    cv_std: float = 0.0
    train_test_gap: float = 0.0         # Overfitting indicator
    calibration_score: float | None = None

    # SHAP-based interpretation
    top_features: list[dict] = field(default_factory=list)
    global_explanation: str = ""         # LLM-written prose
```

---

## Stage 6 — Insight Generation Agent

### Responsibility

Synthesize all pipeline outputs into human-readable, actionable insights. Generates reports, dashboards, and recommendations.

### Insight Types

```
┌────────────────────────────────────────────────────────┐
│                  INSIGHT CATEGORIES                    │
├─────────────────────────┬──────────────────────────────┤
│ Descriptive             │ What happened?                │
│ Diagnostic              │ Why did it happen?            │
│ Predictive              │ What will happen?             │
│ Prescriptive            │ What should we do?            │
└─────────────────────────┴──────────────────────────────┘
```

### Implementation

```python
class InsightGenerationAgent(BaseAgent):
    """
    Synthesizes pipeline outputs into a rich insight report
    with narrative, visualizations, and recommendations.
    """

    def run(self) -> InsightReport:
        # Gather all prior stage outputs
        context = {
            "problem_statement": self.memory.retrieve("problem_statement"),
            "domain_context": self.memory.retrieve("domain_context"),
            "eda_report": self.memory.retrieve("eda_report"),
            "cleaning_log": self.memory.retrieve("cleaning_log"),
            "feature_store": self.memory.retrieve("feature_store"),
            "model_result": self.memory.retrieve("model_result"),
        }

        # Generate structured insight sections
        report = InsightReport(
            executive_summary=self.llm.write_executive_summary(context),
            data_quality_section=self.generate_data_quality_section(context),
            pattern_section=self.generate_pattern_section(context),
            model_section=self.generate_model_section(context),
            recommendations=self.generate_recommendations(context),
            visualizations=self.generate_dashboard(context),
            next_steps=self.llm.suggest_next_steps(context)
        )

        return report

    def generate_recommendations(self, context) -> list[Recommendation]:
        """
        LLM generates actionable recommendations with
        confidence scores and supporting evidence.
        """
        raw_recs = self.llm.generate_recommendations(
            eda_findings=context["eda_report"].findings,
            model_insights=context["model_result"].shap_values,
            business_context=context["problem_statement"],
            format="structured_json"
        )

        return [
            Recommendation(
                title=r["title"],
                description=r["description"],
                impact=r["estimated_impact"],
                confidence=r["confidence"],
                evidence=r["supporting_evidence"],
                action_items=r["action_items"],
                priority=r["priority"]  # high / medium / low
            )
            for r in raw_recs
        ]
```

### Sample Insight Output

```markdown
## Executive Summary

Analysis of 847,231 customer transactions (Jan 2023 – Dec 2024)
reveals three primary drivers of 30-day churn: declining purchase
frequency (SHAP=0.31), increased support ticket volume (SHAP=0.27),
and seasonal disengagement in Q1 (SHAP=0.19).

The XGBoost classifier achieves AUC=0.891 on holdout data, enabling
reliable 30-day churn prediction with 78% recall at a 0.4 threshold.

## Top Recommendations

### [HIGH PRIORITY] Re-engagement Campaign at Day 14
Customers who have not purchased within 14 days show 3.2× higher
churn probability. A targeted email sequence at day 14 is estimated
to recover 12-18% of at-risk customers.

### [MEDIUM PRIORITY] Support Ticket SLA Reduction
Support ticket resolution time >48h correlates with a 41% increase
in churn probability. Reducing P2 ticket SLA from 72h to 36h is
projected to reduce churn by ~2.1 percentage points.
```

---

## Orchestrator Layer

The Orchestrator manages the entire pipeline lifecycle: planning, routing, retry logic, and stage validation.

```python
class PipelineOrchestrator:
    """
    Top-level coordinator that runs all agents in sequence
    with error handling, retry logic, and human checkpoints.
    """

    STAGES = [
        ("data_collection", DataCollectionAgent),
        ("data_cleaning", DataCleaningAgent),
        ("eda", EDAAgent),
        ("feature_engineering", FeatureEngineeringAgent),
        ("modeling", PredictiveModelingAgent),
        ("insight_generation", InsightGenerationAgent),
    ]

    def __init__(self, config: PipelineConfig, memory: MemoryStore):
        self.config = config
        self.memory = memory
        self.llm = LLMClient(model="claude-opus-4")

        # Store problem context before any stage runs
        self.memory.store("problem_statement", config.problem_statement)
        self.memory.store("domain_context", config.domain_context)
        self.memory.store("target_column", config.target_column)
        self.memory.store("task_type", config.task_type)

    def run(self) -> PipelineResult:
        results = {}

        for stage_name, AgentClass in self.STAGES:
            self.log(f"\n{'='*60}")
            self.log(f"STAGE: {stage_name.upper()}")
            self.log(f"{'='*60}")

            # Human checkpoint (if configured)
            if stage_name in self.config.human_checkpoints:
                approval = self.request_human_approval(stage_name, results)
                if not approval.approved:
                    self.log(f"Pipeline paused at {stage_name}: {approval.reason}")
                    break

            agent = AgentClass(
                name=stage_name,
                tools=self.get_tools(stage_name),
                memory=self.memory
            )

            result = self.run_with_retry(agent, stage_name)
            results[stage_name] = result

            # Validate stage output before proceeding
            validation = self.validate_stage_output(stage_name, result)
            if not validation.passed:
                self.log(f"⚠️  Stage validation failed: {validation.issues}")
                recovery = self.llm.suggest_recovery(stage_name, validation.issues)
                if recovery.requires_rerun:
                    result = self.run_with_retry(agent, stage_name, recovery.hints)
                    results[stage_name] = result

        return PipelineResult(stages=results, memory_snapshot=self.memory.snapshot())

    def run_with_retry(self, agent, stage_name, hints=None, max_retries=3):
        for attempt in range(max_retries):
            try:
                return agent.run(hints=hints)
            except AgentException as e:
                self.log(f"Attempt {attempt+1} failed: {e}")
                if attempt < max_retries - 1:
                    hints = self.llm.suggest_retry_hints(stage_name, e)
                else:
                    raise PipelineException(f"Stage {stage_name} failed after {max_retries} attempts")
```

---

## Memory & State Management

```python
class MemoryStore:
    """
    Hierarchical key-value store with versioning and lineage tracking.
    Supports both short-term (pipeline-scoped) and long-term (persistent) memory.
    """

    def __init__(self, backend: str = "redis"):
        self.short_term = {}        # In-process dict for speed
        self.long_term = RedisBackend(backend) if backend else None
        self.lineage_graph = nx.DiGraph()

    def store(self, key: str, value: Any, persist: bool = False):
        self.short_term[key] = DataEntry(
            value=value,
            timestamp=datetime.utcnow(),
            version=self.get_version(key) + 1
        )
        if persist and self.long_term:
            self.long_term.set(key, self.serialize(value))

    def retrieve(self, key: str, version: int = -1) -> Any:
        entry = self.short_term.get(key)
        if entry is None:
            raise MemoryKeyError(f"Key '{key}' not found in memory store.")
        return entry.value

    def snapshot(self) -> dict:
        """Export full pipeline state for reproducibility."""
        return {k: v.value for k, v in self.short_term.items()}
```

### Key Memory Schema

```
memory/
├── problem_statement       # User-defined task description
├── domain_context          # Business / domain knowledge
├── raw_dataset             # Output of DataCollectionAgent
├── schema:{source_uri}     # Per-source inferred schemas
├── clean_dataset           # Output of DataCleaningAgent
├── cleaning_log            # All cleaning operations with justifications
├── eda_report              # Full EDA findings + hypotheses
├── feature_store           # Engineered features + lineage
├── model_result            # Trained model + evaluation + SHAP
└── insight_report          # Final narrative + recommendations
```

---

## Tool Registry

All agent tools are registered centrally and versioned.

```python
TOOL_REGISTRY = {
    # Data I/O
    "http_fetch":       HTTPFetchTool(timeout=30, retry=3),
    "db_query":         DatabaseQueryTool(pool_size=5),
    "file_reader":      FileReaderTool(formats=["csv","parquet","json","xlsx"]),
    "s3_download":      S3DownloadTool(),
    "kafka_consume":    KafkaConsumerTool(),
    "playwright_scrape": PlaywrightScrapeTool(headless=True),

    # Data processing
    "pandas_exec":      PandasExecutionTool(sandbox=True),
    "sql_transform":    SQLTransformTool(),
    "spark_job":        SparkSubmitTool(),

    # ML
    "sklearn_train":    SklearnTrainingTool(),
    "optuna_optimize":  OptunaTool(n_jobs=-1),
    "shap_explain":     SHAPExplainerTool(),

    # Visualization
    "plotly_chart":     PlotlyChartTool(),
    "seaborn_plot":     SeabornPlotTool(),

    # LLM utilities
    "code_interpreter": CodeInterpreterTool(language="python"),
    "web_search":       WebSearchTool(provider="serper"),
}
```

---

## Evaluation & Observability

### Pipeline Telemetry

Every agent emits structured logs and traces compatible with OpenTelemetry:

```python
@dataclass
class AgentTrace:
    agent_name: str
    stage: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    llm_calls: int
    tool_calls: list[ToolCall]
    tokens_used: int
    cost_usd: float
    reasoning_steps: list[str]
    output_schema: dict
    errors: list[str]
    warnings: list[str]
```

### Quality Gates

Define minimum quality thresholds per stage:

```yaml
# quality_gates.yaml
data_cleaning:
  max_missing_pct_after: 0.05        # Max 5% missing after cleaning
  min_rows_retained: 0.90            # Retain at least 90% of rows

feature_engineering:
  min_features: 5
  max_features: 100
  min_mutual_info_score: 0.01        # Drop features with near-zero MI

modeling:
  classification:
    min_roc_auc: 0.70
    max_train_test_gap: 0.10         # Flag if train AUC >> test AUC
  regression:
    min_r2: 0.50
    max_mape: 0.30
```

---

## Full Stack Implementation

### Directory Structure

```
agentic-pipeline/
├── agents/
│   ├── base_agent.py
│   ├── data_collection_agent.py
│   ├── data_cleaning_agent.py
│   ├── eda_agent.py
│   ├── feature_engineering_agent.py
│   ├── modeling_agent.py
│   └── insight_generation_agent.py
├── orchestrator/
│   ├── pipeline_orchestrator.py
│   └── quality_gates.py
├── memory/
│   ├── memory_store.py
│   └── lineage_tracker.py
├── tools/
│   ├── registry.py
│   ├── data_tools.py
│   ├── ml_tools.py
│   └── viz_tools.py
├── config/
│   ├── pipeline_config.yaml
│   └── quality_gates.yaml
├── tests/
│   ├── test_agents/
│   └── test_integration/
├── notebooks/
│   └── pipeline_walkthrough.ipynb
└── main.py
```

### Entry Point

```python
# main.py

from orchestrator import PipelineOrchestrator
from memory import MemoryStore
from config import PipelineConfig

config = PipelineConfig(
    problem_statement="""
        Predict 30-day customer churn for a SaaS product.
        Minimize false negatives (missed churners) — recall
        is more important than precision for this use case.
    """,
    domain_context="""
        B2B SaaS, subscription billing, monthly active users.
        Churn = account cancellation or non-renewal within 30 days.
        High-value accounts (>$10k ARR) should be treated separately.
    """,
    sources=[
        DataSource(type="database", uri="postgresql://prod/analytics",
                   query="SELECT * FROM customer_events WHERE ..."),
        DataSource(type="file", path="s3://data/customer_profiles.parquet"),
    ],
    target_column="churned_30d",
    task_type="classification",
    human_checkpoints=["data_cleaning", "modeling"],  # Pause for approval
    quality_gates="config/quality_gates.yaml"
)

memory = MemoryStore(backend="redis")
orchestrator = PipelineOrchestrator(config=config, memory=memory)
result = orchestrator.run()

print(result.insight_report.executive_summary)
```

### Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.11"
anthropic = "^0.28"
openai = "^1.35"
pandas = "^2.2"
numpy = "^1.26"
scikit-learn = "^1.5"
xgboost = "^2.0"
lightgbm = "^4.4"
optuna = "^3.6"
shap = "^0.45"
plotly = "^5.22"
seaborn = "^0.13"
playwright = "^1.44"
redis = "^5.0"
networkx = "^3.3"
opentelemetry-sdk = "^1.24"
pydantic = "^2.7"
```

---

## Best Practices & Anti-Patterns

### ✅ Best Practices

**Agent Design**
- Give each agent a single, well-scoped responsibility (Unix philosophy)
- Include explicit output schemas — agents should know what success looks like
- Always include a reasoning trace — never just the action, always the why
- Build idempotent agents — rerunning should produce the same result
- Test agents in isolation before wiring them into the orchestrator

**LLM Usage**
- Use structured output formats (JSON schemas) to prevent hallucination drift
- Include few-shot examples in system prompts for domain-specific operations
- Set temperature=0 for code generation and deterministic decisions
- Cache LLM calls for repeated operations (e.g., schema inference)

**Data Safety**
- Never pass raw PII to LLMs — anonymize before any LLM-facing operation
- Always validate LLM-generated code in a sandboxed environment before execution
- Log every data transformation with rollback capability
- Enforce the cleaning log as immutable — append-only

**Reliability**
- Implement circuit breakers for external data sources
- Use exponential backoff for LLM API rate limits
- Store intermediate outputs at every stage checkpoint
- Write integration tests against realistic sample data

### ❌ Anti-Patterns

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Mega-agent | One LLM call doing everything | Break into specialized agents |
| Stateless pipeline | No shared context between stages | Use the memory store pattern |
| Silent failures | Agent fails but pipeline continues | Validate outputs at every stage |
| LLM for everything | Using LLM for simple deterministic logic | Use code for determinism; LLM for reasoning |
| No lineage tracking | Can't explain where features came from | Track every transformation in the lineage graph |
| Hard-coded prompts | Brittle to domain changes | Parameterize prompts via config |
| Skipping cross-validation | Overfitting with no detection | Always use CV; track train/test gap |

---

## Extending the Pipeline

### Adding a New Agent

```python
# 1. Subclass BaseAgent
class AnomalyDetectionAgent(BaseAgent):
    def run(self) -> AnomalyResult:
        df = self.memory.retrieve("clean_dataset")
        # ... your logic
        self.memory.store("anomaly_flags", result)
        return result

# 2. Register in orchestrator
STAGES = [
    ...
    ("anomaly_detection", AnomalyDetectionAgent),  # Insert between cleaning and EDA
    ...
]

# 3. Define quality gate
# quality_gates.yaml
anomaly_detection:
  max_anomaly_pct: 0.05
```

### Multi-Modal Extensions

The pipeline can be extended to support unstructured inputs:

- **Text data** → Add NLP agent (TF-IDF, embeddings, sentiment, NER)
- **Image data** → Add vision agent (CLIP embeddings, object detection features)
- **Time series** → Add temporal agent (STL decomposition, spectral features)
- **Graph data** → Add graph agent (node embeddings, centrality features)

---

*This pipeline blueprint is designed to be incrementally adopted — start with manual tool calls and progressively automate stages as confidence in each agent grows.*
