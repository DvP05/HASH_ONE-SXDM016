"""
Multi-provider LLM backend abstraction.
Supports OpenAI, Anthropic, Google Gemini, Ollama (local), and rule-based fallback.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str = "", **kwargs):
        self.model = model
        self.total_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    @abstractmethod
    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, json_mode: bool = False) -> str:
        """Send a chat completion request and return the response text."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is currently available."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider (GPT-4o, GPT-4o-mini)."""

    def __init__(self, model: str = "gpt-4o-mini", **kwargs):
        super().__init__(model, **kwargs)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY", "")
                )
            except ImportError:
                raise RuntimeError("openai package not installed. Run: pip install openai")
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, json_mode: bool = False) -> str:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        self.call_count += 1
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        return response.choices[0].message.content or ""

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    @property
    def provider_name(self) -> str:
        return "openai"


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API provider (Claude Sonnet, Haiku)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", **kwargs):
        super().__init__(model, **kwargs)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(
                    api_key=os.environ.get("ANTHROPIC_API_KEY", "")
                )
            except ImportError:
                raise RuntimeError("anthropic package not installed. Run: pip install anthropic")
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, json_mode: bool = False) -> str:
        client = self._get_client()

        # Anthropic uses system as a separate param
        system_msg = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                chat_messages.append(msg)

        if json_mode and system_msg:
            system_msg += "\n\nIMPORTANT: Respond ONLY with valid JSON. No other text."
        elif json_mode:
            system_msg = "Respond ONLY with valid JSON. No other text."

        kwargs = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            kwargs["system"] = system_msg

        response = client.messages.create(**kwargs)
        self.call_count += 1
        self.total_tokens += response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text

    def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def provider_name(self) -> str:
        return "anthropic"


class GeminiProvider(BaseLLMProvider):
    """Google Gemini API provider."""

    def __init__(self, model: str = "gemini-2.0-flash", **kwargs):
        super().__init__(model, **kwargs)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get("GOOGLE_API_KEY", ""))
                self._client = genai.GenerativeModel(self.model)
            except ImportError:
                raise RuntimeError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )
        return self._client

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, json_mode: bool = False) -> str:
        client = self._get_client()

        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        if json_mode:
            prompt_parts.append("Respond ONLY with valid JSON. No other text.")

        combined = "\n\n".join(prompt_parts)

        response = client.generate_content(
            combined,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        self.call_count += 1
        return response.text

    def is_available(self) -> bool:
        return bool(os.environ.get("GOOGLE_API_KEY"))

    @property
    def provider_name(self) -> str:
        return "gemini"


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider."""

    def __init__(self, model: str = "", base_url: str = "", **kwargs):
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        # Strip /v1 suffix if present — this provider uses the native
        # Ollama API (/api/chat, /api/tags), not the OpenAI-compat endpoint.
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        if not model:
            model = self._detect_best_model()
        super().__init__(model, **kwargs)

    def _detect_best_model(self) -> str:
        """Auto-detect the best available model from Ollama."""
        import urllib.request
        import urllib.error

        priority = ["llama3.1", "llama3", "mistral", "qwen2.5",
                     "codellama", "deepseek-coder", "phi3"]
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                available = [m["name"].split(":")[0] for m in data.get("models", [])]
                for preferred in priority:
                    for avail in available:
                        if preferred in avail:
                            return avail
                if available:
                    return available[0]
        except Exception:
            pass
        return "llama3.1"

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, json_mode: bool = False) -> str:
        import urllib.request

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())

        self.call_count += 1
        message = result.get("message", {})
        return message.get("content", "")

    def is_available(self) -> bool:
        import urllib.request
        import urllib.error

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def provider_name(self) -> str:
        return "ollama"


class RuleBasedFallback(BaseLLMProvider):
    """
    Deterministic, heuristic-based responses when no LLM is available.
    Allows the pipeline to run end-to-end without any AI backend.
    """

    def __init__(self, **kwargs):
        super().__init__(model="rule-based", **kwargs)

    def chat(self, messages: list[dict], temperature: float = 0.1,
             max_tokens: int = 4096, json_mode: bool = False) -> str:
        """
        Analyze the user message and return heuristic-based responses.
        """
        user_msg = ""
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
                break

        user_lower = user_msg.lower()
        self.call_count += 1

        # Cleaning plan heuristics
        if "cleaning" in user_lower or "clean" in user_lower:
            if json_mode:
                return json.dumps({
                    "operations": [
                        {
                            "name": "drop_duplicates",
                            "target_column": "__all__",
                            "strategy": "drop_exact_duplicates",
                            "llm_justification": "Remove exact duplicate rows to ensure data integrity.",
                            "code": "df_result = df.drop_duplicates()"
                        },
                        {
                            "name": "fill_numeric_missing",
                            "target_column": "__numeric__",
                            "strategy": "median_imputation",
                            "llm_justification": "Fill missing numeric values with column median (robust to outliers).",
                            "code": "df_result = df.copy()\nfor col in df_result.select_dtypes(include='number').columns:\n    df_result[col] = df_result[col].fillna(df_result[col].median())"
                        },
                        {
                            "name": "fill_categorical_missing",
                            "target_column": "__categorical__",
                            "strategy": "mode_imputation",
                            "llm_justification": "Fill missing categorical values with the most frequent value.",
                            "code": "df_result = df.copy()\nfor col in df_result.select_dtypes(include='object').columns:\n    if df_result[col].isnull().any():\n        df_result[col] = df_result[col].fillna(df_result[col].mode()[0] if len(df_result[col].mode()) > 0 else 'Unknown')"
                        },
                        {
                            "name": "cap_outliers_iqr",
                            "target_column": "__numeric__",
                            "strategy": "iqr_cap",
                            "llm_justification": "Cap extreme outliers using IQR method (3x multiplier).",
                            "code": "df_result = df.copy()\nfor col in df_result.select_dtypes(include='number').columns:\n    Q1, Q3 = df_result[col].quantile(0.25), df_result[col].quantile(0.75)\n    IQR = Q3 - Q1\n    lower, upper = Q1 - 3*IQR, Q3 + 3*IQR\n    df_result[col] = df_result[col].clip(lower, upper)"
                        }
                    ]
                })
            return "Apply standard cleaning: drop duplicates, median impute numeric, mode impute categorical, IQR cap outliers."

        # Feature engineering heuristics
        if "feature" in user_lower and ("engineer" in user_lower or "transform" in user_lower or "plan" in user_lower):
            if json_mode:
                return json.dumps({
                    "transforms": [
                        {
                            "name": "log_transform_skewed",
                            "type": "numeric_transform",
                            "input_columns": ["__auto_skewed__"],
                            "operation": "log1p",
                            "output_columns": ["__auto__"],
                            "rationale": "Log-transform highly skewed numeric features to normalize distribution."
                        },
                        {
                            "name": "standard_scale",
                            "type": "numeric_transform",
                            "input_columns": ["__all_numeric__"],
                            "operation": "standard_scale",
                            "output_columns": ["__auto__"],
                            "rationale": "Standardize all numeric features to zero mean and unit variance."
                        }
                    ]
                })
            return "Apply log transforms to skewed features, standardize numeric columns."

        # Feature selection heuristics
        if "select" in user_lower and "feature" in user_lower:
            if json_mode:
                return json.dumps({"selected_features": [], "reasoning": "Selected by mutual information score."})
            return "Select top features by mutual information score."

        # Model recommendation heuristics
        if "recommend" in user_lower and "model" in user_lower:
            if json_mode:
                return json.dumps({
                    "candidates": ["random_forest", "xgboost", "logistic_regression"],
                    "reasoning": "Standard set for tabular classification."
                })
            return "Recommend: Random Forest, XGBoost, Logistic Regression."

        # Model selection heuristics
        if "select" in user_lower and "model" in user_lower:
            if json_mode:
                return json.dumps({"selected_model": "xgboost", "reasoning": "Best CV score."})
            return "Select model with best cross-validation score."

        # EDA narrative / synthesis
        if "synthe" in user_lower or "narrative" in user_lower or "eda" in user_lower:
            if json_mode:
                return json.dumps({
                    "narrative": "Exploratory analysis reveals several key patterns in the dataset.",
                    "hypotheses": []
                })
            return "The dataset shows typical patterns for the domain with some notable correlations."

        # Insight / recommendation generation
        if "insight" in user_lower or "recommendation" in user_lower or "executive" in user_lower:
            if json_mode:
                return json.dumps({
                    "summary": "Analysis complete. Model performs well on the given dataset.",
                    "recommendations": [
                        {
                            "title": "Focus on Top Predictive Features",
                            "description": "The model identifies key features driving predictions. Focus interventions on these areas.",
                            "estimated_impact": "High",
                            "confidence": 0.75,
                            "supporting_evidence": ["SHAP analysis", "Feature importance"],
                            "action_items": ["Review top features", "Design targeted interventions"],
                            "priority": "high"
                        }
                    ],
                    "next_steps": [
                        "Collect more recent data for retraining",
                        "A/B test recommendations",
                        "Monitor model drift"
                    ]
                })
            return "Key findings summary with actionable recommendations."

        # Hyperparameter search space
        if "search_space" in user_lower or "hyperparameter" in user_lower:
            if json_mode:
                return json.dumps({
                    "n_estimators": ["int", 50, 500],
                    "max_depth": ["int", 3, 15],
                    "learning_rate": ["float", 0.01, 0.3],
                    "min_child_weight": ["int", 1, 10],
                    "subsample": ["float", 0.6, 1.0]
                })
            return "Standard search space for tree-based models."

        # Interpretation / commentary
        if "interpret" in user_lower or "comment" in user_lower:
            if json_mode:
                return json.dumps({"commentary": "Standard distribution observed.", "insights": []})
            return "Standard distribution observed for this variable."

        # Model card / limitations
        if "model_card" in user_lower or "limitation" in user_lower:
            if json_mode:
                return json.dumps({
                    "limitations": "Model trained on historical data; may not generalize to distribution shifts.",
                    "model_card": "Trained model with standard evaluation metrics."
                })
            return "Standard model card with evaluation metrics and known limitations."

        # Schema / validation
        if "schema" in user_lower or "validat" in user_lower:
            if json_mode:
                return json.dumps({"schema": {}, "validation_rules": [], "commentary": "Schema inferred from data."})
            return "Schema inferred from data structure."

        # Default
        if json_mode:
            return json.dumps({"response": "Processed successfully.", "status": "ok"})
        return "Analysis complete."

    def is_available(self) -> bool:
        return True  # Always available

    @property
    def provider_name(self) -> str:
        return "rule_based"


def detect_provider(config_provider: str = "auto",
                    config_model: str = "",
                    config_base_url: str = "") -> BaseLLMProvider:
    """
    Auto-detect the best available LLM provider.

    Detection order:
    1. If config_provider is set (not 'auto'), use that provider
    2. Check for API keys: OPENAI → ANTHROPIC → GOOGLE
    3. Check for Ollama at localhost:11434
    4. Fall back to rule-based heuristics
    """
    provider_str = os.environ.get("LLM_PROVIDER", config_provider).lower()
    model_override = os.environ.get("LLM_MODEL", config_model)

    # Explicit provider selection
    if provider_str != "auto":
        providers_map = {
            "openai": lambda: OpenAIProvider(model=model_override or "gpt-4o-mini"),
            "anthropic": lambda: AnthropicProvider(model=model_override or "claude-sonnet-4-20250514"),
            "gemini": lambda: GeminiProvider(model=model_override or "gemini-2.0-flash"),
            "ollama": lambda: OllamaProvider(model=model_override, base_url=config_base_url),
            "rule_based": lambda: RuleBasedFallback(),
        }
        if provider_str in providers_map:
            provider = providers_map[provider_str]()
            logger.info(f"Using explicit provider: {provider.provider_name} ({provider.model})")
            return provider

    # Auto-detection
    # 1. Check cloud API keys
    if os.environ.get("OPENAI_API_KEY"):
        p = OpenAIProvider(model=model_override or "gpt-4o-mini")
        logger.info(f"Auto-detected OpenAI (model: {p.model})")
        return p

    if os.environ.get("ANTHROPIC_API_KEY"):
        p = AnthropicProvider(model=model_override or "claude-sonnet-4-20250514")
        logger.info(f"Auto-detected Anthropic (model: {p.model})")
        return p

    if os.environ.get("GOOGLE_API_KEY"):
        p = GeminiProvider(model=model_override or "gemini-2.0-flash")
        logger.info(f"Auto-detected Gemini (model: {p.model})")
        return p

    # 2. Check Ollama
    ollama = OllamaProvider(model=model_override, base_url=config_base_url)
    if ollama.is_available():
        logger.info(f"Auto-detected Ollama (model: {ollama.model})")
        return ollama

    # 3. Fallback
    logger.warning("No LLM provider found. Using rule-based fallback.")
    return RuleBasedFallback()
