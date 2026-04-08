"""
Unified LLM client facade for the Autonomous Analysis System.
Provides high-level methods used by all agents, delegating to the active provider.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from core.llm_providers import BaseLLMProvider, detect_provider

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified facade for LLM interactions.
    All agents use this class — it delegates to the auto-detected provider.
    """

    def __init__(self, provider: str = "auto", model: str = "",
                 temperature: float = 0.1, max_tokens: int = 4096,
                 base_url: str = ""):
        self.provider: BaseLLMProvider = detect_provider(
            config_provider=provider,
            config_model=model,
            config_base_url=base_url,
        )
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens
        self.call_log: list[dict] = []

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_name(self) -> str:
        return self.provider.model

    @property
    def total_tokens(self) -> int:
        return self.provider.total_tokens

    @property
    def call_count(self) -> int:
        return self.provider.call_count

    # ──────────────────────────────────────
    # Core chat methods
    # ──────────────────────────────────────

    def chat(self, messages: list[dict],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """Send a chat completion request and return the response text."""
        t = temperature if temperature is not None else self.default_temperature
        mt = max_tokens if max_tokens is not None else self.default_max_tokens

        start = time.time()
        try:
            result = self._retry(lambda: self.provider.chat(
                messages=messages, temperature=t,
                max_tokens=mt, json_mode=False
            ))
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            raise

        duration = time.time() - start
        self.call_log.append({
            "method": "chat",
            "duration": round(duration, 2),
            "tokens": self.provider.total_tokens,
        })
        return result

    def chat_json(self, messages: list[dict],
                  temperature: Optional[float] = None,
                  max_tokens: Optional[int] = None) -> dict:
        """Send a chat request expecting a JSON response."""
        t = temperature if temperature is not None else self.default_temperature
        mt = max_tokens if max_tokens is not None else self.default_max_tokens

        start = time.time()
        try:
            raw = self._retry(lambda: self.provider.chat(
                messages=messages, temperature=t,
                max_tokens=mt, json_mode=True
            ))
        except Exception as e:
            logger.error(f"LLM chat_json failed: {e}")
            raise

        duration = time.time() - start
        self.call_log.append({
            "method": "chat_json",
            "duration": round(duration, 2),
        })

        # Parse JSON — handle potential markdown code fences
        return self._parse_json(raw)

    # ──────────────────────────────────────
    # High-level methods used by agents
    # ──────────────────────────────────────

    def plan(self, task: str, context: str = "",
             tools: list[str] | None = None) -> str:
        """Ask the LLM to create an execution plan."""
        system = (
            "You are a data science AI agent planner. "
            "Create a detailed step-by-step plan for the given task. "
            "Be specific about which tools to use and what data to process."
        )
        user_msg = f"Task: {task}\n\nContext:\n{context}"
        if tools:
            user_msg += f"\n\nAvailable tools: {', '.join(tools)}"

        return self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])

    def generate_code(self, instruction: str, context: str = "",
                      language: str = "python") -> str:
        """Generate code for a data transformation."""
        system = (
            f"You are a {language} code generator for data science. "
            "Generate ONLY executable code, no explanations. "
            "The input DataFrame is called 'df'. "
            "Store the result in 'df_result'. "
            "Use only pandas and numpy. No imports needed (they are pre-imported as pd and np)."
        )
        user_msg = f"Instruction: {instruction}"
        if context:
            user_msg += f"\n\nContext:\n{context}"

        response = self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ], temperature=0.0)

        # Strip markdown code fences if present
        return self._extract_code(response)

    def synthesize(self, findings: str, context: str = "") -> str:
        """Synthesize findings into a narrative."""
        system = (
            "You are a data science report writer. "
            "Synthesize the given findings into a clear, actionable narrative. "
            "Use specific numbers and cite evidence."
        )
        user_msg = f"Findings:\n{findings}"
        if context:
            user_msg += f"\n\nDomain context:\n{context}"

        return self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])

    def select(self, options: list[str], criteria: str,
               context: str = "") -> str:
        """Ask the LLM to select the best option."""
        system = "You are a decision-making AI. Select the best option and explain why."
        user_msg = (
            f"Options: {', '.join(options)}\n"
            f"Criteria: {criteria}"
        )
        if context:
            user_msg += f"\n\nContext:\n{context}"

        return self.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])

    # ──────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────

    def _retry(self, fn, max_retries: int = 3,
               base_delay: float = 1.0):
        """Retry with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return fn()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

    def _parse_json(self, raw: str) -> dict:
        """Parse JSON from LLM response, handling code fences."""
        raw = raw.strip()

        # Remove markdown code fences
        if raw.startswith("```"):
            lines = raw.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [l for l in lines if not l.strip().startswith("```")]
            raw = "\n".join(lines)

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to find JSON within the text
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end])
                except json.JSONDecodeError:
                    pass

            # Try array
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                try:
                    return {"data": json.loads(raw[start:end])}
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON from LLM response: {raw[:200]}")
            return {"raw_response": raw}

    def _extract_code(self, response: str) -> str:
        """Extract code from a response that might contain markdown fences."""
        response = response.strip()

        if "```" in response:
            lines = response.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_code = not in_code
                    continue
                if in_code:
                    code_lines.append(line)
            if code_lines:
                return "\n".join(code_lines)

        return response
