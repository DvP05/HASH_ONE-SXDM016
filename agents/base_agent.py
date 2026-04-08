"""
Base agent class implementing the ReAct (Reasoning + Acting) pattern.
All pipeline agents inherit from this class.

Agent loop: Observe → Think → Plan → Act → Reflect → Store
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Optional

from core.llm_client import LLMClient
from core.models import AgentTrace, ToolCall
from memory.memory_store import MemoryStore
from tools.registry import ToolRegistry, registry

logger = logging.getLogger(__name__)


class AgentException(Exception):
    """Raised when an agent encounters an unrecoverable error."""
    pass


class BaseAgent:
    """
    Base class for all pipeline agents.
    Implements the ReAct pattern with memory, tools, and reasoning trace.
    """

    MAX_ITERATIONS = 20

    def __init__(self, name: str, memory: MemoryStore,
                 llm: LLMClient | None = None,
                 tools: list[str] | None = None):
        self.name = name
        self.memory = memory
        self.llm = llm or LLMClient()
        self.tool_names = tools or []
        self.trace = AgentTrace(agent_name=name, stage=name)
        self._start_time: float = 0

    def run(self, hints: str = "") -> Any:
        """
        Execute the agent's main logic.
        Override this in subclasses.
        """
        raise NotImplementedError("Subclasses must implement run()")

    def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """Execute a registered tool by name."""
        start = time.time()
        try:
            tool_fn = registry.get(tool_name)
            result = tool_fn(*args, **kwargs)
            duration = time.time() - start

            self.trace.tool_calls.append(ToolCall(
                tool_name=tool_name,
                args={k: str(v)[:100] for k, v in kwargs.items()},
                result_summary=str(type(result).__name__) if result is not None else "",
                duration_seconds=round(duration, 2),
            ))

            self.log(f"🔧 Tool '{tool_name}' completed in {duration:.2f}s")
            return result

        except Exception as e:
            self.log(f"❌ Tool '{tool_name}' failed: {e}", level="error")
            raise

    def log(self, message: str, level: str = "info") -> None:
        """Log a message and add to reasoning trace."""
        log_fn = getattr(logger, level, logger.info)
        log_fn(f"[{self.name}] {message}")
        self.trace.reasoning_steps.append(message)

    def start_timer(self) -> None:
        """Start timing the agent execution."""
        self._start_time = time.time()
        self.trace.start_time = datetime.now(timezone.utc).isoformat()

    def stop_timer(self) -> None:
        """Stop timing and record duration."""
        self.trace.end_time = datetime.now(timezone.utc).isoformat()
        self.trace.duration_seconds = round(time.time() - self._start_time, 2)

    def llm_chat(self, system: str, user: str,
                 temperature: float | None = None) -> str:
        """Convenience method for LLM chat with trace logging."""
        self.trace.llm_calls += 1
        return self.llm.chat([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=temperature)

    def llm_json(self, system: str, user: str,
                 temperature: float | None = None) -> dict:
        """Convenience method for LLM JSON response."""
        self.trace.llm_calls += 1
        return self.llm.chat_json([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature=temperature)

    def get_trace(self) -> AgentTrace:
        """Get the full agent trace."""
        self.trace.tokens_used = self.llm.total_tokens
        return self.trace
