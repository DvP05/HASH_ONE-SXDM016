"""
Central tool registry with @tool decorator for registration.
Tools are callable units that agents can invoke for specific operations.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class Tool:
    """Wraps a callable function as a registered tool."""

    def __init__(self, name: str, func: Callable, description: str = "",
                 category: str = "general"):
        self.name = name
        self.func = func
        self.description = description
        self.category = category
        self.call_count = 0

    def __call__(self, *args, **kwargs) -> Any:
        self.call_count += 1
        logger.debug(f"Tool '{self.name}' invoked (call #{self.call_count})")
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:
        return f"Tool(name='{self.name}', category='{self.category}')"


class ToolRegistry:
    """Central registry of all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, func: Callable,
                 description: str = "", category: str = "general") -> Tool:
        """Register a tool function."""
        tool = Tool(name=name, func=func, description=description, category=category)
        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")
        return tool

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_tools(self, category: str = "") -> list[Tool]:
        """List all tools, optionally filtered by category."""
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())

    def list_names(self, category: str = "") -> list[str]:
        """List tool names."""
        return [t.name for t in self.list_tools(category)]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)


# Global registry instance
registry = ToolRegistry()


def tool(name: str = "", description: str = "", category: str = "general"):
    """Decorator to register a function as a tool."""
    def decorator(func: Callable):
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        registry.register(tool_name, func, tool_desc, category)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tool_name = tool_name
        return wrapper

    return decorator
