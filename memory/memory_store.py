"""
Versioned in-process memory store with lineage tracking.
No external dependencies (no Redis) — everything in-process for the demo.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DataEntry:
    """A versioned entry in the memory store."""

    def __init__(self, value: Any, version: int = 1):
        self.value = value
        self.version = version
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.created_at


class MemoryStore:
    """
    In-process key-value memory store with versioning and lineage tracking.
    Used by all agents to share data between pipeline stages.
    """

    def __init__(self):
        self._store: dict[str, DataEntry] = {}
        self._lineage: list[dict] = []  # Simple list-based lineage
        self._history: dict[str, list[DataEntry]] = {}

    def store(self, key: str, value: Any, source: str = "",
              persist: bool = False) -> None:
        """
        Store a value with versioning.

        Args:
            key: Storage key
            value: Value to store (will be shallow-copied for safety)
            source: Origin of this value (for lineage tracking)
            persist: Whether to mark for persistence (future use)
        """
        current_version = self._store[key].version if key in self._store else 0
        entry = DataEntry(value=value, version=current_version + 1)

        # Track history
        if key not in self._history:
            self._history[key] = []
        self._history[key].append(entry)

        self._store[key] = entry

        # Lineage
        if source:
            self._lineage.append({
                "key": key,
                "source": source,
                "version": entry.version,
                "timestamp": entry.created_at,
            })

        logger.debug(f"Memory: stored '{key}' (v{entry.version})")

    def retrieve(self, key: str, default: Any = None,
                 version: Optional[int] = None) -> Any:
        """
        Retrieve a value from the store.

        Args:
            key: Storage key
            default: Default value if key not found
            version: Optional zero-based historical version index.
                     If None, return latest value.

        Returns:
            The stored value, or default if not found
        """
        if version is not None:
            history = self._history.get(key, [])
            if not history:
                if default is not None:
                    return default
                logger.warning(f"Memory: key '{key}' not found")
                return None

            if version < 0 or version >= len(history):
                raise IndexError(
                    f"Version index {version} out of range for key '{key}' "
                    f"(available: 0..{len(history) - 1})"
                )
            return history[version].value

        entry = self._store.get(key)
        if entry is None:
            if default is not None:
                return default
            logger.warning(f"Memory: key '{key}' not found")
            return None
        return entry.value

    def has(self, key: str) -> bool:
        """Check if a key exists in the store."""
        return key in self._store

    def list_keys(self) -> list[str]:
        """List all keys in the store."""
        return list(self._store.keys())

    def get_version(self, key: str) -> int:
        """Get the current version number for a key."""
        entry = self._store.get(key)
        return entry.version if entry else 0

    def snapshot(self) -> dict:
        """
        Export a serializable snapshot of the store.
        Skips non-serializable values (DataFrames, models, etc.).
        """
        snapshot = {}
        for key, entry in self._store.items():
            try:
                # Try JSON serialization
                json.dumps(entry.value, default=str)
                snapshot[key] = {
                    "value": entry.value,
                    "version": entry.version,
                    "updated_at": entry.updated_at,
                }
            except (TypeError, ValueError):
                snapshot[key] = {
                    "value": f"<{type(entry.value).__name__}>",
                    "version": entry.version,
                    "updated_at": entry.updated_at,
                }
        return snapshot

    def get_lineage(self) -> list[dict]:
        """Get the full lineage graph."""
        return self._lineage.copy()

    def clear(self) -> None:
        """Clear all stored data."""
        self._store.clear()
        self._history.clear()
        self._lineage.clear()
        logger.info("Memory store cleared")

    def summary(self) -> str:
        """Get a human-readable summary of stored keys."""
        lines = [f"MemoryStore: {len(self._store)} keys"]
        for key, entry in self._store.items():
            val_type = type(entry.value).__name__
            lines.append(f"  {key}: <{val_type}> v{entry.version}")
        return "\n".join(lines)
