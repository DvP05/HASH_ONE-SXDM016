"""
Sandboxed code execution for LLM-generated pandas/numpy code.
Prevents dangerous operations while allowing data transformations.
"""

from __future__ import annotations

import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

import numpy as np
import pandas as pd


# Modules allowed in the sandbox
ALLOWED_MODULES = {
    "pd": pd,
    "pandas": pd,
    "np": np,
    "numpy": np,
}

# Built-in functions allowed in the sandbox
ALLOWED_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter",
    "float", "frozenset", "getattr", "hasattr", "int", "isinstance",
    "issubclass", "iter", "len", "list", "map", "max", "min",
    "next", "print", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "type", "zip",
}

# Dangerous patterns to block
BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "import socket",
    "import http",
    "import urllib",
    "import requests",
    "__import__",
    "eval(",
    "exec(",
    "compile(",
    "open(",
    "file(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
    "breakpoint(",
    "exit(",
    "quit(",
]


class SandboxError(Exception):
    """Raised when sandboxed code violates safety constraints."""
    pass


class SandboxResult:
    """Result of sandboxed code execution."""

    def __init__(self, success: bool, result: Any = None,
                 stdout: str = "", stderr: str = "", error: str = ""):
        self.success = success
        self.result = result
        self.stdout = stdout
        self.stderr = stderr
        self.error = error


def validate_code(code: str) -> list[str]:
    """
    Check code for dangerous patterns before execution.
    Returns a list of violations found.
    """
    violations = []
    code_lower = code.lower()

    for pattern in BLOCKED_PATTERNS:
        if pattern.lower() in code_lower:
            violations.append(f"Blocked pattern detected: '{pattern}'")

    return violations


def safe_exec(code: str, df: pd.DataFrame,
              extra_vars: dict | None = None,
              timeout: int = 30) -> SandboxResult:
    """
    Execute LLM-generated code in a restricted environment.

    Args:
        code: Python code string to execute
        df: Input DataFrame available as 'df' in the code
        extra_vars: Additional variables to make available
        timeout: Maximum execution time in seconds

    Returns:
        SandboxResult with the output DataFrame or error info
    """
    # Validate code safety
    violations = validate_code(code)
    if violations:
        return SandboxResult(
            success=False,
            error=f"Code safety violations: {'; '.join(violations)}"
        )

    # Build restricted namespace
    safe_builtins = {
        name: __builtins__[name] if isinstance(__builtins__, dict)
        else getattr(__builtins__, name)
        for name in ALLOWED_BUILTINS
        if (isinstance(__builtins__, dict) and name in __builtins__) or
           (not isinstance(__builtins__, dict) and hasattr(__builtins__, name))
    }

    namespace = {
        "__builtins__": safe_builtins,
        "df": df.copy(),
        **ALLOWED_MODULES,
    }

    if extra_vars:
        namespace.update(extra_vars)

    # Capture stdout/stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, namespace)  # noqa: S102

        # Look for the result DataFrame
        result_df = namespace.get("df_result", namespace.get("df", df))

        return SandboxResult(
            success=True,
            result=result_df,
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
        )

    except Exception as e:
        return SandboxResult(
            success=False,
            error=f"{type(e).__name__}: {str(e)}",
            stdout=stdout_capture.getvalue(),
            stderr=stderr_capture.getvalue(),
        )


def safe_exec_simple(code: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Simplified version that returns the DataFrame directly or raises.
    """
    result = safe_exec(code, df)
    if not result.success:
        raise SandboxError(result.error)
    if isinstance(result.result, pd.DataFrame):
        return result.result
    return df
