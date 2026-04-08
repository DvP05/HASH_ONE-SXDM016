"""
Data I/O and processing tools.
FileReader, PandasExec, ComputeStats.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from core.sandbox import safe_exec, SandboxResult
from tools.registry import tool

logger = logging.getLogger(__name__)


@tool(name="file_reader", description="Read data files (CSV, Parquet, JSON, Excel)", category="data")
def file_reader(path: str, **kwargs) -> pd.DataFrame:
    """Read a data file and return a DataFrame."""
    path = os.path.normpath(path)
    ext = os.path.splitext(path)[1].lower()

    readers = {
        ".csv": pd.read_csv,
        ".parquet": pd.read_parquet,
        ".json": pd.read_json,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
        ".tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
    }

    if ext not in readers:
        raise ValueError(f"Unsupported file format: {ext}")

    logger.info(f"Reading file: {path}")
    df = readers[ext](path, **kwargs)
    logger.info(f"Loaded {len(df)} rows × {len(df.columns)} columns")
    return df


@tool(name="pandas_exec", description="Execute pandas code in a sandbox", category="data")
def pandas_exec(code: str, df: pd.DataFrame,
                extra_vars: dict | None = None) -> SandboxResult:
    """Execute sandboxed pandas code on a DataFrame."""
    logger.info(f"Executing sandboxed code ({len(code)} chars)")
    result = safe_exec(code, df, extra_vars)
    if result.success:
        logger.info(f"Code executed successfully")
    else:
        logger.warning(f"Code execution failed: {result.error}")
    return result


@tool(name="compute_stats", description="Compute descriptive statistics", category="data")
def compute_stats(df: pd.DataFrame, column: str = "") -> dict:
    """Compute comprehensive statistics for a DataFrame or column."""
    if column and column in df.columns:
        series = df[column]
        stats = {
            "count": int(series.count()),
            "missing": int(series.isnull().sum()),
            "missing_pct": round(float(series.isnull().mean()), 4),
            "dtype": str(series.dtype),
            "unique": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            stats.update({
                "mean": round(float(desc.get("mean", 0)), 4),
                "std": round(float(desc.get("std", 0)), 4),
                "min": float(desc.get("min", 0)),
                "25%": float(desc.get("25%", 0)),
                "50%": float(desc.get("50%", 0)),
                "75%": float(desc.get("75%", 0)),
                "max": float(desc.get("max", 0)),
                "skewness": round(float(series.skew()), 4) if len(series.dropna()) > 2 else 0,
                "kurtosis": round(float(series.kurtosis()), 4) if len(series.dropna()) > 3 else 0,
            })
        else:
            top_values = series.value_counts().head(10).to_dict()
            stats["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        return stats

    # Full DataFrame stats
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
        "missing": {col: int(df[col].isnull().sum()) for col in df.columns},
        "missing_pct": {col: round(float(df[col].isnull().mean()), 4) for col in df.columns},
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "category"]).columns),
        "duplicates": int(df.duplicated().sum()),
    }


@tool(name="detect_outliers", description="Detect outliers using IQR method", category="data")
def detect_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> dict[str, int]:
    """Detect outlier counts per numeric column using IQR method."""
    outlier_counts = {}
    for col in df.select_dtypes(include="number").columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_outliers > 0:
            outlier_counts[col] = n_outliers
    return outlier_counts
