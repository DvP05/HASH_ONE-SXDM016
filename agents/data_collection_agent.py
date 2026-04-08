"""
Stage 1 — Data Collection Agent
Ingests, validates, and normalizes raw data from configured sources.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from agents.base_agent import BaseAgent
from core.models import DataProfile

logger = logging.getLogger(__name__)


class DataCollectionAgent(BaseAgent):
    """
    Autonomously discovers, ingests, validates schema, and normalizes
    raw data from file sources (CSV, Parquet, JSON, Excel).
    """

    def run(self, hints: str = "") -> dict:
        self.start_timer()
        self.log("═══ Stage 1: Data Collection ═══")

        # Get data source from config
        sources = self.memory.retrieve("data_sources", [])
        data_path = self.memory.retrieve("data_path", "")

        frames = []

        if data_path and os.path.exists(data_path):
            self.log(f"📂 Loading data from: {data_path}")
            df = self.execute_tool("file_reader", path=data_path)
            frames.append(df)

            # Infer schema
            schema = self._infer_schema(df)
            self.memory.store(f"schema:{data_path}", schema, source="data_collection")
            self.log(f"📋 Schema inferred: {len(df.columns)} columns, {len(df)} rows")

        elif sources:
            for source in sources:
                source_type = source.get("source_type", "file")
                if source_type == "nasa_api":
                    self.log("🛰️ Executing NASA FIRMS API extraction...")
                    map_key = source.get("map_key", "VIIRS_SNPP_NRT")
                    area_coords = source.get("area_coords", "world")
                    df = self.execute_tool("fetch_nasa_firms", map_key=map_key, area_coords=area_coords)
                    frames.append(df)
                    schema = self._infer_schema(df)
                    self.memory.store(f"schema:nasa_api_{map_key}", schema, source="data_collection")
                    self.log(f"   FIRMS: {len(df)} rows, {len(df.columns)} columns (enriched with weather features)")

                    # Also fetch EONET natural disaster events
                    try:
                        self.log("🌍 Fetching NASA EONET natural disaster events...")
                        eonet_df = self.execute_tool("fetch_nasa_eonet", days=30, limit=50)
                        if eonet_df is not None and not eonet_df.empty:
                            self.log(f"   EONET: {len(eonet_df)} event observations fetched")
                            self.memory.store("eonet_events", eonet_df, source="data_collection")
                    except Exception as e:
                        self.log(f"⚠️ EONET fetch skipped: {e}", level="warning")
                else:
                    path = source.get("path", source.get("uri", ""))
                    if path and os.path.exists(path):
                        self.log(f"📂 Loading from: {path}")
                        df = self.execute_tool("file_reader", path=path)
                        frames.append(df)
                        schema = self._infer_schema(df)
                        self.memory.store(f"schema:{path}", schema, source="data_collection")
        else:
            raise ValueError("No data sources configured. Provide data_path or data_sources.")

        if not frames:
            raise ValueError("No data could be loaded from any source.")

        # Merge if multiple sources
        if len(frames) == 1:
            merged = frames[0]
        else:
            merged = pd.concat(frames, ignore_index=True)
            self.log(f"🔗 Merged {len(frames)} sources into {len(merged)} rows")

        # Basic validation
        self._validate(merged)

        # Store in memory
        self.memory.store("raw_dataset", merged, source="data_collection")
        self.log(f"✅ Data collection complete: {merged.shape[0]} rows × {merged.shape[1]} columns")

        # LLM commentary on the dataset
        try:
            stats = self.execute_tool("compute_stats", df=merged)
            commentary = self.llm_chat(
                system="You are a data quality analyst. Briefly describe the dataset characteristics.",
                user=f"Dataset summary:\n{self._format_stats(stats)}"
            )
            self.memory.store("data_commentary", commentary, source="data_collection")
            self.log(f"💬 LLM Commentary: {commentary[:200]}")
        except Exception as e:
            self.log(f"⚠️ LLM commentary skipped: {e}", level="warning")

        self.stop_timer()
        return {
            "rows": merged.shape[0],
            "columns": merged.shape[1],
            "column_names": list(merged.columns),
        }

    def _infer_schema(self, df: pd.DataFrame) -> dict:
        """Infer schema from DataFrame."""
        return {
            "columns": {
                col: {
                    "dtype": str(df[col].dtype),
                    "nullable": bool(df[col].isnull().any()),
                    "unique_count": int(df[col].nunique()),
                    "sample_values": [str(v) for v in df[col].dropna().head(3).tolist()],
                }
                for col in df.columns
            },
            "row_count": len(df),
            "column_count": len(df.columns),
        }

    def _validate(self, df: pd.DataFrame) -> None:
        """Basic data validation."""
        if df.empty:
            raise ValueError("Dataset is empty!")

        if len(df.columns) == 0:
            raise ValueError("Dataset has no columns!")

        # Check target column exists
        target = self.memory.retrieve("target_column", "")
        if target and target not in df.columns:
            self.log(f"⚠️ Target column '{target}' not found in dataset!", level="warning")
            self.log(f"   Available columns: {list(df.columns)}")

        # Log warnings
        missing_pct = df.isnull().mean()
        high_missing = missing_pct[missing_pct > 0.5]
        if len(high_missing) > 0:
            self.log(f"⚠️ {len(high_missing)} columns have >50% missing values", level="warning")

    def _format_stats(self, stats: dict) -> str:
        """Format stats dict as readable text."""
        lines = []
        lines.append(f"Shape: {stats.get('shape', 'unknown')}")
        lines.append(f"Duplicates: {stats.get('duplicates', 0)}")
        lines.append(f"Numeric columns: {len(stats.get('numeric_columns', []))}")
        lines.append(f"Categorical columns: {len(stats.get('categorical_columns', []))}")

        missing = stats.get("missing_pct", {})
        if missing:
            top_missing = sorted(missing.items(), key=lambda x: x[1], reverse=True)[:5]
            lines.append("Top missing:")
            for col, pct in top_missing:
                if pct > 0:
                    lines.append(f"  {col}: {pct:.1%}")

        return "\n".join(lines)
