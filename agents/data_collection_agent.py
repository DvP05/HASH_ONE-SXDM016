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
                    # ── Fetch global FIRMS data ──
                    self.log("🛰️ Executing NASA FIRMS API extraction (global)...")
                    try:
                        df = self.execute_tool("fetch_nasa_firms_global")
                        if df is not None and not df.empty:
                            frames.append(df)
                            schema = self._infer_schema(df)
                            self.memory.store("schema:nasa_firms_global", schema, source="data_collection")
                            self.log(f"   FIRMS Global: {len(df)} rows, {len(df.columns)} columns")
                    except Exception as e:
                        self.log(f"⚠️ FIRMS global failed, trying regional: {e}", level="warning")
                        map_key = source.get("map_key", "VIIRS_SNPP_NRT")
                        area_coords = source.get("area_coords", "world")
                        df = self.execute_tool("fetch_nasa_firms", map_key=map_key, area_coords=area_coords)
                        frames.append(df)
                        schema = self._infer_schema(df)
                        self.memory.store(f"schema:nasa_api_{map_key}", schema, source="data_collection")
                        self.log(f"   FIRMS: {len(df)} rows, {len(df.columns)} columns (enriched with weather features)")

                    # ── Fetch EONET events ──
                    try:
                        self.log("🌍 Fetching NASA EONET natural disaster events...")
                        eonet_df = self.execute_tool("fetch_nasa_eonet", days=30, limit=100)
                        if eonet_df is not None and not eonet_df.empty:
                            self.log(f"   EONET: {len(eonet_df)} event observations fetched")
                            self.memory.store("eonet_events", eonet_df, source="data_collection")
                    except Exception as e:
                        self.log(f"⚠️ EONET fetch skipped: {e}", level="warning")

                    # ── Load cached live data if available ──
                    try:
                        from scheduler import load_latest_cache
                        cache = load_latest_cache()
                        if cache.get("firms") is not None and not cache["firms"].empty:
                            self.log(f"📦 Loading cached FIRMS data: {len(cache['firms'])} rows")
                            frames.append(cache["firms"])
                    except Exception:
                        pass

                    # ── Load Kaggle training data ──
                    try:
                        self.log("📚 Loading Kaggle disaster training data...")
                        from data.kaggle_downloader import generate_disaster_training_data
                        kaggle_df = generate_disaster_training_data()
                        if kaggle_df is not None and not kaggle_df.empty:
                            self.log(f"   Kaggle Training Data: {len(kaggle_df)} rows, {len(kaggle_df.columns)} columns")
                            self.log(f"   Event types: {kaggle_df['event_type'].value_counts().to_dict()}")
                            self.memory.store("kaggle_training_data", kaggle_df, source="data_collection")
                            # Merge selected columns that overlap with FIRMS data
                            kaggle_overlap = self._align_kaggle_to_firms(kaggle_df)
                            if kaggle_overlap is not None and not kaggle_overlap.empty:
                                frames.append(kaggle_overlap)
                                self.log(f"   Merged {len(kaggle_overlap)} Kaggle rows into training set")
                    except Exception as e:
                        self.log(f"⚠️ Kaggle data loading skipped: {e}", level="warning")
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

    def _align_kaggle_to_firms(self, kaggle_df) -> 'pd.DataFrame | None':
        """
        Align Kaggle training data columns to match FIRMS schema for merging.
        Maps Kaggle columns to the enriched FIRMS columns.
        """
        import pandas as pd
        import numpy as np

        if kaggle_df is None or kaggle_df.empty:
            return None

        df = kaggle_df.copy()

        # Map Kaggle columns to FIRMS-enriched schema
        col_mapping = {
            "temperature": "temperature_2m",
            "humidity": "relative_humidity",
            "severity": "frp",
        }

        for old_col, new_col in col_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # Add missing FIRMS columns with defaults
        if "brightness" not in df.columns:
            frp = df.get("frp", pd.Series([10.0] * len(df)))
            df["brightness"] = 300 + frp.fillna(10) * 1.2 + np.random.normal(0, 8, len(df))

        if "confidence" not in df.columns:
            # FIX: Avoid using crop_impact to derive confidence (DIRECT LEAK)
            # Instead, derive it from raw intensity (frp or severity) with noise
            intensity = df.get("frp", df.get("severity", pd.Series([10.0] * len(df)))).fillna(10)
            
            # Create a noisy categorical mapping
            # h: high intensity (>30), n: nominal (10-30), l: low (<10)
            # We add Gaussian noise to the intensity before thresholding to make it non-deterministic
            noisy_intensity = intensity + np.random.normal(0, 5, len(df))
            
            df["confidence"] = np.where(
                noisy_intensity > 30, "h",
                np.where(noisy_intensity > 10, "n", "l")
            )

        if "satellite" not in df.columns:
            df["satellite"] = "N"

        if "daynight" not in df.columns:
            df["daynight"] = "D"

        if "acq_date" not in df.columns:
            if "year" in df.columns and "month" in df.columns:
                df["acq_date"] = df.apply(
                    lambda r: f"{int(r.get('year', 2025))}-{int(r.get('month', 1)):02d}-15", axis=1
                )
            else:
                df["acq_date"] = "2025-01-15"

        df["data_source"] = df.get("data_source", "kaggle_training")

        return df
