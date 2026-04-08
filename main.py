#!/usr/bin/env python3
"""
Autonomous Analysis System — Entry Point
Runs the full agentic AI pipeline and optionally launches the dashboard.

Usage:
    python main.py                                    # Run with defaults
    python main.py --data path/to/data.csv            # Custom dataset
    python main.py --config config/pipeline_config.yaml
    python main.py --dashboard                         # Launch dashboard after
    python main.py --generate-data                     # Generate sample data only
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

from core.llm_client import LLMClient
from core.models import DataSource, LLMConfig, PipelineConfig, TaskType
from memory.memory_store import MemoryStore
from orchestrator.pipeline_orchestrator import PipelineOrchestrator

# Import tools to trigger @tool decorator registration
import tools.data_tools   # noqa: F401
import tools.ml_tools     # noqa: F401
import tools.viz_tools    # noqa: F401
import tools.api_tools    # noqa: F401

# ─── Logging Setup ───
def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s | %(levelname)-5s | %(message)s"
    datefmt = "%H:%M:%S"

    # Force UTF-8 on the stream handler so emojis / box-drawing chars
    # don't crash on Windows cp1252 consoles.
    import io
    stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logging.basicConfig(level=level, handlers=[handler])

    # Quiet noisy libraries
    for lib in ("urllib3", "matplotlib", "PIL", "optuna", "lightgbm"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def load_config(config_path: str, data_override: str = "",
                target_override: str = "") -> PipelineConfig:
    """Load pipeline configuration from YAML."""
    config_data = {}

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # Build config
    sources = []
    for src in config_data.get("sources", []):
        sources.append(DataSource(**src))

    llm_data = config_data.get("llm", {})
    llm_config = LLMConfig(
        provider=llm_data.get("provider", "auto"),
        model=llm_data.get("model", ""),
        temperature=llm_data.get("temperature", 0.1),
        max_tokens=llm_data.get("max_tokens", 4096),
        base_url=llm_data.get("base_url", ""),
    )

    task_type_str = config_data.get("task_type", "classification")
    try:
        task_type = TaskType(task_type_str)
    except ValueError:
        task_type = TaskType.CLASSIFICATION

    config = PipelineConfig(
        problem_statement=config_data.get("problem_statement", "Data analysis task"),
        domain_context=config_data.get("domain_context", ""),
        sources=sources,
        target_column=target_override or config_data.get("target_column", ""),
        task_type=task_type,
        human_checkpoints=config_data.get("human_checkpoints", []),
        quality_gates_path=config_data.get("quality_gates_path", "config/quality_gates.yaml"),
        llm=llm_config,
        output_dir=config_data.get("output_dir", "output"),
    )

    # Override data path
    if data_override:
        config.sources = [DataSource(source_type="file", path=data_override)]

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Analysis System — Agentic AI Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="",
                        help="Path to dataset (CSV/Parquet/JSON)")
    parser.add_argument("--config", type=str, default="config/pipeline_config.yaml",
                        help="Path to pipeline config YAML")
    parser.add_argument("--target", type=str, default="",
                        help="Target column name (overrides config)")
    parser.add_argument("--task-type", type=str, default="",
                        choices=["classification", "regression"],
                        help="Task type (overrides config)")
    parser.add_argument("--dashboard", action="store_true",
                        help="Launch web dashboard after pipeline completes")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate sample dataset and exit")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--output", type=str, default="",
                        help="Output directory (overrides config)")
    parser.add_argument("--provider", type=str, default="",
                        help="Force LLM provider (openai/anthropic/gemini/ollama/rule_based)")

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # ── Generate Data Only ──
    if args.generate_data:
        from data.generate_sample_data import generate_churn_data
        generate_churn_data()
        return

    # ── Load Config ──
    config = load_config(args.config, args.data, args.target)

    if args.task_type:
        config.task_type = TaskType(args.task_type)
    if args.output:
        config.output_dir = args.output
    if args.provider:
        config.llm.provider = args.provider

    # ── Ensure data exists ──
    data_path = ""
    is_api_source = False
    if config.sources:
        data_path = config.sources[0].path or ""
        is_api_source = config.sources[0].source_type != "file"

    if is_api_source:
        # For API sources, data_path is irrelevant — agents fetch at runtime.
        # Store a placeholder so downstream code doesn't see an empty path.
        if not data_path:
            data_path = "<api_source>"
    elif not data_path or not os.path.exists(data_path):
        logger.info("No dataset found. Generating sample data...")
        from data.generate_sample_data import generate_churn_data
        generate_churn_data(output_path="data/sample_churn_data.csv")
        data_path = "data/sample_churn_data.csv"
        config.sources = [DataSource(source_type="file", path=data_path)]

    # ── Initialize Pipeline ──
    memory = MemoryStore()
    memory.store("data_path", data_path)
    memory.store("data_sources", [src.model_dump() for src in config.sources])

    # ── Run Pipeline ──
    logger.info("🚀 Starting Autonomous Analysis Pipeline...")
    orchestrator = PipelineOrchestrator(config=config, memory=memory)
    result = orchestrator.run()

    # ── Dashboard ──
    if args.dashboard:
        logger.info("\n🌐 Launching dashboard at http://localhost:5000 ...")
        try:
            from dashboard.app import create_app
            app = create_app(config.output_dir)
            app.run(host="0.0.0.0", port=5000, debug=False)
        except ImportError as e:
            logger.error(f"Dashboard failed to start: {e}")
            logger.info("Install Flask: pip install flask")
    else:
        logger.info(f"\n📊 Results saved to {config.output_dir}/")
        logger.info(f"   Launch dashboard: python main.py --dashboard")

    return result


if __name__ == "__main__":
    main()
