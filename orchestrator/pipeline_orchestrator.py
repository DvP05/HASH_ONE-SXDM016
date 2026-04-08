"""
Pipeline Orchestrator — coordinates all 6 agent stages.
Handles execution order, retry logic, quality gates, and telemetry.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any

from agents.base_agent import AgentException, BaseAgent
from agents.data_cleaning_agent import DataCleaningAgent
from agents.data_collection_agent import DataCollectionAgent
from agents.eda_agent import EDAAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.insight_agent import InsightAgent
from agents.modeling_agent import ModelingAgent
from core.llm_client import LLMClient
from core.models import PipelineConfig, PipelineResult, StageResult
from memory.memory_store import MemoryStore
from orchestrator.quality_gates import QualityGates

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Top-level coordinator that runs all agents in sequence
    with error handling, retry logic, and quality gates.
    """

    STAGES = [
        ("data_collection", DataCollectionAgent),
        ("data_cleaning", DataCleaningAgent),
        ("eda", EDAAgent),
        ("feature_engineering", FeatureEngineeringAgent),
        ("modeling", ModelingAgent),
        ("insight_generation", InsightAgent),
    ]

    def __init__(self, config: PipelineConfig, memory: MemoryStore,
                 llm: LLMClient | None = None):
        self.config = config
        self.memory = memory
        self.llm = llm or LLMClient(
            provider=config.llm.provider,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            base_url=config.llm.base_url,
        )
        self.quality_gates = QualityGates(config.quality_gates_path)
        self.pipeline_start: float = 0

        # Store configuration in memory
        self.memory.store("problem_statement", config.problem_statement)
        self.memory.store("domain_context", config.domain_context)
        self.memory.store("target_column", config.target_column)
        self.memory.store("task_type", config.task_type.value)
        self.memory.store("output_dir", config.output_dir)

    def run(self) -> PipelineResult:
        """Execute the full pipeline."""
        self.pipeline_start = time.time()
        results: dict[str, StageResult] = {}

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.config.output_dir, "charts"), exist_ok=True)

        logger.info("\n" + "=" * 70)
        logger.info("  🚀 AUTONOMOUS ANALYSIS PIPELINE — STARTING")
        logger.info(f"  LLM Provider: {self.llm.provider_name} ({self.llm.model_name})")
        logger.info(f"  Target: {self.config.target_column}")
        logger.info(f"  Task: {self.config.task_type.value}")
        logger.info("=" * 70)

        for stage_name, AgentClass in self.STAGES:
            logger.info(f"\n{'─' * 60}")
            logger.info(f"  STAGE: {stage_name.upper()}")
            logger.info(f"{'─' * 60}")

            # Human checkpoint
            if stage_name in self.config.human_checkpoints:
                logger.info(f"⏸️  Human checkpoint at {stage_name} — skipping (autonomous mode)")

            # Create and run agent
            agent = AgentClass(
                name=stage_name,
                memory=self.memory,
                llm=self.llm,
            )

            try:
                result = self._run_with_retry(agent, stage_name)
                stage_result = StageResult(
                    stage_name=stage_name,
                    status="success",
                    trace=agent.get_trace(),
                    output_summary=str(result)[:500] if result else "",
                )

                # Quality gate
                gate_result = self.quality_gates.validate(
                    stage_name, result, self.memory
                )
                stage_result.quality_gate_passed = gate_result.passed

                if not gate_result.passed:
                    logger.warning(f"⚠️ Quality gate issues: {gate_result.issues}")
                    # Try to recover with a retry
                    if len(gate_result.issues) < 3:
                        logger.info("🔄 Attempting recovery retry...")
                        try:
                            agent2 = AgentClass(
                                name=stage_name,
                                memory=self.memory,
                                llm=self.llm,
                            )
                            result2 = agent2.run(hints=f"Previous issues: {gate_result.issues}")
                            gate2 = self.quality_gates.validate(
                                stage_name, result2, self.memory
                            )
                            if gate2.passed:
                                logger.info("✅ Recovery successful!")
                                stage_result.quality_gate_passed = True
                        except Exception:
                            pass

                results[stage_name] = stage_result
                logger.info(f"✅ {stage_name} completed "
                            f"(gate: {'PASS' if stage_result.quality_gate_passed else 'WARN'})")

            except Exception as e:
                logger.error(f"❌ {stage_name} FAILED: {e}")
                results[stage_name] = StageResult(
                    stage_name=stage_name,
                    status="failed",
                    output_summary=str(e),
                )
                # Continue to next stage (best-effort)
                if stage_name in ("data_collection",):
                    # Critical stages — cannot continue
                    logger.error("💀 Critical stage failed. Pipeline aborted.")
                    break

        # Build final result
        total_duration = time.time() - self.pipeline_start
        total_llm_calls = self.llm.call_count
        total_tokens = self.llm.total_tokens

        pipeline_result = PipelineResult(
            stages=results,
            total_duration_seconds=round(total_duration, 2),
            total_llm_calls=total_llm_calls,
            total_tokens=total_tokens,
            total_cost_usd=0.0,  # Cost tracking is provider-dependent
            insight_report=self.memory.retrieve("insight_report"),
            completed_at=datetime.now(timezone.utc).isoformat(),
        )

        # Save pipeline result
        self._save_pipeline_result(pipeline_result)

        # Print summary
        self._print_summary(pipeline_result)

        return pipeline_result

    def _run_with_retry(self, agent: BaseAgent, stage_name: str,
                         max_retries: int = 3) -> Any:
        """Run an agent with retry logic."""
        last_error = None

        for attempt in range(max_retries):
            try:
                return agent.run()
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for "
                               f"{stage_name}: {e}")

                if attempt < max_retries - 1:
                    # Create a fresh agent for retry
                    agent = type(agent)(
                        name=stage_name,
                        memory=self.memory,
                        llm=self.llm,
                    )
                    time.sleep(1)  # Brief pause before retry

        raise AgentException(
            f"Stage {stage_name} failed after {max_retries} attempts: {last_error}"
        )

    def _save_pipeline_result(self, result: PipelineResult) -> None:
        """Save the pipeline result to disk."""
        output_path = os.path.join(self.config.output_dir, "pipeline_result.json")

        # Convert to serializable dict
        result_dict = {
            "completed_at": result.completed_at,
            "total_duration_seconds": result.total_duration_seconds,
            "total_llm_calls": result.total_llm_calls,
            "total_tokens": result.total_tokens,
            "stages": {},
        }

        for name, stage in result.stages.items():
            result_dict["stages"][name] = {
                "status": stage.status,
                "quality_gate_passed": stage.quality_gate_passed,
                "output_summary": stage.output_summary[:200],
            }
            if stage.trace:
                result_dict["stages"][name]["trace"] = {
                    "duration_seconds": stage.trace.duration_seconds,
                    "llm_calls": stage.trace.llm_calls,
                    "reasoning_steps_count": len(stage.trace.reasoning_steps),
                }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        # Also save cleaning log
        cleaning_log = self.memory.retrieve("cleaning_log")
        if cleaning_log:
            log_path = os.path.join(self.config.output_dir, "cleaning_log.json")
            with open(log_path, "w") as f:
                json.dump(cleaning_log, f, indent=2, default=str)

        # Save EDA report
        eda_report = self.memory.retrieve("eda_report")
        if eda_report:
            eda_path = os.path.join(self.config.output_dir, "eda_report.json")
            try:
                with open(eda_path, "w") as f:
                    json.dump(eda_report.model_dump(), f, indent=2, default=str)
            except Exception:
                pass

        logger.info(f"💾 Pipeline result saved to {output_path}")

    def _print_summary(self, result: PipelineResult) -> None:
        """Print a pipeline execution summary."""
        logger.info("\n" + "=" * 70)
        logger.info("  📊 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 70)
        logger.info(f"  Duration:   {result.total_duration_seconds:.1f}s")
        logger.info(f"  LLM Calls:  {result.total_llm_calls}")
        logger.info(f"  Tokens:     {result.total_tokens:,}")
        logger.info("")

        for name, stage in result.stages.items():
            icon = "✅" if stage.status == "success" else "❌"
            gate = "PASS" if stage.quality_gate_passed else "WARN"
            duration = stage.trace.duration_seconds if stage.trace else 0
            logger.info(f"  {icon} {name:30s} {stage.status:10s} "
                        f"gate={gate:5s} {duration:6.1f}s")

        logger.info("")

        if result.insight_report and result.insight_report.executive_summary:
            logger.info("  📝 Executive Summary:")
            summary = result.insight_report.executive_summary
            for line in summary.split("\n")[:5]:
                logger.info(f"     {line.strip()}")

        logger.info("=" * 70)
