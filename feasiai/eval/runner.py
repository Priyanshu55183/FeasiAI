"""
Evaluation runner: runs the pipeline against a labeled dataset.

Design decisions:
1. MOCK-FRIENDLY: The runner accepts an injected LLMClient and MemoryStore,
   so tests can run without API keys by injecting mocks.

2. PER-SAMPLE LLM CALL COUNTING: We reset the LLM budget before each sample
   so we can attribute exact LLM call costs per requirement.

3. GRACEFUL ERRORS: A failed classification is recorded with is_correct=False
   and the error message, rather than aborting the whole eval run.

4. AGENT INTROSPECTION: If agent mode is used and the classification has an
   AgentTrace, we extract tool call counts, early-stop flag, and fallback flag.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from feasiai import config
from feasiai.agent.planner import PlannerAgent
from feasiai.classification.classifier import classify_requirement
from feasiai.eval.schemas import EvalResult, EvalSample
from feasiai.llm.client import LLMClient
from feasiai.retrieval.memory_store import MemoryStore
from feasiai.schemas import FeasibilityLabel, Requirement

logger = logging.getLogger(__name__)


class EvalRunner:
    """
    Runs the pipeline against a labeled dataset of EvalSamples.

    Supports two modes:
    - 'agent': Uses the PlannerAgent (Phase 2)
    - 'single_shot': Uses the classifier directly (Phase 1 baseline)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_store: MemoryStore,
        use_agent: bool = True,
        agent_max_steps: Optional[int] = None,
        agent_confidence_threshold: Optional[float] = None,
    ):
        self._llm = llm_client
        self._memory = memory_store
        self._use_agent = use_agent

        self._agent: Optional[PlannerAgent] = None
        if use_agent:
            self._agent = PlannerAgent(
                llm_client=llm_client,
                memory_store=memory_store,
                max_steps=agent_max_steps if agent_max_steps is not None else config.AGENT_MAX_STEPS,
                confidence_threshold=(
                    agent_confidence_threshold
                    if agent_confidence_threshold is not None
                    else config.AGENT_CONFIDENCE_THRESHOLD
                ),
            )

    def run(self, samples: list[EvalSample]) -> list[EvalResult]:
        """
        Evaluate all samples and return per-sample results.

        Args:
            samples: Ground-truth labeled requirements to evaluate.

        Returns:
            List of EvalResult, one per sample.
        """
        results = []
        mode = "agent" if self._use_agent else "single_shot"
        logger.info(
            "Starting eval run: %d samples, mode=%s", len(samples), mode
        )

        for i, sample in enumerate(samples, 1):
            logger.info(
                "Evaluating [%d/%d]: %.60s...", i, len(samples), sample.requirement_text
            )
            result = self._evaluate_sample(sample)
            results.append(result)

            status = "✓" if result.is_correct else "✗"
            logger.info(
                "  %s Expected=%s | Got=%s | conf=%.0f%% | llm_calls=%d",
                status,
                result.expected_label.value,
                result.predicted_label.value,
                result.confidence * 100,
                result.llm_calls_used,
            )

        correct = sum(1 for r in results if r.is_correct)
        logger.info(
            "Eval complete: %d/%d correct (%.0f%%)",
            correct, len(results), correct / len(results) * 100 if results else 0,
        )
        return results

    def _evaluate_sample(self, sample: EvalSample) -> EvalResult:
        """Evaluate a single sample, returning an EvalResult."""
        # Build a Requirement from the sample
        req = Requirement(
            id=sample.id,
            text=sample.requirement_text,
            original_section=sample.section,
            is_ambiguous=sample.is_ambiguous,
        )

        # Reset LLM call counter so we can count per-sample
        calls_before = self._llm.call_count

        start = time.perf_counter()
        error_msg = None
        predicted_label = FeasibilityLabel.UNSUPPORTED_UNCLEAR
        confidence = 0.0
        agent_tool_calls = 0
        agent_early_stopped = False
        fallback_used = False

        try:
            # Retrieve initial evidence
            evidences = self._memory.retrieve(req.text)

            if self._agent:
                classification = self._agent.assess(req, evidences)
                # Extract agent trace stats
                if classification.agent_trace:
                    trace = classification.agent_trace
                    agent_tool_calls = trace.total_tool_calls
                    agent_early_stopped = trace.early_stopped
                    fallback_used = trace.fallback_used
            else:
                classification = classify_requirement(req, evidences, self._llm)

            predicted_label = classification.label
            confidence = classification.confidence
            # Treat a confidence=0.0 failure classification as an error record
            if classification.confidence == 0.0 and "failed" in classification.reasoning.lower():
                error_msg = classification.reasoning

        except Exception as e:
            error_msg = str(e)
            logger.error("Eval sample %s failed: %s", sample.id[:8], e)

        elapsed = time.perf_counter() - start
        llm_calls_used = self._llm.call_count - calls_before

        return EvalResult(
            sample_id=sample.id,
            requirement_text=sample.requirement_text,
            expected_label=sample.expected_label,
            predicted_label=predicted_label,
            confidence=confidence,
            is_correct=(predicted_label == sample.expected_label),
            llm_calls_used=llm_calls_used,
            latency_seconds=round(elapsed, 3),
            used_agent=self._use_agent,
            agent_tool_calls=agent_tool_calls,
            agent_early_stopped=agent_early_stopped,
            fallback_used=fallback_used,
            error=error_msg,
        )
