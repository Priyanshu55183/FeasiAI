"""
Planner agent: multi-step feasibility assessment with tool use.

Design decisions:
1. ReAct LOOP: The agent follows a Think → Act → Observe cycle.
   Each iteration either calls a tool or produces a final classification.
   This is more effective than chain-of-thought alone because the agent
   can gather targeted evidence based on intermediate reasoning.

2. EARLY STOPPING: If the initial evidence is strong enough (confidence
   above threshold), the agent classifies immediately without tool calls.
   This saves LLM budget for easy requirements.

3. STEP BUDGET: Each requirement gets at most AGENT_MAX_STEPS tool calls.
   After exhausting the budget, the agent must classify with what it has.
   This prevents runaway cost on ambiguous requirements.

4. GRACEFUL FALLBACK: If the agent loop fails for any reason (JSON parse
   error, unexpected output, exception), we fall back to the Phase 1
   single-shot classifier. The user always gets a result.

5. CONVERSATION HISTORY: The agent's messages accumulate in a conversation
   so each step has full context of prior reasoning and observations.
   This enables coherent multi-step reasoning.

6. FULL TRACING: Every step is recorded in an AgentTrace object attached
   to the Classification. This provides complete observability into
   how the agent reached its conclusion.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from feasiai import config
from feasiai.agent.prompts import (
    build_system_prompt,
    format_initial_context,
    format_tool_result,
)
from feasiai.agent.tools import AgentToolkit
from feasiai.classification.classifier import classify_requirement, _parse_label
from feasiai.llm.client import LLMClient
from feasiai.retrieval.memory_store import MemoryStore
from feasiai.schemas import (
    AgentAction,
    AgentStep,
    AgentTrace,
    Classification,
    Evidence,
    FeasibilityLabel,
    Requirement,
)

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Multi-step feasibility assessment agent.

    Uses a ReAct loop to iteratively gather evidence and classify
    requirements. Falls back to single-shot classification on failure.

    Usage:
        agent = PlannerAgent(llm_client, memory_store)
        classification = agent.assess(requirement, initial_evidence)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        memory_store: MemoryStore,
        max_steps: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
    ):
        self._llm = llm_client
        self._toolkit = AgentToolkit(memory_store)
        self._max_steps = max_steps if max_steps is not None else config.AGENT_MAX_STEPS
        self._confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else config.AGENT_CONFIDENCE_THRESHOLD
        )

    def assess(
        self,
        requirement: Requirement,
        initial_evidence: list[Evidence],
    ) -> Classification:
        """
        Assess a requirement using multi-step reasoning.

        Args:
            requirement: The requirement to assess.
            initial_evidence: Evidence from the initial retrieval pass.

        Returns:
            Classification with agent trace attached.
        """
        try:
            return self._agent_loop(requirement, initial_evidence)
        except Exception as e:
            logger.warning(
                "Agent loop failed for requirement %s, falling back to "
                "single-shot: %s",
                requirement.id[:8],
                e,
            )
            return self._fallback_classify(requirement, initial_evidence)

    def _agent_loop(
        self,
        requirement: Requirement,
        initial_evidence: list[Evidence],
    ) -> Classification:
        """
        Run the ReAct agent loop.

        Returns:
            Classification with full agent trace.
        """
        # Reset toolkit for this requirement
        self._toolkit.reset()

        # Seed the toolkit with initial evidence (for dedup tracking)
        for ev in initial_evidence:
            chunk_id = ev.chunk_id or ev.id
            self._toolkit._seen_chunk_ids.add(chunk_id)
            self._toolkit._all_evidence.append(ev)

        # Build conversation
        system_prompt = build_system_prompt()
        initial_msg = format_initial_context(requirement, initial_evidence)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_msg},
        ]

        steps: list[AgentStep] = []
        tool_call_count = 0
        llm_call_count = 0

        for step_num in range(1, self._max_steps + 2):
            # +1 for the final classify step after exhausting tool budget

            # Ask the agent for its next action
            response = self._llm.chat_json(
                messages,
                purpose=f"agent_step_{step_num}",
                temperature=0.1,
            )
            llm_call_count += 1

            action_type = response.get("action", "classify")

            if action_type == "tool_call" and tool_call_count < self._max_steps:
                # Agent wants to call a tool
                tool_name = response.get("tool", "search_memory")
                query = response.get("query", requirement.text)
                thought = response.get("thought", "")

                # Execute the tool
                try:
                    new_evidence = self._toolkit.execute(tool_name, query)
                except ValueError as e:
                    # Unknown tool — tell the agent and let it retry
                    logger.warning("Agent called unknown tool: %s", e)
                    new_evidence = []

                tool_call_count += 1

                # Record the step
                step = AgentStep(
                    step_number=step_num,
                    thought=thought,
                    action=AgentAction(
                        tool=tool_name,
                        query=query,
                        result_count=len(new_evidence),
                    ),
                    observation_summary=(
                        f"{len(new_evidence)} new evidence chunk(s) from {tool_name}"
                    ),
                )
                steps.append(step)

                # Format observation and add to conversation
                observation = format_tool_result(tool_name, query, new_evidence)
                messages.append({"role": "assistant", "content": json.dumps(response)})
                messages.append({"role": "user", "content": observation})

            elif action_type == "classify" or tool_call_count >= self._max_steps:
                # Agent is ready to classify (or budget exhausted)
                if action_type != "classify":
                    # Budget exhausted — force a classification
                    logger.info(
                        "Agent step budget exhausted (%d/%d), forcing classification",
                        tool_call_count,
                        self._max_steps,
                    )
                    # Ask agent to classify with what it has
                    messages.append(
                        {"role": "assistant", "content": json.dumps(response)}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have used all available tool calls. "
                                "You MUST now classify with the evidence you have. "
                                "Return a JSON with action='classify'."
                            ),
                        }
                    )
                    response = self._llm.chat_json(
                        messages,
                        purpose="agent_forced_classify",
                        temperature=0.1,
                    )
                    llm_call_count += 1

                # Record final step
                thought = response.get("thought", "")
                steps.append(
                    AgentStep(
                        step_number=step_num if action_type == "classify" else step_num + 1,
                        thought=thought,
                        action=None,
                        observation_summary="Final classification",
                    )
                )

                # Parse classification
                all_evidence = self._toolkit.all_evidence
                confidence = max(0.0, min(1.0, float(response.get("confidence", 0.0))))
                early_stopped = (
                    tool_call_count == 0 and confidence >= self._confidence_threshold
                )

                trace = AgentTrace(
                    requirement_id=requirement.id,
                    steps=steps,
                    total_tool_calls=tool_call_count,
                    total_llm_calls=llm_call_count,
                    early_stopped=early_stopped,
                    fallback_used=False,
                )

                return Classification(
                    requirement_id=requirement.id,
                    label=_parse_label(response.get("label", "Unsupported / Unclear")),
                    confidence=confidence,
                    supporting_evidence=all_evidence,
                    reasoning=response.get("reasoning", ""),
                    what_would_change=response.get("what_would_change", ""),
                    draft_response=response.get("draft_response", ""),
                    agent_trace=trace,
                )

            else:
                # Unexpected action — treat as classify attempt
                logger.warning(
                    "Unexpected agent action: %s. Treating as classification.",
                    action_type,
                )
                break

        # Should not reach here, but safety net
        return self._fallback_classify(requirement, initial_evidence)

    def _fallback_classify(
        self,
        requirement: Requirement,
        evidences: list[Evidence],
    ) -> Classification:
        """
        Fall back to Phase 1 single-shot classification.

        Attaches an AgentTrace marking fallback_used=True.
        """
        logger.info(
            "Falling back to single-shot for requirement %s",
            requirement.id[:8],
        )
        classification = classify_requirement(requirement, evidences, self._llm)

        # Wrap with a trace indicating fallback
        trace = AgentTrace(
            requirement_id=requirement.id,
            steps=[],
            total_tool_calls=0,
            total_llm_calls=1,
            early_stopped=False,
            fallback_used=True,
        )

        # Re-create classification with trace attached
        return Classification(
            requirement_id=classification.requirement_id,
            label=classification.label,
            confidence=classification.confidence,
            supporting_evidence=classification.supporting_evidence,
            reasoning=classification.reasoning,
            what_would_change=classification.what_would_change,
            draft_response=classification.draft_response,
            agent_trace=trace,
        )

