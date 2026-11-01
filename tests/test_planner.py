"""
Tests for the Planner Agent (Phase 2).

Tests cover:
- Full agent loop (think → act → observe → classify)
- Early stopping when initial evidence is strong
- Step budget enforcement
- Fallback to single-shot on error
- Evidence deduplication across tool calls
- Agent trace recording
- Prompt security rules
- Tool selection
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from feasiai.agent.planner import PlannerAgent
from feasiai.agent.tools import AgentToolkit
from feasiai.agent.prompts import (
    build_system_prompt,
    format_initial_context,
    format_tool_result,
    format_tool_descriptions,
)
from feasiai.schemas import (
    AgentAction,
    AgentStep,
    AgentTrace,
    Classification,
    Evidence,
    EvidenceSource,
    FeasibilityLabel,
    Requirement,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_requirement():
    return Requirement(
        id="req-001",
        text="The system must support SSO via SAML 2.0",
        original_section="Authentication",
        is_ambiguous=False,
    )


@pytest.fixture
def ambiguous_requirement():
    return Requirement(
        id="req-002",
        text="The system should be user-friendly",
        original_section="General",
        is_ambiguous=True,
        ambiguity_reason="No measurable criteria for user-friendliness",
    )


@pytest.fixture
def sample_evidence():
    return [
        Evidence(
            id="ev-1",
            content="SSO via SAML 2.0 and OIDC supported.",
            source=EvidenceSource.MEMORY,
            source_document="product_capabilities.md",
            chunk_id="chunk-abc",
            relevance_score=0.85,
        ),
        Evidence(
            id="ev-2",
            content="MFA via TOTP and WebAuthn/FIDO2.",
            source=EvidenceSource.MEMORY,
            source_document="product_capabilities.md",
            chunk_id="chunk-def",
            relevance_score=0.6,
        ),
    ]


@pytest.fixture
def gap_evidence():
    return [
        Evidence(
            id="ev-gap-1",
            content="No Shopify, WooCommerce, or Magento integration.",
            source=EvidenceSource.MEMORY,
            source_document="known_gaps.md",
            chunk_id="chunk-gap-1",
            relevance_score=0.9,
        ),
    ]


@pytest.fixture
def mock_memory_store(sample_evidence, gap_evidence):
    store = MagicMock()
    store.retrieve.return_value = sample_evidence
    return store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.call_count = 0
    llm.call_log = []
    llm.reset_budget = MagicMock()
    return llm


# ─── Agent Loop Tests ──────────────────────────────────────────────────────


class TestAgentBasicFlow:
    def test_agent_tool_call_then_classify(
        self, mock_llm, mock_memory_store, sample_requirement, sample_evidence
    ):
        """Agent calls a tool, gets evidence, then classifies."""
        # Step 1: Agent decides to call check_gaps
        # Step 2: Agent classifies with accumulated evidence
        mock_llm.chat_json.side_effect = [
            {
                "action": "tool_call",
                "thought": "Let me check for known gaps around SAML SSO",
                "tool": "check_gaps",
                "query": "SSO SAML 2.0 gaps limitations",
            },
            {
                "action": "classify",
                "thought": "No gaps found for SSO. Classifying as Available.",
                "label": "Available",
                "confidence": 0.92,
                "reasoning": "Product capabilities confirm SAML 2.0 SSO.",
                "what_would_change": "Custom attribute mapping requirements",
                "draft_response": "We support SSO via SAML 2.0.",
            },
        ]

        agent = PlannerAgent(mock_llm, mock_memory_store, max_steps=3)
        result = agent.assess(sample_requirement, sample_evidence)

        assert result.label == FeasibilityLabel.AVAILABLE
        assert result.confidence == 0.92
        assert result.agent_trace is not None
        assert result.agent_trace.total_tool_calls == 1
        assert result.agent_trace.fallback_used is False
        assert len(result.agent_trace.steps) == 2  # tool call + classify

    def test_agent_multiple_tool_calls(
        self, mock_llm, mock_memory_store, sample_requirement, sample_evidence
    ):
        """Agent makes multiple tool calls before classifying."""
        mock_llm.chat_json.side_effect = [
            {
                "action": "tool_call",
                "thought": "Check gaps first",
                "tool": "check_gaps",
                "query": "SSO SAML limitations",
            },
            {
                "action": "tool_call",
                "thought": "Check past proposals",
                "tool": "search_past_proposals",
                "query": "SSO SAML proposal",
            },
            {
                "action": "classify",
                "thought": "Full picture. Classifying.",
                "label": "Available",
                "confidence": 0.95,
                "reasoning": "Strong evidence from multiple sources.",
                "what_would_change": "None anticipated",
                "draft_response": "Fully supported.",
            },
        ]

        agent = PlannerAgent(mock_llm, mock_memory_store, max_steps=3)
        result = agent.assess(sample_requirement, sample_evidence)

        assert result.agent_trace.total_tool_calls == 2
        assert len(result.agent_trace.steps) == 3  # 2 tool calls + classify


class TestAgentEarlyStopping:
    def test_early_stop_on_strong_initial_evidence(
        self, mock_llm, mock_memory_store, sample_requirement, sample_evidence
    ):
        """Agent classifies immediately when initial evidence is strong."""
        mock_llm.chat_json.side_effect = [
            {
                "action": "classify",
                "thought": "Initial evidence clearly confirms SAML 2.0 SSO.",
                "label": "Available",
                "confidence": 0.93,
                "reasoning": "Product docs confirm SSO via SAML 2.0.",
                "what_would_change": "Custom attribute mapping",
                "draft_response": "We support SSO via SAML 2.0.",
            },
        ]

        agent = PlannerAgent(
            mock_llm, mock_memory_store,
            max_steps=3, confidence_threshold=0.85,
        )
        result = agent.assess(sample_requirement, sample_evidence)

        assert result.label == FeasibilityLabel.AVAILABLE
        assert result.agent_trace.early_stopped is True
        assert result.agent_trace.total_tool_calls == 0
        assert result.agent_trace.total_llm_calls == 1


class TestAgentStepBudget:
    def test_budget_forces_classification(
        self, mock_llm, mock_memory_store, sample_requirement, sample_evidence
    ):
        """When step budget is exhausted, agent is forced to classify."""
        # Agent keeps wanting to call tools but hits the limit
        mock_llm.chat_json.side_effect = [
            {
                "action": "tool_call",
                "thought": "Search memory",
                "tool": "search_memory",
                "query": "SSO SAML",
            },
            # Budget exhausted (max_steps=1), agent tries another tool call
            {
                "action": "tool_call",
                "thought": "Need more info",
                "tool": "check_gaps",
                "query": "SSO gaps",
            },
            # Forced classification
            {
                "action": "classify",
                "thought": "Classifying with available evidence.",
                "label": "Available",
                "confidence": 0.8,
                "reasoning": "Based on available evidence.",
                "what_would_change": "More evidence",
                "draft_response": "Supported.",
            },
        ]

        agent = PlannerAgent(mock_llm, mock_memory_store, max_steps=1)
        result = agent.assess(sample_requirement, sample_evidence)

        assert result.label == FeasibilityLabel.AVAILABLE
        # Agent was forced to classify after 1 tool call
        assert result.agent_trace.total_tool_calls == 1


class TestAgentFallback:
    def test_fallback_on_json_error(
        self, mock_llm, mock_memory_store, sample_requirement, sample_evidence
    ):
        """Falls back to single-shot when agent loop throws."""
        # Agent loop fails
        mock_llm.chat_json.side_effect = [
            ValueError("Unparseable response"),
            # Fallback single-shot classification
            {
                "label": "Available",
                "confidence": 0.85,
                "reasoning": "SSO supported.",
                "what_would_change": "Custom mapping",
                "draft_response": "We support SSO.",
            },
        ]

        agent = PlannerAgent(mock_llm, mock_memory_store, max_steps=3)
        result = agent.assess(sample_requirement, sample_evidence)

        assert result.agent_trace is not None
        assert result.agent_trace.fallback_used is True

    def test_fallback_on_exception(
        self, mock_llm, mock_memory_store, sample_requirement, sample_evidence
    ):
        """Falls back on any unhandled exception."""
        mock_llm.chat_json.side_effect = [
            RuntimeError("Unexpected error"),
            # Fallback
            {
                "label": "Unsupported / Unclear",
                "confidence": 0.5,
                "reasoning": "Unable to assess.",
                "what_would_change": "Retry",
                "draft_response": "Further investigation needed.",
            },
        ]

        agent = PlannerAgent(mock_llm, mock_memory_store, max_steps=3)
        result = agent.assess(sample_requirement, sample_evidence)

        assert result.agent_trace.fallback_used is True


class TestAgentEvidenceDedup:
    def test_dedup_across_tool_calls(
        self, mock_llm, sample_requirement, sample_evidence
    ):
        """Evidence is deduplicated across multiple tool calls."""
        # Memory store returns overlapping results
        store = MagicMock()
        store.retrieve.return_value = sample_evidence  # Same evidence every time

        mock_llm.chat_json.side_effect = [
            {
                "action": "tool_call",
                "thought": "Search memory",
                "tool": "search_memory",
                "query": "SSO SAML",
            },
            {
                "action": "tool_call",
                "thought": "Search again differently",
                "tool": "search_memory",
                "query": "single sign-on SAML 2.0",
            },
            {
                "action": "classify",
                "thought": "Classifying.",
                "label": "Available",
                "confidence": 0.9,
                "reasoning": "Confirmed.",
                "what_would_change": "Nothing",
                "draft_response": "Supported.",
            },
        ]

        agent = PlannerAgent(mock_llm, store, max_steps=3)
        result = agent.assess(sample_requirement, sample_evidence)

        # Evidence should be deduplicated — initial 2 + 0 new (all dupes)
        # The initial evidence is seeded, so tool calls return only new ones
        assert result.agent_trace.total_tool_calls == 2


class TestAgentTraceRecording:
    def test_trace_attached_to_classification(
        self, mock_llm, mock_memory_store, sample_requirement, sample_evidence
    ):
        """Full agent trace is recorded and attached."""
        mock_llm.chat_json.side_effect = [
            {
                "action": "tool_call",
                "thought": "Checking gaps",
                "tool": "check_gaps",
                "query": "SSO gaps",
            },
            {
                "action": "classify",
                "thought": "All clear.",
                "label": "Available",
                "confidence": 0.9,
                "reasoning": "Supported.",
                "what_would_change": "Nothing",
                "draft_response": "Yes.",
            },
        ]

        agent = PlannerAgent(mock_llm, mock_memory_store, max_steps=3)
        result = agent.assess(sample_requirement, sample_evidence)

        trace = result.agent_trace
        assert trace is not None
        assert trace.requirement_id == "req-001"
        assert len(trace.steps) == 2
        assert trace.steps[0].action is not None
        assert trace.steps[0].action.tool == "check_gaps"
        assert trace.steps[0].action.query == "SSO gaps"
        assert trace.steps[1].action is None  # classify step
        assert trace.total_llm_calls >= 2


# ─── Tool Tests ────────────────────────────────────────────────────────────


class TestAgentToolkit:
    def test_search_memory(self, mock_memory_store, sample_evidence):
        toolkit = AgentToolkit(mock_memory_store)
        results = toolkit.execute("search_memory", "SSO SAML")
        assert len(results) == len(sample_evidence)

    def test_check_gaps_boosts_gap_docs(self, sample_evidence, gap_evidence):
        """check_gaps returns gap documents first."""
        store = MagicMock()
        # Return mixed evidence — gap evidence has lower rank in raw results
        store.retrieve.return_value = sample_evidence + gap_evidence

        toolkit = AgentToolkit(store, top_k=5)
        results = toolkit.execute("check_gaps", "e-commerce Shopify")

        # Gap evidence should come first
        gap_sources = [r.source_document for r in results if "gap" in (r.source_document or "").lower()]
        assert len(gap_sources) > 0

    def test_unknown_tool_raises(self, mock_memory_store):
        toolkit = AgentToolkit(mock_memory_store)
        with pytest.raises(ValueError, match="Unknown tool"):
            toolkit.execute("nonexistent_tool", "query")

    def test_deduplication(self, mock_memory_store, sample_evidence):
        toolkit = AgentToolkit(mock_memory_store)

        # First call — all evidence is new
        results1 = toolkit.execute("search_memory", "SSO")
        assert len(results1) == len(sample_evidence)

        # Second call — same evidence, should be deduplicated
        results2 = toolkit.execute("search_memory", "SSO again")
        assert len(results2) == 0

        # All evidence is tracked
        assert len(toolkit.all_evidence) == len(sample_evidence)

    def test_reset(self, mock_memory_store, sample_evidence):
        toolkit = AgentToolkit(mock_memory_store)
        toolkit.execute("search_memory", "SSO")
        assert len(toolkit.all_evidence) > 0

        toolkit.reset()
        assert len(toolkit.all_evidence) == 0

        # Can retrieve same evidence again after reset
        results = toolkit.execute("search_memory", "SSO")
        assert len(results) == len(sample_evidence)


# ─── Prompt Tests ──────────────────────────────────────────────────────────


class TestPrompts:
    def test_system_prompt_has_security_rules(self):
        prompt = build_system_prompt()
        assert "SECURITY" in prompt
        assert "IGNORE" in prompt
        assert "XML tags" in prompt

    def test_system_prompt_has_tools(self):
        prompt = build_system_prompt()
        assert "search_memory" in prompt
        assert "check_gaps" in prompt
        assert "search_past_proposals" in prompt

    def test_system_prompt_has_labels(self):
        prompt = build_system_prompt()
        assert "Available" in prompt
        assert "Configurable" in prompt
        assert "Requires Development" in prompt
        assert "Unsupported / Unclear" in prompt

    def test_format_initial_context_basic(self, sample_requirement, sample_evidence):
        msg = format_initial_context(sample_requirement, sample_evidence)
        assert "<requirement>" in msg
        assert "SAML 2.0" in msg
        assert "<evidence>" in msg

    def test_format_initial_context_ambiguous(self, ambiguous_requirement):
        msg = format_initial_context(ambiguous_requirement, [])
        assert "AMBIGUOUS" in msg
        assert "No measurable criteria" in msg

    def test_format_initial_context_no_evidence(self, sample_requirement):
        msg = format_initial_context(sample_requirement, [])
        assert "No initial evidence" in msg

    def test_format_tool_result_with_evidence(self, sample_evidence):
        msg = format_tool_result("check_gaps", "SSO gaps", sample_evidence)
        assert "check_gaps" in msg
        assert "2 result(s)" in msg
        assert "<evidence>" in msg

    def test_format_tool_result_empty(self):
        msg = format_tool_result("search_memory", "nothing", [])
        assert "NO results" in msg

    def test_tool_descriptions(self):
        desc = format_tool_descriptions()
        assert "search_memory" in desc
        assert "check_gaps" in desc
        assert "search_past_proposals" in desc


# ─── Schema Tests ──────────────────────────────────────────────────────────


class TestAgentSchemas:
    def test_agent_action_frozen(self):
        action = AgentAction(tool="search_memory", query="test", result_count=3)
        with pytest.raises(Exception):
            action.tool = "other"

    def test_agent_step_frozen(self):
        step = AgentStep(step_number=1, thought="thinking", observation_summary="done")
        with pytest.raises(Exception):
            step.thought = "other"

    def test_agent_trace_creation(self):
        trace = AgentTrace(
            requirement_id="req-1",
            steps=[
                AgentStep(
                    step_number=1,
                    thought="Checking gaps",
                    action=AgentAction(tool="check_gaps", query="SSO", result_count=2),
                    observation_summary="2 chunks found",
                ),
                AgentStep(
                    step_number=2,
                    thought="Classifying",
                    action=None,
                    observation_summary="Final classification",
                ),
            ],
            total_tool_calls=1,
            total_llm_calls=2,
            early_stopped=False,
            fallback_used=False,
        )
        assert len(trace.steps) == 2
        assert trace.total_tool_calls == 1

    def test_classification_with_trace(self, sample_evidence):
        trace = AgentTrace(
            requirement_id="req-1",
            steps=[],
            total_tool_calls=0,
            total_llm_calls=1,
            early_stopped=True,
            fallback_used=False,
        )
        classification = Classification(
            requirement_id="req-1",
            label=FeasibilityLabel.AVAILABLE,
            confidence=0.9,
            supporting_evidence=sample_evidence,
            reasoning="Confirmed",
            what_would_change="Nothing",
            draft_response="Supported",
            agent_trace=trace,
        )
        assert classification.agent_trace is not None
        assert classification.agent_trace.early_stopped is True

    def test_classification_without_trace(self):
        """Backward compatible — agent_trace is optional."""
        classification = Classification(
            requirement_id="req-1",
            label=FeasibilityLabel.AVAILABLE,
            confidence=0.9,
            reasoning="OK",
        )
        assert classification.agent_trace is None

