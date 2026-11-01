"""
Pydantic schemas for all inter-component data structures.

Design decisions:
- Every schema is immutable (frozen) by default for safety.
- UUIDs are generated automatically for traceability.
- Enums are used for closed sets (FeasibilityLabel) so we get
  validation + serialization for free.
- Evidence carries full provenance (source, chunk_id, timestamp)
  so every claim is auditable.
- Classification includes "what_would_change" to support
  uncertainty-aware reasoning and human review.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enums ──────────────────────────────────────────────────────────────────


class FeasibilityLabel(str, Enum):
    """
    Four-class feasibility taxonomy. 'UNSUPPORTED_UNCLEAR' is a first-class
    outcome — the system is rewarded for admitting ignorance.
    """

    AVAILABLE = "Available"
    CONFIGURABLE = "Configurable"
    REQUIRES_DEVELOPMENT = "Requires Development"
    UNSUPPORTED_UNCLEAR = "Unsupported / Unclear"


class EvidenceSource(str, Enum):
    """Where a piece of evidence originated."""

    MEMORY = "memory"           # Institutional knowledge (docs, past proposals)
    LIVE_DATA = "live_data"     # Operational reality (roadmap, tickets, etc.)
    LLM_REASONING = "llm_reasoning"  # LLM's own inference (lower trust)


# ─── Core Schemas ───────────────────────────────────────────────────────────


class Requirement(BaseModel):
    """
    A single atomic, evaluable requirement extracted from an RFP.

    'is_ambiguous' flags requirements where the extractor could not
    determine a clear, testable condition. These still flow through
    the pipeline but are treated with lower confidence.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The atomic requirement statement")
    original_section: Optional[str] = Field(
        None, description="Section/heading in the source document"
    )
    is_ambiguous: bool = Field(
        False,
        description="True if the requirement is vague or has multiple interpretations",
    )
    ambiguity_reason: Optional[str] = Field(
        None,
        description="Why this requirement was flagged as ambiguous",
    )
    source_document: Optional[str] = Field(
        None, description="Filename or identifier of the source RFP"
    )

    model_config = {"frozen": True}


class Evidence(BaseModel):
    """
    A single piece of evidence retrieved from any source.

    Carries full provenance so reviewers can audit every claim.
    'relevance_score' is the retrieval score (dense, sparse, or fused).
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str = Field(..., description="The evidence text chunk")
    source: EvidenceSource = Field(..., description="Origin system")
    source_document: Optional[str] = Field(
        None, description="Document filename or API endpoint"
    )
    chunk_id: Optional[str] = Field(
        None, description="Chunk identifier within the source"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this evidence was retrieved or last updated",
    )
    relevance_score: float = Field(
        0.0, description="Retrieval relevance score (higher = more relevant)"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata (category, date of original doc, etc.)",
    )

    model_config = {"frozen": True}


class Classification(BaseModel):
    """
    Feasibility classification for a single requirement.

    Key design choice: 'what_would_change' forces the LLM to articulate
    its uncertainty boundary. This is critical for human reviewers and
    for later confidence calibration evaluation.
    """

    requirement_id: str = Field(..., description="ID of the classified requirement")
    label: FeasibilityLabel = Field(..., description="Feasibility class")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calibrated confidence score (0.0–1.0)",
    )
    supporting_evidence: list[Evidence] = Field(
        default_factory=list,
        description="Evidence supporting this classification",
    )
    reasoning: str = Field(
        "", description="LLM's chain-of-thought reasoning for this label"
    )
    what_would_change: str = Field(
        "",
        description=(
            "What additional evidence or information would change "
            "this classification — supports uncertainty-aware review"
        ),
    )
    draft_response: str = Field(
        "",
        description="Draft client-facing response text grounded in evidence",
    )
    agent_trace: Optional["AgentTrace"] = Field(
        None,
        description="Full agent reasoning trace (None for single-shot Phase 1 mode)",
    )

    model_config = {"frozen": True}


# ─── Agent Schemas (Phase 2) ───────────────────────────────────────────────


class AgentAction(BaseModel):
    """A single tool call made by the planner agent."""

    tool: str = Field(..., description="Tool name: search_memory | check_gaps | search_past_proposals")
    query: str = Field(..., description="The query the agent formulated for the tool")
    result_count: int = Field(0, description="Number of evidence chunks returned")

    model_config = {"frozen": True}


class AgentStep(BaseModel):
    """One think → act → observe cycle in the agent loop."""

    step_number: int = Field(..., description="1-indexed step number")
    thought: str = Field("", description="Agent's reasoning about what to do next")
    action: Optional[AgentAction] = Field(
        None, description="Tool call (None if this is the final classify step)"
    )
    observation_summary: str = Field(
        "", description="Brief summary of what the tool returned"
    )

    model_config = {"frozen": True}


class AgentTrace(BaseModel):
    """
    Full trace of the planner agent's reasoning for one requirement.

    This provides full observability into multi-step reasoning:
    every thought, tool call, and observation is recorded.
    """

    requirement_id: str = Field(..., description="ID of the requirement being assessed")
    steps: list[AgentStep] = Field(default_factory=list)
    total_tool_calls: int = Field(0, description="Number of tool invocations")
    total_llm_calls: int = Field(0, description="Number of LLM calls (including reasoning)")
    early_stopped: bool = Field(
        False, description="True if confidence threshold triggered early stop"
    )
    fallback_used: bool = Field(
        False, description="True if fell back to single-shot classification"
    )

    model_config = {"frozen": True}


class ExtractionResult(BaseModel):
    """Output of the requirement extraction step."""

    requirements: list[Requirement] = Field(default_factory=list)
    source_document: Optional[str] = None
    raw_text_length: int = 0
    extraction_metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Output of a retrieval operation for one requirement."""

    requirement_id: str
    evidences: list[Evidence] = Field(default_factory=list)
    retrieval_metadata: dict = Field(default_factory=dict)


class FeasibilityReport(BaseModel):
    """
    The final structured report for an entire RFP.

    Contains all classifications plus aggregate statistics.
    This is what the human reviewer sees.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_document: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    requirements: list[Requirement] = Field(default_factory=list)
    classifications: list[Classification] = Field(default_factory=list)
    summary: dict = Field(
        default_factory=dict,
        description=(
            "Aggregate stats: counts per label, average confidence, "
            "number of ambiguous requirements, etc."
        ),
    )
    llm_call_count: int = Field(
        0, description="Total LLM calls used for this report"
    )
    total_latency_seconds: float = Field(
        0.0, description="Wall-clock time for the full pipeline"
    )

    model_config = {"frozen": True}


# ─── LLM Logging Schema ────────────────────────────────────────────────────


class LLMCallRecord(BaseModel):
    """
    Log entry for a single LLM call. Every call goes through the
    unified wrapper and produces one of these for observability.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    model: str = ""
    purpose: str = Field("", description="e.g. 'extraction', 'classification'")
    prompt_messages: list[dict] = Field(default_factory=list)
    raw_response: str = ""
    latency_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error: Optional[str] = None
