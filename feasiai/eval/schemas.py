"""
Evaluation schemas.

Design decisions:
- EvalSample is the ground-truth unit: one requirement + expected label.
- EvalResult captures one pipeline run against one sample.
- EvalReport aggregates all results for one ablation configuration.
- All schemas are frozen Pydantic models for consistency with the main codebase.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from feasiai.schemas import FeasibilityLabel


class EvalSample(BaseModel):
    """
    A single labeled ground-truth sample for evaluation.

    'expected_label' is the human-verified correct answer.
    'notes' explains the reasoning behind the label — useful for
    understanding disagreements between the pipeline and ground truth.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    requirement_text: str = Field(..., description="The requirement to classify")
    expected_label: FeasibilityLabel = Field(..., description="Ground-truth label")
    section: Optional[str] = Field(None, description="Source section/category")
    is_ambiguous: bool = Field(False, description="Whether this requirement is intentionally ambiguous")
    notes: str = Field("", description="Human rationale for the expected label")
    source: str = Field("", description="Which dataset/RFP this came from")

    model_config = {"frozen": True}


class EvalResult(BaseModel):
    """
    Result of running the pipeline on a single EvalSample.

    Tracks predicted label, confidence, whether it was correct,
    and resource usage (LLM calls, latency).
    """

    sample_id: str = Field(..., description="ID of the EvalSample")
    requirement_text: str = Field(..., description="The requirement text")
    expected_label: FeasibilityLabel
    predicted_label: FeasibilityLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_correct: bool = Field(..., description="predicted == expected")
    llm_calls_used: int = Field(0, description="LLM calls for this requirement")
    latency_seconds: float = Field(0.0)
    used_agent: bool = Field(False, description="True if agent mode was used")
    agent_tool_calls: int = Field(0, description="Tool calls by the agent (0 for single-shot)")
    agent_early_stopped: bool = Field(False)
    fallback_used: bool = Field(False, description="Agent fell back to single-shot")
    error: Optional[str] = Field(None, description="Error message if classification failed")

    model_config = {"frozen": True}


class EvalReport(BaseModel):
    """
    Aggregated evaluation report for one ablation run.

    Contains per-sample results and aggregate metrics.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: str = Field(..., description="'agent' or 'single_shot'")
    model_name: str = Field("", description="LLM model used")
    total_samples: int = 0
    results: list[EvalResult] = Field(default_factory=list)

    # Aggregate metrics (populated by metrics module)
    accuracy: float = 0.0
    macro_f1: float = 0.0
    ece: float = Field(0.0, description="Expected Calibration Error")
    avg_confidence: float = 0.0
    avg_llm_calls: float = 0.0
    avg_latency_seconds: float = 0.0
    total_llm_calls: int = 0
    total_latency_seconds: float = 0.0

    # Per-label breakdown
    per_label_precision: dict[str, float] = Field(default_factory=dict)
    per_label_recall: dict[str, float] = Field(default_factory=dict)
    per_label_f1: dict[str, float] = Field(default_factory=dict)
    per_label_support: dict[str, int] = Field(default_factory=dict)

    # Agent-specific stats (only relevant for agent mode)
    early_stop_rate: float = 0.0
    fallback_rate: float = 0.0
    avg_tool_calls: float = 0.0

    model_config = {"frozen": True}

