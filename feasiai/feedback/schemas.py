"""
Pydantic schemas for reviewer feedback records.

Design decisions:
1. REVIEW DECISION ENUM: Explicit enum prevents free-text confusion.
   'approved' means the reviewer endorsed the AI's (or overridden) label
   and the draft response is ready to include in a proposal.
2. OPTIONAL OVERRIDES: If the reviewer accepts the AI label, ai_label
   equals final_label. The diff is what matters for retraining signals.
3. OUTCOME FIELD: Left blank at review time; filled in later when the
   proposal outcome (won/lost/delivered) is known. This is the write-back
   trigger for the memory store.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ReviewDecision(str, Enum):
    """The reviewer's final disposition for a requirement's classification."""

    APPROVED = "approved"            # Accept AI label; include in proposal
    OVERRIDDEN = "overridden"        # Reviewer changed the label
    FLAGGED = "flagged"              # Needs further discussion; not proposal-ready
    REJECTED = "rejected"           # Requirement cannot be addressed; flag in proposal


class FeedbackRecord(BaseModel):
    """
    One reviewer decision on one requirement classification.

    Stored append-only in SQLite. Used both for quality monitoring
    and as the trigger for memory write-back after delivery outcomes.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Linking fields
    report_id: str = Field(..., description="FeasibilityReport.id")
    requirement_id: str = Field(..., description="Requirement.id")
    requirement_text: str = Field(..., description="Snapshot of requirement text")

    # Classification outcome
    ai_label: str = Field(..., description="Label the system produced")
    ai_confidence: float = Field(..., description="Confidence the system reported")
    final_label: str = Field(..., description="Label after reviewer decision (may equal ai_label)")
    decision: ReviewDecision = Field(..., description="Reviewer's disposition")

    # Human input
    reviewer_notes: str = Field("", description="Reviewer's free-text notes")
    draft_response_approved: bool = Field(
        False, description="True if the draft response is ready for the proposal"
    )

    # Proposal outcome (filled in later via memory write-back)
    proposal_outcome: Optional[str] = Field(
        None,
        description="Delivery outcome: 'won' | 'lost' | 'delivered' | 'cancelled' | None",
    )
    actual_effort_days: Optional[float] = Field(
        None, description="Actual engineering effort if requirement was delivered"
    )
    outcome_notes: Optional[str] = Field(
        None, description="Post-delivery notes for memory write-back"
    )

    model_config = {"frozen": True}

