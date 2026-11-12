"""
Schemas for memory write-back records.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class WritebackSignalType(str, Enum):
    """The type of institutional knowledge being written back."""

    LESSON = "lesson"                     # Generalizable finding
    DO_NOT_CLAIM = "do_not_claim"         # Capability that is not real / overstated
    EFFORT_CALIBRATION = "effort_calibration"  # Actual engineering cost signal
    CAPABILITY_CONFIRMED = "capability_confirmed"  # Positive confirmation (was correct)


class WritebackRecord(BaseModel):
    """
    A single piece of institutional knowledge derived from a delivery outcome.

    Created from a FeedbackRecord once the proposal outcome is known.
    Written to a markdown file in data/memory/ and re-ingested.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Linking back to the original requirement
    source_feedback_id: str = Field(..., description="FeedbackRecord.id that triggered this")
    requirement_text: str = Field(..., description="The requirement this applies to")

    # Signal content
    signal_type: WritebackSignalType = Field(..., description="Type of knowledge")
    content: str = Field(..., description="The actual lesson/finding/signal (1-3 sentences)")
    capability_area: Optional[str] = Field(
        None, description="Category tag: e.g. 'SSO', 'compliance', 'integrations'"
    )

    # Delivery context
    proposal_outcome: str = Field(
        ..., description="won | lost | delivered | cancelled"
    )
    actual_effort_days: Optional[float] = Field(None)
    client_vertical: Optional[str] = Field(
        None, description="Industry vertical for filtering (e.g. 'healthcare', 'fintech')"
    )

    model_config = {"frozen": True}

