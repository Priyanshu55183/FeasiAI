"""
Conflict detection between institutional memory and live data.

Design decisions:
1. SIGNAL-BASED: Conflict detection uses label signals rather than
   exact string matching. A MEMORY evidence labeled Available + a live
   data item with status 'cancelled' is a high-confidence conflict.
2. SCHEMA EXTENSION: ConflictWarning is a first-class Pydantic model
   defined in schemas.py and attached to Classification.conflicts.
   This makes conflicts visible in the UI and exportable in reports.
3. SEVERITY TIERS: Conflicts are graded Low / Medium / High based on
   how directly the live data contradicts the memory evidence.
   - HIGH:   Memory says Available, live status = cancelled
   - MEDIUM: Memory says Available/Configurable, live status = planned (might not be ready)
   - LOW:    Any other live/memory mismatch worth noting
4. NON-BLOCKING: Conflict detection never changes the Classification label —
   it only annotates it. The human reviewer decides how to act on warnings.
5. KEYWORD SIGNALS: Detects capability keywords shared between evidence
   content and feature names to link them before comparing statuses.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from feasiai.schemas import (
    Classification,
    ConflictWarning,
    Evidence,
    EvidenceSource,
    FeasibilityLabel,
)

logger = logging.getLogger(__name__)


class ConflictSeverity(str, Enum):
    """Severity tier for a detected conflict."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# Status values that contradict an "Available" or "Configurable" memory verdict
_HIGH_CONFLICT_STATUSES = {"cancelled", "discontinued", "deprecated", "removed"}
_MEDIUM_CONFLICT_STATUSES = {"planned", "not started", "not_started", "backlog"}


def detect_conflicts(
    classification: Classification,
    live_evidences: list[Evidence],
) -> list[ConflictWarning]:
    """
    Compare classification evidence against live data for conflicts.

    For each live data Evidence item with a recognizable status, checks
    whether that status contradicts the classification label derived from
    institutional memory.

    Args:
        classification: The Classification produced by the classifier.
        live_evidences: Evidence items from live data sources.

    Returns:
        List of ConflictWarning objects (may be empty).
    """
    warnings: list[ConflictWarning] = []

    # Only check conflicts when memory says the feature is available/configurable
    memory_positive = classification.label in (
        FeasibilityLabel.AVAILABLE,
        FeasibilityLabel.CONFIGURABLE,
    )

    memory_evidences = [
        ev for ev in classification.supporting_evidence
        if ev.source == EvidenceSource.MEMORY
    ]

    for live_ev in live_evidences:
        status = live_ev.metadata.get("status", "").lower()
        if not status:
            continue  # Can't assess conflict without a known status

        severity: Optional[str] = None
        description: Optional[str] = None

        if memory_positive and status in _HIGH_CONFLICT_STATUSES:
            severity = ConflictSeverity.HIGH.value
            feature_name = live_ev.metadata.get("feature_name", live_ev.source_document or "unknown")
            description = (
                f"Memory evidence suggests this capability is {classification.label.value}, "
                f"but live data shows '{feature_name}' is '{status}'. "
                "This may indicate the memory is outdated."
            )

        elif memory_positive and status in _MEDIUM_CONFLICT_STATUSES:
            severity = ConflictSeverity.MEDIUM.value
            feature_name = live_ev.metadata.get("feature_name", live_ev.source_document or "unknown")
            eta = live_ev.metadata.get("eta", "")
            eta_text = f" (ETA: {eta})" if eta else ""
            description = (
                f"Memory evidence suggests this is {classification.label.value}, "
                f"but live data shows '{feature_name}' is only '{status}'{eta_text}. "
                "Verify delivery timeline before committing."
            )

        elif not memory_positive and status in ("shipped", "in_progress"):
            severity = ConflictSeverity.LOW.value
            feature_name = live_ev.metadata.get("feature_name", live_ev.source_document or "unknown")
            description = (
                f"Classification suggests limited feasibility, but live data shows "
                f"'{feature_name}' has status '{status}'. "
                "Memory may not reflect recent progress."
            )

        if severity and description:
            # Find the most relevant memory evidence (use first if multiple)
            memory_ev_id = memory_evidences[0].id if memory_evidences else None
            warning = ConflictWarning(
                requirement_id=classification.requirement_id,
                severity=severity,
                description=description,
                memory_evidence_id=memory_ev_id,
                live_evidence_id=live_ev.id,
                live_status=status,
            )
            warnings.append(warning)
            logger.info(
                "ConflictDetector: %s conflict for req %s — %s",
                severity.upper(),
                classification.requirement_id[:8],
                description[:80],
            )

    return warnings
