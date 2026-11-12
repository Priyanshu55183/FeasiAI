"""
Memory write-back service: persists delivery outcomes to institutional memory.

Design decisions:
1. MARKDOWN OUTPUT: Generates structured .md documents that the MemoryStore
   can ingest. Human-readable, git-diffable, and auditable.
2. ONE FILE PER WRITEBACK SESSION: Groups all records from one write-back
   call into a single timestamped file so they can be reviewed together.
3. SIGNAL-AWARE FORMATTING: 'do_not_claim' records get a prominent warning
   header so they are easy to find in memory doc reviews.
4. CATEGORY METADATA: Each chunk includes a 'category' field in its markdown
   front-matter. MemoryStore's metadata filters can then preferentially
   surface do_not_claim records during classification — matching the
   existing check_gaps tool pattern.
5. RE-INGEST IS CALLER'S RESPONSIBILITY: The writer produces the file;
   the caller (UI or CLI) decides when to trigger MemoryStore.ingest_directory().
   This keeps the writer stateless.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from feasiai.feedback.schemas import FeedbackRecord, ReviewDecision
from feasiai.writeback.schemas import WritebackRecord, WritebackSignalType

logger = logging.getLogger(__name__)

_DEFAULT_MEMORY_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "memory"


class MemoryWritebackService:
    """
    Converts delivery-outcome FeedbackRecords into institutional memory documents.

    Usage:
        service = MemoryWritebackService()
        records = service.records_from_feedback(feedback_record, outcome="won")
        path = service.write(records)
        # Then re-ingest: memory_store.ingest_directory()
    """

    def __init__(self, memory_dir: Optional[str | Path] = None):
        self._memory_dir = Path(memory_dir) if memory_dir else _DEFAULT_MEMORY_DIR
        self._memory_dir.mkdir(parents=True, exist_ok=True)

    def records_from_feedback(
        self,
        feedback: FeedbackRecord,
        outcome: str,
        actual_effort_days: Optional[float] = None,
        outcome_notes: Optional[str] = None,
        client_vertical: Optional[str] = None,
    ) -> list[WritebackRecord]:
        """
        Derive WritebackRecords from a completed FeedbackRecord.

        Args:
            feedback: The reviewer's feedback record.
            outcome: 'won' | 'lost' | 'delivered' | 'cancelled'
            actual_effort_days: Actual effort if the requirement was delivered.
            outcome_notes: Additional context for the memory entry.
            client_vertical: Industry vertical (e.g. 'healthcare').

        Returns:
            List of WritebackRecord objects ready to be written.
        """
        records: list[WritebackRecord] = []

        # Did the AI over-claim? (AI said Available/Configurable, actual outcome showed it wasn't)
        ai_was_positive = feedback.ai_label in ("Available", "Configurable")
        ai_was_negative = feedback.ai_label in (
            "Requires Development", "Unsupported / Unclear"
        )
        reviewer_overrode_down = (
            feedback.final_label in ("Requires Development", "Unsupported / Unclear")
            and ai_was_positive
        )
        reviewer_overrode_up = (
            feedback.final_label in ("Available", "Configurable")
            and ai_was_negative
        )

        # 1. Do-not-claim record if AI over-claimed and outcome confirms it
        if reviewer_overrode_down or (outcome == "lost" and ai_was_positive):
            content = (
                f"CAUTION: For requirements like '{feedback.requirement_text[:120]}', "
                f"the system initially assessed this as '{feedback.ai_label}' but the "
                f"reviewer corrected it to '{feedback.final_label}'."
            )
            if outcome_notes:
                content += f" Context: {outcome_notes}"
            records.append(WritebackRecord(
                source_feedback_id=feedback.id,
                requirement_text=feedback.requirement_text,
                signal_type=WritebackSignalType.DO_NOT_CLAIM,
                content=content,
                proposal_outcome=outcome,
                actual_effort_days=actual_effort_days,
                client_vertical=client_vertical,
            ))

        # 2. Capability-confirmed record if AI was correct on a positive label
        if ai_was_positive and feedback.ai_label == feedback.final_label and outcome in ("won", "delivered"):
            content = (
                f"CONFIRMED: '{feedback.requirement_text[:120]}' was assessed as "
                f"'{feedback.ai_label}' and successfully delivered. "
                f"Confidence: {feedback.ai_confidence:.0%}."
            )
            if outcome_notes:
                content += f" Notes: {outcome_notes}"
            records.append(WritebackRecord(
                source_feedback_id=feedback.id,
                requirement_text=feedback.requirement_text,
                signal_type=WritebackSignalType.CAPABILITY_CONFIRMED,
                content=content,
                proposal_outcome=outcome,
                actual_effort_days=actual_effort_days,
                client_vertical=client_vertical,
            ))

        # 3. Effort calibration if actual effort is known
        if actual_effort_days is not None and outcome in ("delivered", "won"):
            content = (
                f"EFFORT CALIBRATION: Requirement '{feedback.requirement_text[:100]}' "
                f"(label: '{feedback.final_label}') required {actual_effort_days:.1f} "
                f"engineering days to deliver."
            )
            if client_vertical:
                content += f" Client vertical: {client_vertical}."
            records.append(WritebackRecord(
                source_feedback_id=feedback.id,
                requirement_text=feedback.requirement_text,
                signal_type=WritebackSignalType.EFFORT_CALIBRATION,
                content=content,
                proposal_outcome=outcome,
                actual_effort_days=actual_effort_days,
                client_vertical=client_vertical,
            ))

        # 4. General lesson from reviewer notes
        if feedback.reviewer_notes and len(feedback.reviewer_notes) > 20:
            content = (
                f"LESSON from requirement '{feedback.requirement_text[:100]}': "
                f"{feedback.reviewer_notes}"
            )
            records.append(WritebackRecord(
                source_feedback_id=feedback.id,
                requirement_text=feedback.requirement_text,
                signal_type=WritebackSignalType.LESSON,
                content=content,
                proposal_outcome=outcome,
                actual_effort_days=actual_effort_days,
                client_vertical=client_vertical,
            ))

        return records

    def write(self, records: list[WritebackRecord]) -> Optional[Path]:
        """
        Write a batch of WritebackRecords to a new markdown memory document.

        Args:
            records: WritebackRecords to persist.

        Returns:
            Path to the created file, or None if records list is empty.
        """
        if not records:
            logger.info("MemoryWritebackService: no records to write")
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"delivery_outcomes_{timestamp}.md"
        filepath = self._memory_dir / filename

        lines = [
            f"# Delivery Outcomes — {timestamp}",
            "",
            "_Auto-generated by FeasiAI write-back service. Do not edit manually._",
            "",
            f"**Records:** {len(records)}",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            "",
            "---",
            "",
        ]

        for record in records:
            # Signal type header with visual cue
            icon = {
                WritebackSignalType.DO_NOT_CLAIM: "🚫",
                WritebackSignalType.CAPABILITY_CONFIRMED: "✅",
                WritebackSignalType.EFFORT_CALIBRATION: "⏱️",
                WritebackSignalType.LESSON: "💡",
            }.get(record.signal_type, "📝")

            lines.append(f"## {icon} {record.signal_type.value.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"**Category:** {record.capability_area or 'general'}")
            lines.append(f"**Outcome:** {record.proposal_outcome}")
            if record.actual_effort_days is not None:
                lines.append(f"**Actual effort:** {record.actual_effort_days:.1f} days")
            if record.client_vertical:
                lines.append(f"**Client vertical:** {record.client_vertical}")
            lines.append("")
            lines.append(record.content)
            lines.append("")
            lines.append(f"_Source: feedback/{record.source_feedback_id[:8]}_")
            lines.append("")
            lines.append("---")
            lines.append("")

        filepath.write_text("\n".join(lines), encoding="utf-8")
        logger.info(
            "MemoryWritebackService: wrote %d records to %s", len(records), filepath
        )
        return filepath

    def process_feedback_batch(
        self,
        feedback_records: list[FeedbackRecord],
        outcome: str,
        actual_effort_days: Optional[float] = None,
        outcome_notes: Optional[str] = None,
        client_vertical: Optional[str] = None,
    ) -> Optional[Path]:
        """
        High-level convenience: derive + write records for a batch of feedback.

        Args:
            feedback_records: Reviewer feedback (typically the 'approved' records).
            outcome: 'won' | 'lost' | 'delivered' | 'cancelled'
            actual_effort_days: Total engineering effort for the whole proposal.
            outcome_notes: Context to attach to all records in this batch.
            client_vertical: Industry vertical.

        Returns:
            Path to the written memory document, or None if nothing was produced.
        """
        all_writeback: list[WritebackRecord] = []
        for fb in feedback_records:
            wb_records = self.records_from_feedback(
                fb,
                outcome=outcome,
                actual_effort_days=actual_effort_days,
                outcome_notes=outcome_notes,
                client_vertical=client_vertical,
            )
            all_writeback.extend(wb_records)

        if not all_writeback:
            logger.info("No write-back records generated from %d feedback items", len(feedback_records))
            return None

        return self.write(all_writeback)

