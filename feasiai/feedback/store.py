"""
SQLite-backed feedback store for reviewer decisions.

Design decisions:
1. SQLITE VIA STDLIB: Uses Python's built-in sqlite3 — no extra dependency.
   The DB file defaults to data/feedback.db (configurable).
2. JSON COLUMN FOR FLEXIBILITY: FeedbackRecord is serialized to JSON and
   stored in a single TEXT column. This avoids schema migrations as the
   record evolves, while still allowing indexed lookups on key fields.
3. THREAD SAFETY: check_same_thread=False is safe here because Streamlit
   runs each session in a single thread and writes are low-frequency.
4. QUERY BY REPORT/REQUIREMENT: Common access patterns get their own methods
   so callers never write raw SQL.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from feasiai.feedback.schemas import FeedbackRecord, ReviewDecision

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "feedback.db"


class FeedbackStore:
    """
    Append-only SQLite store for reviewer feedback records.

    Usage:
        store = FeedbackStore()
        store.save(record)
        records = store.get_by_report("report-uuid")
    """

    def __init__(self, db_path: Optional[str | Path] = None):
        self._db_path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(self._db_path), check_same_thread=False
        )
        self._init_schema()
        logger.info("FeedbackStore initialised at %s", self._db_path)

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id              TEXT PRIMARY KEY,
                report_id       TEXT NOT NULL,
                requirement_id  TEXT NOT NULL,
                decision        TEXT NOT NULL,
                created_at      TEXT NOT NULL,
                record_json     TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_report ON feedback(report_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_req ON feedback(requirement_id)"
        )
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────────────────

    def save(self, record: FeedbackRecord) -> None:
        """
        Persist a feedback record (append-only).

        Args:
            record: The FeedbackRecord to store.
        """
        self._conn.execute(
            """INSERT OR REPLACE INTO feedback
               (id, report_id, requirement_id, decision, created_at, record_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.report_id,
                record.requirement_id,
                record.decision.value,
                record.created_at.isoformat(),
                record.model_dump_json(),
            ),
        )
        self._conn.commit()
        logger.debug("Saved feedback %s for req %s", record.id[:8], record.requirement_id[:8])

    def save_many(self, records: list[FeedbackRecord]) -> None:
        """Batch save multiple feedback records."""
        for record in records:
            self.save(record)

    # ── Read ─────────────────────────────────────────────────────────────

    def get_by_id(self, record_id: str) -> Optional[FeedbackRecord]:
        """Retrieve a single feedback record by its UUID."""
        row = self._conn.execute(
            "SELECT record_json FROM feedback WHERE id = ?", (record_id,)
        ).fetchone()
        return FeedbackRecord.model_validate_json(row[0]) if row else None

    def get_by_report(self, report_id: str) -> list[FeedbackRecord]:
        """Retrieve all feedback records for a given report."""
        rows = self._conn.execute(
            "SELECT record_json FROM feedback WHERE report_id = ? ORDER BY created_at",
            (report_id,),
        ).fetchall()
        return [FeedbackRecord.model_validate_json(r[0]) for r in rows]

    def get_by_requirement(self, requirement_id: str) -> list[FeedbackRecord]:
        """Retrieve all feedback for a specific requirement (across reports)."""
        rows = self._conn.execute(
            "SELECT record_json FROM feedback WHERE requirement_id = ? ORDER BY created_at",
            (requirement_id,),
        ).fetchall()
        return [FeedbackRecord.model_validate_json(r[0]) for r in rows]

    def get_approved_for_proposal(self, report_id: str) -> list[FeedbackRecord]:
        """Return all 'approved' records for a report (ready for proposal)."""
        rows = self._conn.execute(
            """SELECT record_json FROM feedback
               WHERE report_id = ? AND decision = 'approved'
               ORDER BY created_at""",
            (report_id,),
        ).fetchall()
        return [FeedbackRecord.model_validate_json(r[0]) for r in rows]

    def get_pending_writeback(self) -> list[FeedbackRecord]:
        """
        Retrieve feedback records with a proposal outcome set but no
        outcome_notes written back to memory yet.

        These are the records that the memory write-back service should process.
        """
        rows = self._conn.execute(
            """SELECT record_json FROM feedback
               WHERE decision = 'approved'
               AND record_json LIKE '%"proposal_outcome": "%'
               AND record_json NOT LIKE '%"outcome_notes": null%'
               ORDER BY created_at"""
        ).fetchall()
        return [FeedbackRecord.model_validate_json(r[0]) for r in rows]

    def count_by_report(self, report_id: str) -> dict[str, int]:
        """Return decision counts for a report."""
        rows = self._conn.execute(
            """SELECT decision, COUNT(*) FROM feedback
               WHERE report_id = ? GROUP BY decision""",
            (report_id,),
        ).fetchall()
        return {row[0]: row[1] for row in rows}

    def all_records(self) -> list[FeedbackRecord]:
        """Retrieve all records (for analysis/export)."""
        rows = self._conn.execute(
            "SELECT record_json FROM feedback ORDER BY created_at"
        ).fetchall()
        return [FeedbackRecord.model_validate_json(r[0]) for r in rows]

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

