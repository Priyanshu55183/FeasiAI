"""
Tests for Phase 4: Feedback capture, memory write-back, and closed-loop.

Coverage:
- FeedbackStore: save/retrieve, decision filtering, approval queries
- FeedbackRecord: Pydantic validation, frozen model
- MemoryWritebackService: derive records from feedback, write markdown files
- WritebackRecord: signal types, content generation
- End-to-end: feedback → write-back → markdown file content
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from datetime import datetime, timezone

import pytest

from feasiai.feedback.schemas import FeedbackRecord, ReviewDecision
from feasiai.feedback.store import FeedbackStore
from feasiai.writeback.schemas import WritebackRecord, WritebackSignalType
from feasiai.writeback.writer import MemoryWritebackService


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_feedback.db")


@pytest.fixture
def store(db_path):
    return FeedbackStore(db_path=db_path)


@pytest.fixture
def memory_dir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture
def writeback_service(memory_dir):
    return MemoryWritebackService(memory_dir=memory_dir)


def _make_record(
    report_id: str = "report-001",
    requirement_id: str = "req-001",
    decision: ReviewDecision = ReviewDecision.APPROVED,
    ai_label: str = "Available",
    final_label: str = "Available",
    ai_confidence: float = 0.85,
    reviewer_notes: str = "",
    draft_response_approved: bool = True,
    requirement_text: str = "The system must support SSO via SAML 2.0",
) -> FeedbackRecord:
    return FeedbackRecord(
        report_id=report_id,
        requirement_id=requirement_id,
        ai_label=ai_label,
        ai_confidence=ai_confidence,
        final_label=final_label,
        decision=decision,
        reviewer_notes=reviewer_notes,
        draft_response_approved=draft_response_approved,
        requirement_text=requirement_text,
    )


# ─── FeedbackRecord Schema Tests ─────────────────────────────────────────────


class TestFeedbackRecordSchema:
    def test_has_auto_id(self):
        r = _make_record()
        assert r.id and len(r.id) > 0

    def test_has_created_at(self):
        r = _make_record()
        assert isinstance(r.created_at, datetime)

    def test_decision_enum(self):
        r = _make_record(decision=ReviewDecision.OVERRIDDEN)
        assert r.decision == ReviewDecision.OVERRIDDEN

    def test_all_decision_values(self):
        for decision in ReviewDecision:
            r = _make_record(decision=decision)
            assert r.decision == decision

    def test_frozen_model(self):
        r = _make_record()
        with pytest.raises(Exception):
            r.ai_label = "changed"  # type: ignore

    def test_optional_outcome_fields_default_none(self):
        r = _make_record()
        assert r.proposal_outcome is None
        assert r.actual_effort_days is None
        assert r.outcome_notes is None

    def test_confidence_stored(self):
        r = _make_record(ai_confidence=0.75)
        assert r.ai_confidence == 0.75

    def test_serializes_to_json(self):
        r = _make_record()
        json_str = r.model_dump_json()
        assert "report-001" in json_str
        assert "Available" in json_str

    def test_roundtrip_from_json(self):
        r = _make_record()
        json_str = r.model_dump_json()
        r2 = FeedbackRecord.model_validate_json(json_str)
        assert r2.id == r.id
        assert r2.decision == r.decision


# ─── FeedbackStore Tests ──────────────────────────────────────────────────────


class TestFeedbackStore:
    def test_save_and_retrieve_by_id(self, store):
        r = _make_record()
        store.save(r)
        retrieved = store.get_by_id(r.id)
        assert retrieved is not None
        assert retrieved.id == r.id

    def test_retrieve_missing_returns_none(self, store):
        assert store.get_by_id("nonexistent-id") is None

    def test_save_multiple_retrieve_by_report(self, store):
        r1 = _make_record(report_id="rpt-1", requirement_id="req-1")
        r2 = _make_record(report_id="rpt-1", requirement_id="req-2")
        r3 = _make_record(report_id="rpt-2", requirement_id="req-3")
        store.save(r1)
        store.save(r2)
        store.save(r3)

        rpt1_records = store.get_by_report("rpt-1")
        assert len(rpt1_records) == 2
        rpt2_records = store.get_by_report("rpt-2")
        assert len(rpt2_records) == 1

    def test_retrieve_by_requirement(self, store):
        r1 = _make_record(report_id="rpt-1", requirement_id="req-sso")
        r2 = _make_record(report_id="rpt-2", requirement_id="req-sso")
        store.save(r1)
        store.save(r2)

        records = store.get_by_requirement("req-sso")
        assert len(records) == 2

    def test_get_approved_for_proposal(self, store):
        r_approved = _make_record(
            report_id="rpt-1", requirement_id="req-1",
            decision=ReviewDecision.APPROVED,
        )
        r_flagged = _make_record(
            report_id="rpt-1", requirement_id="req-2",
            decision=ReviewDecision.FLAGGED,
        )
        store.save(r_approved)
        store.save(r_flagged)

        approved = store.get_approved_for_proposal("rpt-1")
        assert len(approved) == 1
        assert approved[0].decision == ReviewDecision.APPROVED

    def test_len(self, store):
        assert len(store) == 0
        store.save(_make_record(requirement_id="req-1"))
        store.save(_make_record(requirement_id="req-2"))
        assert len(store) == 2

    def test_save_many(self, store):
        records = [
            _make_record(requirement_id=f"req-{i}")
            for i in range(5)
        ]
        store.save_many(records)
        assert len(store) == 5

    def test_all_records(self, store):
        store.save(_make_record(requirement_id="req-1"))
        store.save(_make_record(requirement_id="req-2"))
        all_recs = store.all_records()
        assert len(all_recs) == 2

    def test_count_by_report(self, store):
        store.save(_make_record(requirement_id="req-1", decision=ReviewDecision.APPROVED))
        store.save(_make_record(requirement_id="req-2", decision=ReviewDecision.OVERRIDDEN))
        store.save(_make_record(requirement_id="req-3", decision=ReviewDecision.APPROVED))

        counts = store.count_by_report("report-001")
        assert counts.get("approved", 0) == 2
        assert counts.get("overridden", 0) == 1

    def test_save_overwrites_on_same_id(self, store):
        """INSERT OR REPLACE behavior — same ID replaces the record."""
        r = _make_record()
        store.save(r)
        # Save again with same ID — should not create a duplicate
        store.save(r)
        all_recs = store.all_records()
        assert len(all_recs) == 1

    def test_db_persists_across_instances(self, db_path):
        """Data written by one FeedbackStore instance is readable by another."""
        store1 = FeedbackStore(db_path=db_path)
        r = _make_record()
        store1.save(r)
        store1.close()

        store2 = FeedbackStore(db_path=db_path)
        retrieved = store2.get_by_id(r.id)
        assert retrieved is not None
        assert retrieved.id == r.id
        store2.close()

    def test_decision_roundtrip(self, store):
        """Enum values survive SQLite serialization."""
        for decision in ReviewDecision:
            r = _make_record(
                requirement_id=f"req-{decision.value}",
                decision=decision,
            )
            store.save(r)
            retrieved = store.get_by_id(r.id)
            assert retrieved.decision == decision


# ─── WritebackRecord Schema Tests ─────────────────────────────────────────────


class TestWritebackRecordSchema:
    def test_has_auto_id(self):
        r = WritebackRecord(
            source_feedback_id="fb-001",
            requirement_text="SSO support required",
            signal_type=WritebackSignalType.LESSON,
            content="Lesson learned: SSO setup took 2 extra days.",
            proposal_outcome="won",
        )
        assert r.id and len(r.id) > 0

    def test_all_signal_types(self):
        for signal in WritebackSignalType:
            r = WritebackRecord(
                source_feedback_id="fb-001",
                requirement_text="test",
                signal_type=signal,
                content="test content",
                proposal_outcome="won",
            )
            assert r.signal_type == signal

    def test_frozen(self):
        r = WritebackRecord(
            source_feedback_id="fb-001",
            requirement_text="test",
            signal_type=WritebackSignalType.LESSON,
            content="test",
            proposal_outcome="won",
        )
        with pytest.raises(Exception):
            r.content = "changed"  # type: ignore


# ─── MemoryWritebackService Tests ─────────────────────────────────────────────


class TestMemoryWritebackService:
    def test_records_from_feedback_do_not_claim_on_override_down(self, writeback_service):
        """AI said Available, reviewer corrected to Requires Development → do_not_claim."""
        fb = _make_record(
            ai_label="Available",
            final_label="Requires Development",
            decision=ReviewDecision.OVERRIDDEN,
        )
        records = writeback_service.records_from_feedback(fb, outcome="won")
        types = [r.signal_type for r in records]
        assert WritebackSignalType.DO_NOT_CLAIM in types

    def test_records_from_feedback_capability_confirmed_on_positive_delivered(self, writeback_service):
        """AI said Available, reviewer approved, outcome=delivered → capability_confirmed."""
        fb = _make_record(
            ai_label="Available",
            final_label="Available",
            decision=ReviewDecision.APPROVED,
        )
        records = writeback_service.records_from_feedback(fb, outcome="delivered")
        types = [r.signal_type for r in records]
        assert WritebackSignalType.CAPABILITY_CONFIRMED in types

    def test_records_from_feedback_effort_calibration_when_effort_given(self, writeback_service):
        """actual_effort_days provided → effort_calibration record generated."""
        fb = _make_record(
            ai_label="Requires Development",
            final_label="Requires Development",
            decision=ReviewDecision.APPROVED,
        )
        records = writeback_service.records_from_feedback(
            fb, outcome="delivered", actual_effort_days=14.5
        )
        types = [r.signal_type for r in records]
        assert WritebackSignalType.EFFORT_CALIBRATION in types

        effort_rec = next(r for r in records if r.signal_type == WritebackSignalType.EFFORT_CALIBRATION)
        assert "14.5" in effort_rec.content

    def test_records_from_feedback_lesson_from_notes(self, writeback_service):
        """Long reviewer notes produce a LESSON record."""
        fb = _make_record(
            reviewer_notes="The integration took 3 extra days due to undocumented rate limits.",
            decision=ReviewDecision.APPROVED,
        )
        records = writeback_service.records_from_feedback(fb, outcome="delivered")
        types = [r.signal_type for r in records]
        assert WritebackSignalType.LESSON in types

    def test_records_from_feedback_no_lesson_for_short_notes(self, writeback_service):
        """Short reviewer notes (< 20 chars) don't generate a LESSON record."""
        fb = _make_record(reviewer_notes="ok", decision=ReviewDecision.APPROVED)
        records = writeback_service.records_from_feedback(fb, outcome="won")
        types = [r.signal_type for r in records]
        assert WritebackSignalType.LESSON not in types

    def test_records_from_feedback_no_capability_confirmed_on_loss(self, writeback_service):
        """Available + outcome=lost should not generate capability_confirmed."""
        fb = _make_record(ai_label="Available", final_label="Available")
        records = writeback_service.records_from_feedback(fb, outcome="lost")
        types = [r.signal_type for r in records]
        assert WritebackSignalType.CAPABILITY_CONFIRMED not in types

    def test_write_creates_markdown_file(self, writeback_service, memory_dir):
        """write() creates a .md file in the memory directory."""
        wb_record = WritebackRecord(
            source_feedback_id="fb-001",
            requirement_text="SSO test",
            signal_type=WritebackSignalType.LESSON,
            content="A lesson about SSO integration.",
            proposal_outcome="won",
        )
        path = writeback_service.write([wb_record])
        assert path is not None
        assert path.exists()
        assert path.suffix == ".md"
        assert path.parent == memory_dir

    def test_write_file_contains_content(self, writeback_service):
        wb_record = WritebackRecord(
            source_feedback_id="fb-001",
            requirement_text="SSO test",
            signal_type=WritebackSignalType.DO_NOT_CLAIM,
            content="CAUTION: SSO was over-claimed.",
            proposal_outcome="lost",
        )
        path = writeback_service.write([wb_record])
        text = path.read_text(encoding="utf-8")
        assert "CAUTION: SSO was over-claimed." in text
        assert "Do Not Claim" in text or "do_not_claim" in text.lower()

    def test_write_file_contains_outcome(self, writeback_service):
        wb_record = WritebackRecord(
            source_feedback_id="fb-001",
            requirement_text="test",
            signal_type=WritebackSignalType.CAPABILITY_CONFIRMED,
            content="Confirmed capability.",
            proposal_outcome="delivered",
        )
        path = writeback_service.write([wb_record])
        text = path.read_text(encoding="utf-8")
        assert "delivered" in text

    def test_write_empty_records_returns_none(self, writeback_service):
        """No records → no file created, returns None."""
        result = writeback_service.write([])
        assert result is None

    def test_write_multiple_records_in_one_file(self, writeback_service, memory_dir):
        """Multiple records go into one file per write() call."""
        records = [
            WritebackRecord(
                source_feedback_id="fb-001",
                requirement_text="test",
                signal_type=WritebackSignalType.LESSON,
                content=f"Lesson {i}",
                proposal_outcome="won",
            )
            for i in range(3)
        ]
        path = writeback_service.write(records)
        assert path is not None
        text = path.read_text(encoding="utf-8")
        assert "Lesson 0" in text
        assert "Lesson 1" in text
        assert "Lesson 2" in text

    def test_process_feedback_batch_end_to_end(self, writeback_service):
        """process_feedback_batch() generates records and writes them."""
        fb1 = _make_record(
            requirement_id="req-1",
            ai_label="Available",
            final_label="Requires Development",
            decision=ReviewDecision.OVERRIDDEN,
        )
        fb2 = _make_record(
            requirement_id="req-2",
            ai_label="Available",
            final_label="Available",
            decision=ReviewDecision.APPROVED,
        )
        path = writeback_service.process_feedback_batch(
            feedback_records=[fb1, fb2],
            outcome="won",
            actual_effort_days=10.0,
            client_vertical="healthcare",
        )
        assert path is not None
        text = path.read_text(encoding="utf-8")
        # fb1 → do_not_claim (AI over-claimed, reviewer corrected)
        # fb2 → capability_confirmed (available + won) + effort_calibration (days given)
        assert len(text) > 100

    def test_process_feedback_batch_empty_returns_none(self, writeback_service):
        path = writeback_service.process_feedback_batch([], outcome="won")
        assert path is None

    def test_writeback_client_vertical_in_content(self, writeback_service):
        fb = _make_record(
            ai_label="Requires Development",
            final_label="Requires Development",
            decision=ReviewDecision.APPROVED,
        )
        records = writeback_service.records_from_feedback(
            fb, outcome="delivered",
            actual_effort_days=5.0,
            client_vertical="fintech",
        )
        effort_recs = [r for r in records if r.signal_type == WritebackSignalType.EFFORT_CALIBRATION]
        assert any("fintech" in r.content for r in effort_recs)

    def test_do_not_claim_when_lost_and_ai_was_positive(self, writeback_service):
        """AI said Available, outcome=lost → do_not_claim even without reviewer override."""
        fb = _make_record(
            ai_label="Available",
            final_label="Available",
            decision=ReviewDecision.APPROVED,
        )
        records = writeback_service.records_from_feedback(fb, outcome="lost")
        types = [r.signal_type for r in records]
        assert WritebackSignalType.DO_NOT_CLAIM in types

