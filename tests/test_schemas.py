"""Tests for Pydantic schemas — validation, serialization, edge cases."""

import pytest
from feasiai.schemas import (
    Classification,
    Evidence,
    EvidenceSource,
    ExtractionResult,
    FeasibilityLabel,
    FeasibilityReport,
    LLMCallRecord,
    Requirement,
    RetrievalResult,
)


class TestRequirement:
    def test_create_minimal(self):
        req = Requirement(text="Must support SAML 2.0 SSO")
        assert req.text == "Must support SAML 2.0 SSO"
        assert req.id  # UUID auto-generated
        assert req.is_ambiguous is False
        assert req.ambiguity_reason is None

    def test_create_ambiguous(self):
        req = Requirement(
            text="System should be fast",
            is_ambiguous=True,
            ambiguity_reason="No measurable performance criteria",
        )
        assert req.is_ambiguous is True
        assert "measurable" in req.ambiguity_reason

    def test_frozen(self):
        req = Requirement(text="Test requirement")
        with pytest.raises(Exception):
            req.text = "Modified"  # type: ignore

    def test_serialization(self):
        req = Requirement(
            text="Test",
            original_section="Section 1",
            source_document="rfp.pdf",
        )
        data = req.model_dump()
        assert data["text"] == "Test"
        assert data["original_section"] == "Section 1"

    def test_empty_text_allowed(self):
        """Empty text is technically valid at schema level; 
        extractor filters these out."""
        req = Requirement(text="")
        assert req.text == ""


class TestEvidence:
    def test_create(self):
        ev = Evidence(
            content="SAML 2.0 is supported",
            source=EvidenceSource.MEMORY,
            source_document="product_capabilities.md",
        )
        assert ev.content == "SAML 2.0 is supported"
        assert ev.source == EvidenceSource.MEMORY
        assert ev.relevance_score == 0.0
        assert ev.id  # Auto UUID

    def test_with_score(self):
        ev = Evidence(
            content="Test",
            source=EvidenceSource.LIVE_DATA,
            relevance_score=0.85,
        )
        assert ev.relevance_score == 0.85


class TestClassification:
    def test_create_available(self):
        clf = Classification(
            requirement_id="test-id",
            label=FeasibilityLabel.AVAILABLE,
            confidence=0.95,
            reasoning="Confirmed in docs",
            what_would_change="If custom mapping is needed",
            draft_response="We support this.",
        )
        assert clf.label == FeasibilityLabel.AVAILABLE
        assert clf.confidence == 0.95

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            Classification(
                requirement_id="test",
                label=FeasibilityLabel.AVAILABLE,
                confidence=1.5,  # Out of range
            )

    def test_all_labels(self):
        for label in FeasibilityLabel:
            clf = Classification(
                requirement_id="test",
                label=label,
                confidence=0.5,
            )
            assert clf.label == label


class TestFeasibilityLabel:
    def test_values(self):
        assert FeasibilityLabel.AVAILABLE.value == "Available"
        assert FeasibilityLabel.CONFIGURABLE.value == "Configurable"
        assert FeasibilityLabel.REQUIRES_DEVELOPMENT.value == "Requires Development"
        assert FeasibilityLabel.UNSUPPORTED_UNCLEAR.value == "Unsupported / Unclear"


class TestFeasibilityReport:
    def test_create_empty(self):
        report = FeasibilityReport()
        assert report.requirements == []
        assert report.classifications == []
        assert report.summary == {}
        assert report.id  # Auto UUID

    def test_serialization_roundtrip(self):
        report = FeasibilityReport(
            source_document="test.pdf",
            summary={"total_requirements": 5},
            llm_call_count=10,
        )
        data = report.model_dump(mode="json")
        assert data["source_document"] == "test.pdf"
        assert data["summary"]["total_requirements"] == 5


class TestLLMCallRecord:
    def test_create(self):
        record = LLMCallRecord(
            model="test-model",
            purpose="extraction",
            raw_response='{"result": "ok"}',
            latency_seconds=1.5,
        )
        assert record.model == "test-model"
        assert record.purpose == "extraction"
        assert record.latency_seconds == 1.5
