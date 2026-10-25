"""Tests for the report generator."""

import pytest

from feasiai.report.generator import build_report, format_report_text
from feasiai.schemas import (
    Classification,
    Evidence,
    EvidenceSource,
    FeasibilityLabel,
    Requirement,
)


@pytest.fixture
def sample_report_data():
    """Create a set of requirements and classifications for testing."""
    reqs = [
        Requirement(text="Must support SSO via SAML 2.0"),
        Requirement(
            text="System should be fast",
            is_ambiguous=True,
            ambiguity_reason="No measurable criteria",
        ),
        Requirement(text="Must integrate with Shopify"),
    ]

    classifications = [
        Classification(
            requirement_id=reqs[0].id,
            label=FeasibilityLabel.AVAILABLE,
            confidence=0.92,
            reasoning="Confirmed in product docs",
            what_would_change="Custom SAML attribute mapping",
            draft_response="We support SSO via SAML 2.0.",
            supporting_evidence=[
                Evidence(
                    content="SSO SAML 2.0 supported",
                    source=EvidenceSource.MEMORY,
                    source_document="product_capabilities.md",
                )
            ],
        ),
        Classification(
            requirement_id=reqs[1].id,
            label=FeasibilityLabel.UNSUPPORTED_UNCLEAR,
            confidence=0.3,
            reasoning="Ambiguous requirement without metrics",
            what_would_change="Specific performance benchmarks",
            draft_response="Please clarify performance requirements.",
        ),
        Classification(
            requirement_id=reqs[2].id,
            label=FeasibilityLabel.UNSUPPORTED_UNCLEAR,
            confidence=0.95,
            reasoning="Listed as known gap",
            what_would_change="If we build a Shopify connector",
            draft_response="Shopify integration is not currently available.",
            supporting_evidence=[
                Evidence(
                    content="No Shopify integration",
                    source=EvidenceSource.MEMORY,
                    source_document="known_gaps.md",
                )
            ],
        ),
    ]

    return reqs, classifications


class TestBuildReport:
    def test_summary_stats(self, sample_report_data):
        reqs, classifications = sample_report_data
        report = build_report(reqs, classifications, source_document="test.md")

        assert report.summary["total_requirements"] == 3
        assert report.summary["available_count"] == 1
        assert report.summary["unsupported_count"] == 2
        assert report.summary["ambiguous_requirements"] == 1
        assert 0.0 <= report.summary["average_confidence"] <= 1.0

    def test_empty_report(self):
        report = build_report([], [])
        assert report.summary["total_requirements"] == 0

    def test_source_document(self, sample_report_data):
        reqs, classifications = sample_report_data
        report = build_report(reqs, classifications, source_document="rfp.pdf")
        assert report.source_document == "rfp.pdf"

    def test_llm_metadata(self, sample_report_data):
        reqs, classifications = sample_report_data
        report = build_report(
            reqs, classifications,
            llm_call_count=15,
            total_latency_seconds=12.5,
        )
        assert report.llm_call_count == 15
        assert report.total_latency_seconds == 12.5


class TestFormatReport:
    def test_format_contains_key_sections(self, sample_report_data):
        reqs, classifications = sample_report_data
        report = build_report(reqs, classifications, source_document="test.md")
        text = format_report_text(report)

        assert "FEASIBILITY ASSESSMENT REPORT" in text
        assert "SUMMARY" in text
        assert "DETAILED ASSESSMENTS" in text
        assert "HUMAN REVIEW REQUIRED" in text

    def test_format_shows_labels(self, sample_report_data):
        reqs, classifications = sample_report_data
        report = build_report(reqs, classifications)
        text = format_report_text(report)

        assert "Available" in text
        assert "Unsupported / Unclear" in text

    def test_format_shows_ambiguity(self, sample_report_data):
        reqs, classifications = sample_report_data
        report = build_report(reqs, classifications)
        text = format_report_text(report)

        assert "AMBIGUOUS" in text
