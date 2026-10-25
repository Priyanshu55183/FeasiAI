"""
Integration test for the full pipeline.
Uses mock LLM client so no API key is needed.
Tests the orchestration logic, not the LLM quality.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from feasiai.pipeline import FeasibilityPipeline
from feasiai.schemas import FeasibilityLabel


@pytest.fixture
def mock_pipeline(sample_memory_dir):
    """Pipeline with mock LLM and real memory store."""
    llm = MagicMock()
    llm.call_count = 0
    llm.call_log = []
    llm.reset_budget = MagicMock()

    # First call: extraction
    extraction_response = {
        "requirements": [
            {
                "text": "The system must support SSO via SAML 2.0",
                "original_section": "Authentication",
                "is_ambiguous": False,
                "ambiguity_reason": None,
            },
            {
                "text": "Native Shopify integration required",
                "original_section": "Integration",
                "is_ambiguous": False,
                "ambiguity_reason": None,
            },
        ]
    }

    # Classification responses (one per requirement)
    classification_available = {
        "label": "Available",
        "confidence": 0.9,
        "reasoning": "SSO SAML 2.0 is confirmed in product docs",
        "what_would_change": "Custom SAML attribute mapping",
        "draft_response": "We support SSO via SAML 2.0.",
    }

    classification_unsupported = {
        "label": "Unsupported / Unclear",
        "confidence": 0.95,
        "reasoning": "Shopify is listed as a known gap",
        "what_would_change": "Building a Shopify connector",
        "draft_response": "Shopify integration is not available.",
    }

    # chat_json is called: once for extraction, then once per requirement
    llm.chat_json.side_effect = [
        extraction_response,
        classification_available,
        classification_unsupported,
    ]

    pipeline = FeasibilityPipeline(llm_client=llm)
    pipeline.initialize(memory_docs_dir=str(sample_memory_dir))

    return pipeline


class TestPipelineEndToEnd:
    def test_full_pipeline_with_text(self, mock_pipeline):
        report = mock_pipeline.run(
            input_text="Test RFP content with requirements",
            source_name="test_rfp.txt",
        )

        assert report.source_document == "test_rfp.txt"
        assert len(report.requirements) == 2
        assert len(report.classifications) == 2

        # Check labels
        labels = [c.label for c in report.classifications]
        assert FeasibilityLabel.AVAILABLE in labels
        assert FeasibilityLabel.UNSUPPORTED_UNCLEAR in labels

    def test_full_pipeline_with_file(self, mock_pipeline, tmp_path):
        """Test with a file input."""
        rfp_file = tmp_path / "test.md"
        rfp_file.write_text("# Test RFP\n\nRequirement: Support SSO")

        report = mock_pipeline.run(input_path=str(rfp_file))
        assert report.source_document == "test.md"
        assert len(report.requirements) > 0

    def test_empty_input_raises(self, mock_pipeline):
        with pytest.raises(ValueError, match="Must provide"):
            mock_pipeline.run()

    def test_report_formatting(self, mock_pipeline):
        report = mock_pipeline.run(input_text="Test content")
        text = mock_pipeline.format_report(report)

        assert "FEASIBILITY ASSESSMENT REPORT" in text
        assert "Available" in text
        assert "HUMAN REVIEW REQUIRED" in text

    def test_summary_stats(self, mock_pipeline):
        report = mock_pipeline.run(input_text="Test content")
        summary = report.summary

        assert summary["total_requirements"] == 2
        assert summary["available_count"] == 1
        assert summary["unsupported_count"] == 1

    def test_partial_failure_tolerance(self, sample_memory_dir):
        """Pipeline should handle individual requirement failures gracefully."""
        llm = MagicMock()
        llm.call_count = 0
        llm.call_log = []
        llm.reset_budget = MagicMock()

        # Extraction succeeds
        llm.chat_json.side_effect = [
            {
                "requirements": [
                    {"text": "Req 1", "is_ambiguous": False, "ambiguity_reason": None, "original_section": None},
                    {"text": "Req 2", "is_ambiguous": False, "ambiguity_reason": None, "original_section": None},
                ]
            },
            # First classification succeeds
            {
                "label": "Available",
                "confidence": 0.8,
                "reasoning": "OK",
                "what_would_change": "Nothing",
                "draft_response": "Yes",
            },
            # Second classification fails
            Exception("LLM timeout"),
        ]

        pipeline = FeasibilityPipeline(llm_client=llm)
        pipeline.initialize(str(sample_memory_dir))

        report = pipeline.run(input_text="Test content")

        # Should still produce a report with both requirements
        assert len(report.classifications) == 2
        # Failed one should be classified as Unsupported/Unclear
        failed = [c for c in report.classifications if c.confidence == 0.0]
        assert len(failed) == 1
