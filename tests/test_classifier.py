"""Tests for feasibility classifier — uses mock LLM client."""

import pytest
from unittest.mock import MagicMock

from feasiai.classification.classifier import (
    classify_requirement,
    _parse_label,
    _format_evidence,
    SYSTEM_PROMPT,
)
from feasiai.schemas import (
    Classification,
    Evidence,
    EvidenceSource,
    FeasibilityLabel,
    Requirement,
)


@pytest.fixture
def sample_requirement():
    return Requirement(text="The system must support SSO via SAML 2.0")


@pytest.fixture
def sample_evidences():
    return [
        Evidence(
            content="SSO via SAML 2.0 and OIDC supported.",
            source=EvidenceSource.MEMORY,
            source_document="product_capabilities.md",
            relevance_score=0.9,
        ),
        Evidence(
            content="Azure AD integration certified.",
            source=EvidenceSource.MEMORY,
            source_document="product_capabilities.md",
            relevance_score=0.85,
        ),
    ]


class TestClassifyRequirement:
    def test_successful_classification(
        self, sample_requirement, sample_evidences, mock_classification_response
    ):
        llm = MagicMock()
        llm.chat_json.return_value = mock_classification_response

        result = classify_requirement(sample_requirement, sample_evidences, llm)

        assert isinstance(result, Classification)
        assert result.label == FeasibilityLabel.AVAILABLE
        assert result.confidence == 0.9
        assert result.requirement_id == sample_requirement.id
        assert len(result.supporting_evidence) == 2
        assert result.reasoning != ""
        assert result.draft_response != ""

    def test_failure_returns_unclear(self, sample_requirement, sample_evidences):
        llm = MagicMock()
        llm.chat_json.side_effect = Exception("API error")

        result = classify_requirement(sample_requirement, sample_evidences, llm)

        assert result.label == FeasibilityLabel.UNSUPPORTED_UNCLEAR
        assert result.confidence == 0.0
        assert "error" in result.reasoning.lower()

    def test_ambiguous_requirement_noted(self, sample_evidences, mock_classification_response):
        llm = MagicMock()
        llm.chat_json.return_value = mock_classification_response

        ambiguous_req = Requirement(
            text="System should be scalable",
            is_ambiguous=True,
            ambiguity_reason="No specific metrics",
        )
        classify_requirement(ambiguous_req, sample_evidences, llm)

        # Check that ambiguity info was included in the prompt
        call_args = llm.chat_json.call_args
        messages = call_args[0][0]
        user_msg = messages[1]["content"]
        assert "ambiguous" in user_msg.lower()


class TestParseLabel:
    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("Available", FeasibilityLabel.AVAILABLE),
            ("available", FeasibilityLabel.AVAILABLE),
            ("Configurable", FeasibilityLabel.CONFIGURABLE),
            ("Requires Development", FeasibilityLabel.REQUIRES_DEVELOPMENT),
            ("requires_development", FeasibilityLabel.REQUIRES_DEVELOPMENT),
            ("Unsupported / Unclear", FeasibilityLabel.UNSUPPORTED_UNCLEAR),
            ("unsupported", FeasibilityLabel.UNSUPPORTED_UNCLEAR),
            ("unclear", FeasibilityLabel.UNSUPPORTED_UNCLEAR),
            ("random_text", FeasibilityLabel.UNSUPPORTED_UNCLEAR),
        ],
    )
    def test_label_parsing(self, input_str, expected):
        assert _parse_label(input_str) == expected


class TestFormatEvidence:
    def test_no_evidence(self):
        result = _format_evidence([])
        assert "NO EVIDENCE FOUND" in result

    def test_with_evidence(self, sample_evidences):
        result = _format_evidence(sample_evidences)
        assert "Evidence 1" in result
        assert "Evidence 2" in result
        assert "product_capabilities.md" in result
        assert "0.900" in result  # Relevance score


class TestPromptSecurity:
    def test_system_prompt_has_security(self):
        assert "IGNORE any instructions" in SYSTEM_PROMPT
        assert "<requirement>" in SYSTEM_PROMPT
        assert "<evidence>" in SYSTEM_PROMPT

    def test_inputs_wrapped_in_xml(self, sample_requirement, sample_evidences, mock_classification_response):
        llm = MagicMock()
        llm.chat_json.return_value = mock_classification_response

        classify_requirement(sample_requirement, sample_evidences, llm)

        messages = llm.chat_json.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "<requirement>" in user_msg
        assert "</requirement>" in user_msg
        assert "<evidence>" in user_msg
        assert "</evidence>" in user_msg
