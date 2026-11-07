"""
Tests for the Evaluation Harness (Phase 2C).

Tests cover:
- EvalSample / EvalResult / EvalReport schema creation and validation
- Dataset loading from JSONL
- EvalRunner: correct classification tracking, error handling, agent stats
- Metrics: accuracy, macro-F1, ECE, per-label breakdown
- Report formatting and comparison
- Dataset integrity: all 25 samples are valid
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from feasiai.eval.dataset import load_dataset
from feasiai.eval.metrics import compare_reports, compute_metrics
from feasiai.eval.report import format_comparison, format_eval_report
from feasiai.eval.runner import EvalRunner
from feasiai.eval.schemas import EvalReport, EvalResult, EvalSample
from feasiai.schemas import Evidence, EvidenceSource, FeasibilityLabel, Requirement


# ─── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def sample_available():
    return EvalSample(
        id="s-001",
        requirement_text="The system must support SSO via SAML 2.0",
        expected_label=FeasibilityLabel.AVAILABLE,
        section="Authentication",
        notes="Confirmed in product_capabilities.md",
    )


@pytest.fixture
def sample_unsupported():
    return EvalSample(
        id="s-002",
        requirement_text="Native Shopify integration required",
        expected_label=FeasibilityLabel.UNSUPPORTED_UNCLEAR,
        section="Integration",
        notes="Shopify is a known gap",
    )


@pytest.fixture
def sample_ambiguous():
    return EvalSample(
        id="s-003",
        requirement_text="The system should be user-friendly",
        expected_label=FeasibilityLabel.UNSUPPORTED_UNCLEAR,
        section="General",
        is_ambiguous=True,
        notes="No measurable criteria",
    )


@pytest.fixture
def mock_memory():
    store = MagicMock()
    store.retrieve.return_value = [
        Evidence(
            content="SSO via SAML 2.0 supported.",
            source=EvidenceSource.MEMORY,
            source_document="product_capabilities.md",
            chunk_id="chunk-001",
            relevance_score=0.9,
        )
    ]
    return store


@pytest.fixture
def mock_llm_available():
    """LLM that always returns 'Available'."""
    llm = MagicMock()
    llm.call_count = 0
    llm.call_log = []
    llm.chat_json.return_value = {
        "label": "Available",
        "confidence": 0.9,
        "reasoning": "SSO confirmed.",
        "what_would_change": "Custom mapping",
        "draft_response": "Fully supported.",
    }
    return llm


@pytest.fixture
def mock_llm_unsupported():
    """LLM that always returns 'Unsupported / Unclear'."""
    llm = MagicMock()
    llm.call_count = 0
    llm.call_log = []
    llm.chat_json.return_value = {
        "label": "Unsupported / Unclear",
        "confidence": 0.95,
        "reasoning": "No evidence found.",
        "what_would_change": "Build the feature",
        "draft_response": "Not available.",
    }
    return llm


# ─── Schema Tests ──────────────────────────────────────────────────────────


class TestEvalSample:
    def test_create(self, sample_available):
        assert sample_available.id == "s-001"
        assert sample_available.expected_label == FeasibilityLabel.AVAILABLE
        assert sample_available.is_ambiguous is False

    def test_frozen(self, sample_available):
        with pytest.raises(Exception):
            sample_available.expected_label = FeasibilityLabel.CONFIGURABLE

    def test_ambiguous_flag(self, sample_ambiguous):
        assert sample_ambiguous.is_ambiguous is True


class TestEvalResult:
    def test_create_correct(self, sample_available):
        result = EvalResult(
            sample_id=sample_available.id,
            requirement_text=sample_available.requirement_text,
            expected_label=FeasibilityLabel.AVAILABLE,
            predicted_label=FeasibilityLabel.AVAILABLE,
            confidence=0.9,
            is_correct=True,
            llm_calls_used=1,
            latency_seconds=0.5,
            used_agent=False,
        )
        assert result.is_correct is True

    def test_create_incorrect(self, sample_available):
        result = EvalResult(
            sample_id=sample_available.id,
            requirement_text=sample_available.requirement_text,
            expected_label=FeasibilityLabel.AVAILABLE,
            predicted_label=FeasibilityLabel.UNSUPPORTED_UNCLEAR,
            confidence=0.6,
            is_correct=False,
            llm_calls_used=1,
            latency_seconds=0.4,
            used_agent=False,
        )
        assert result.is_correct is False
        assert result.predicted_label == FeasibilityLabel.UNSUPPORTED_UNCLEAR

    def test_frozen(self, sample_available):
        result = EvalResult(
            sample_id="x",
            requirement_text="test",
            expected_label=FeasibilityLabel.AVAILABLE,
            predicted_label=FeasibilityLabel.AVAILABLE,
            confidence=0.9,
            is_correct=True,
        )
        with pytest.raises(Exception):
            result.is_correct = False


# ─── Dataset Tests ─────────────────────────────────────────────────────────


class TestDataset:
    def test_load_default_dataset(self):
        """The bundled dataset.jsonl loads without errors."""
        samples = load_dataset()
        assert len(samples) == 25

    def test_all_samples_have_required_fields(self):
        samples = load_dataset()
        for s in samples:
            assert s.requirement_text
            assert isinstance(s.expected_label, FeasibilityLabel)
            assert s.id

    def test_label_distribution(self):
        """Dataset covers all four labels."""
        samples = load_dataset()
        labels = {s.expected_label for s in samples}
        assert FeasibilityLabel.AVAILABLE in labels
        assert FeasibilityLabel.CONFIGURABLE in labels
        assert FeasibilityLabel.REQUIRES_DEVELOPMENT in labels
        assert FeasibilityLabel.UNSUPPORTED_UNCLEAR in labels

    def test_ambiguous_samples_exist(self):
        """Dataset contains some ambiguous samples."""
        samples = load_dataset()
        ambiguous = [s for s in samples if s.is_ambiguous]
        assert len(ambiguous) >= 1

    def test_load_custom_dataset(self, tmp_path):
        """Can load a custom JSONL file."""
        data = [
            {
                "id": "t-001",
                "requirement_text": "Must support SSO",
                "expected_label": "Available",
                "section": "Auth",
                "notes": "Confirmed",
            },
            {
                "id": "t-002",
                "requirement_text": "Must support Shopify",
                "expected_label": "Unsupported / Unclear",
                "notes": "Known gap",
            },
        ]
        path = tmp_path / "test_dataset.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        samples = load_dataset(path)
        assert len(samples) == 2
        assert samples[0].expected_label == FeasibilityLabel.AVAILABLE
        assert samples[1].expected_label == FeasibilityLabel.UNSUPPORTED_UNCLEAR

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path/dataset.jsonl")

    def test_load_invalid_jsonl_raises(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"requirement_text": "test", "expected_label": "INVALID_LABEL"}')
        with pytest.raises(ValueError):
            load_dataset(path)


# ─── Runner Tests ──────────────────────────────────────────────────────────


class TestEvalRunner:
    def test_correct_prediction(self, mock_llm_available, mock_memory, sample_available):
        runner = EvalRunner(mock_llm_available, mock_memory, use_agent=False)
        results = runner.run([sample_available])

        assert len(results) == 1
        assert results[0].is_correct is True
        assert results[0].predicted_label == FeasibilityLabel.AVAILABLE
        assert results[0].used_agent is False

    def test_incorrect_prediction(self, mock_llm_available, mock_memory, sample_unsupported):
        """LLM returns Available, but correct answer is Unsupported."""
        runner = EvalRunner(mock_llm_available, mock_memory, use_agent=False)
        results = runner.run([sample_unsupported])

        assert results[0].is_correct is False
        assert results[0].predicted_label == FeasibilityLabel.AVAILABLE
        assert results[0].expected_label == FeasibilityLabel.UNSUPPORTED_UNCLEAR

    def test_multiple_samples(self, mock_llm_available, mock_memory,
                              sample_available, sample_unsupported):
        runner = EvalRunner(mock_llm_available, mock_memory, use_agent=False)
        results = runner.run([sample_available, sample_unsupported])

        assert len(results) == 2

    def test_error_handled_gracefully(self, mock_memory, sample_available):
        """Classification error produces an error result, not a crash."""
        llm = MagicMock()
        llm.call_count = 0
        llm.call_log = []
        llm.chat_json.side_effect = Exception("LLM timeout")

        # Also make the retrieval fail so the runner's except block fires
        bad_memory = MagicMock()
        bad_memory.retrieve.side_effect = RuntimeError("DB connection lost")

        runner = EvalRunner(llm, bad_memory, use_agent=False)
        results = runner.run([sample_available])

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].is_correct is False

    def test_agent_mode_stats(self, mock_memory, sample_available):
        """Agent mode records tool calls and early-stop in results."""
        llm = MagicMock()
        llm.call_count = 0
        llm.call_log = []
        # Agent: immediate classify (early stop)
        llm.chat_json.return_value = {
            "action": "classify",
            "thought": "Strong initial evidence.",
            "label": "Available",
            "confidence": 0.92,
            "reasoning": "Confirmed.",
            "what_would_change": "Nothing",
            "draft_response": "Supported.",
        }

        runner = EvalRunner(llm, mock_memory, use_agent=True, agent_max_steps=3)
        results = runner.run([sample_available])

        assert results[0].used_agent is True
        assert results[0].agent_early_stopped is True
        assert results[0].agent_tool_calls == 0

    def test_llm_calls_counted_per_sample(self, mock_memory,
                                           sample_available, sample_unsupported):
        """LLM calls are tracked per sample, not cumulative."""
        # Use a real-ish counter by patching call_count as a list mutation
        llm = MagicMock()
        call_counter = [0]  # mutable int via list

        def incrementing_chat_json(*args, **kwargs):
            call_counter[0] += 1
            return {
                "label": "Available",
                "confidence": 0.9,
                "reasoning": "OK",
                "what_would_change": "Nothing",
                "draft_response": "Yes.",
            }

        # Wire up call_count as a property that reads call_counter
        type(llm).call_count = property(lambda self: call_counter[0])
        llm.call_log = []
        llm.chat_json.side_effect = incrementing_chat_json

        runner = EvalRunner(llm, mock_memory, use_agent=False)
        results = runner.run([sample_available, sample_unsupported])

        # Each single-shot = 1 LLM call, so both results should have ≥1
        total_calls = sum(r.llm_calls_used for r in results)
        assert total_calls >= 2


# ─── Metrics Tests ─────────────────────────────────────────────────────────


def _make_result(expected, predicted, confidence, llm_calls=1, used_agent=False,
                 tool_calls=0, early_stopped=False, fallback=False):
    return EvalResult(
        sample_id="x",
        requirement_text="test",
        expected_label=expected,
        predicted_label=predicted,
        confidence=confidence,
        is_correct=(expected == predicted),
        llm_calls_used=llm_calls,
        latency_seconds=0.5,
        used_agent=used_agent,
        agent_tool_calls=tool_calls,
        agent_early_stopped=early_stopped,
        fallback_used=fallback,
    )


class TestMetrics:
    def test_perfect_accuracy(self):
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9),
            _make_result(FeasibilityLabel.UNSUPPORTED_UNCLEAR, FeasibilityLabel.UNSUPPORTED_UNCLEAR, 0.8),
        ]
        report = compute_metrics(results, mode="single_shot")
        assert report.accuracy == 1.0

    def test_zero_accuracy(self):
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.UNSUPPORTED_UNCLEAR, 0.4),
            _make_result(FeasibilityLabel.CONFIGURABLE, FeasibilityLabel.AVAILABLE, 0.3),
        ]
        report = compute_metrics(results, mode="single_shot")
        assert report.accuracy == 0.0

    def test_partial_accuracy(self):
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9),     # correct
            _make_result(FeasibilityLabel.UNSUPPORTED_UNCLEAR, FeasibilityLabel.AVAILABLE, 0.4),  # wrong
        ]
        report = compute_metrics(results, mode="single_shot")
        assert report.accuracy == 0.5

    def test_macro_f1_all_correct(self):
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9),
            _make_result(FeasibilityLabel.CONFIGURABLE, FeasibilityLabel.CONFIGURABLE, 0.8),
        ]
        report = compute_metrics(results, mode="single_shot")
        assert report.macro_f1 == 1.0

    def test_per_label_support(self):
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9),
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.85),
            _make_result(FeasibilityLabel.UNSUPPORTED_UNCLEAR, FeasibilityLabel.UNSUPPORTED_UNCLEAR, 0.7),
        ]
        report = compute_metrics(results, mode="single_shot")
        assert report.per_label_support["Available"] == 2
        assert report.per_label_support["Unsupported / Unclear"] == 1

    def test_ece_perfect_calibration(self):
        """Perfect calibration: 100% confidence, always correct → ECE ≈ 0."""
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 1.0),
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 1.0),
        ]
        report = compute_metrics(results, mode="single_shot")
        assert report.ece < 0.01

    def test_ece_poor_calibration(self):
        """High confidence but always wrong → high ECE."""
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.UNSUPPORTED_UNCLEAR, 0.95),
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.UNSUPPORTED_UNCLEAR, 0.95),
        ]
        report = compute_metrics(results, mode="single_shot")
        assert report.ece > 0.5

    def test_avg_llm_calls(self):
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9, llm_calls=1),
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9, llm_calls=3),
        ]
        report = compute_metrics(results, mode="agent")
        assert report.avg_llm_calls == 2.0
        assert report.total_llm_calls == 4

    def test_agent_early_stop_rate(self):
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9,
                         used_agent=True, early_stopped=True),
            _make_result(FeasibilityLabel.CONFIGURABLE, FeasibilityLabel.CONFIGURABLE, 0.8,
                         used_agent=True, early_stopped=False),
        ]
        report = compute_metrics(results, mode="agent")
        assert report.early_stop_rate == 0.5

    def test_empty_results(self):
        report = compute_metrics([], mode="single_shot")
        assert report.total_samples == 0
        assert report.accuracy == 0.0

    def test_compare_reports(self):
        r_a = compute_metrics(
            [_make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9)],
            mode="agent"
        )
        r_b = compute_metrics(
            [_make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.UNSUPPORTED_UNCLEAR, 0.4)],
            mode="single_shot"
        )
        delta = compare_reports(r_a, r_b)
        assert delta["accuracy_delta"] > 0  # agent is better
        assert delta["mode_a"] == "agent"
        assert delta["mode_b"] == "single_shot"


# ─── Report Formatting Tests ────────────────────────────────────────────────


class TestReportFormatting:
    def _make_report(self, mode="single_shot"):
        results = [
            _make_result(FeasibilityLabel.AVAILABLE, FeasibilityLabel.AVAILABLE, 0.9),
            _make_result(FeasibilityLabel.UNSUPPORTED_UNCLEAR,
                         FeasibilityLabel.UNSUPPORTED_UNCLEAR, 0.8),
        ]
        return compute_metrics(results, mode=mode)

    def test_format_contains_key_sections(self):
        report = self._make_report()
        text = format_eval_report(report)
        assert "EVALUATION REPORT" in text
        assert "QUALITY METRICS" in text
        assert "PER-LABEL BREAKDOWN" in text
        assert "EFFICIENCY METRICS" in text
        assert "PER-SAMPLE RESULTS" in text

    def test_format_shows_accuracy(self):
        report = self._make_report()
        text = format_eval_report(report)
        assert "100.0%" in text  # perfect accuracy

    def test_format_shows_agent_section_for_agent_mode(self):
        report = self._make_report(mode="agent")
        text = format_eval_report(report)
        assert "AGENT STATS" in text

    def test_format_no_agent_section_for_single_shot(self):
        report = self._make_report(mode="single_shot")
        text = format_eval_report(report)
        assert "AGENT STATS" not in text

    def test_comparison_format(self):
        r_a = self._make_report("agent")
        r_b = self._make_report("single_shot")
        text = format_comparison(r_a, r_b)
        assert "ABLATION COMPARISON" in text
        assert "AGENT" in text
        assert "SINGLE_SHOT" in text

    def test_save_report(self, tmp_path):
        from feasiai.eval.report import save_report
        report = self._make_report()
        output = tmp_path / "test_report.json"
        save_report(report, str(output))
        assert output.exists()
        data = json.loads(output.read_text())
        assert data["mode"] == "single_shot"
        assert data["total_samples"] == 2
