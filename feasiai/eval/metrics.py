"""
Evaluation metrics: accuracy, macro-F1, ECE, and efficiency stats.

Design decisions:
1. MACRO-F1: We use macro averaging (equal weight per class) rather than
   micro/weighted, because we care equally about all four label classes
   regardless of their frequency in the dataset.

2. EXPECTED CALIBRATION ERROR (ECE): Measures whether confidence scores
   are reliable. A well-calibrated system with 80% confidence should be
   correct ~80% of the time. ECE is computed with 10 equal-width bins.

3. PER-LABEL BREAKDOWN: Precision, recall, and F1 per label surface
   exactly which capabilities the system is mis-classifying.

4. EFFICIENCY METRICS: LLM call count and latency are first-class
   metrics because cost and speed matter in production.
"""

from __future__ import annotations

import math
from collections import defaultdict

from feasiai.eval.schemas import EvalReport, EvalResult
from feasiai.schemas import FeasibilityLabel

_ALL_LABELS = [label.value for label in FeasibilityLabel]
_N_ECE_BINS = 10


def compute_metrics(results: list[EvalResult], mode: str, model_name: str = "") -> EvalReport:
    """
    Compute all evaluation metrics from a list of EvalResults.

    Args:
        results: Per-sample results from the eval runner.
        mode: 'agent' or 'single_shot'
        model_name: LLM model identifier.

    Returns:
        Fully populated EvalReport.
    """
    if not results:
        return EvalReport(mode=mode, model_name=model_name, total_samples=0, results=[])

    total = len(results)

    # ── Basic accuracy ──────────────────────────────────────────────
    correct = sum(1 for r in results if r.is_correct)
    accuracy = correct / total

    # ── Per-label precision / recall / F1 ──────────────────────────
    # Confusion matrix: {true_label: {pred_label: count}}
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)

    for r in results:
        expected = r.expected_label.value
        predicted = r.predicted_label.value
        support[expected] += 1

        if expected == predicted:
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1

    per_label_precision: dict[str, float] = {}
    per_label_recall: dict[str, float] = {}
    per_label_f1: dict[str, float] = {}

    for label in _ALL_LABELS:
        p_denom = tp[label] + fp[label]
        r_denom = tp[label] + fn[label]
        prec = tp[label] / p_denom if p_denom > 0 else 0.0
        rec = tp[label] / r_denom if r_denom > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        per_label_precision[label] = round(prec, 4)
        per_label_recall[label] = round(rec, 4)
        per_label_f1[label] = round(f1, 4)

    # Macro F1: average over labels that have support
    supported_labels = [l for l in _ALL_LABELS if support[l] > 0]
    macro_f1 = (
        sum(per_label_f1[l] for l in supported_labels) / len(supported_labels)
        if supported_labels else 0.0
    )

    # ── ECE ─────────────────────────────────────────────────────────
    ece = _compute_ece(results)

    # ── Efficiency stats ─────────────────────────────────────────────
    total_llm_calls = sum(r.llm_calls_used for r in results)
    total_latency = sum(r.latency_seconds for r in results)
    avg_confidence = sum(r.confidence for r in results) / total

    # ── Agent-specific stats ─────────────────────────────────────────
    agent_results = [r for r in results if r.used_agent]
    early_stop_rate = (
        sum(1 for r in agent_results if r.agent_early_stopped) / len(agent_results)
        if agent_results else 0.0
    )
    fallback_rate = (
        sum(1 for r in agent_results if r.fallback_used) / len(agent_results)
        if agent_results else 0.0
    )
    avg_tool_calls = (
        sum(r.agent_tool_calls for r in agent_results) / len(agent_results)
        if agent_results else 0.0
    )

    return EvalReport(
        mode=mode,
        model_name=model_name,
        total_samples=total,
        results=results,
        accuracy=round(accuracy, 4),
        macro_f1=round(macro_f1, 4),
        ece=round(ece, 4),
        avg_confidence=round(avg_confidence, 4),
        avg_llm_calls=round(total_llm_calls / total, 2),
        avg_latency_seconds=round(total_latency / total, 2),
        total_llm_calls=total_llm_calls,
        total_latency_seconds=round(total_latency, 2),
        per_label_precision=per_label_precision,
        per_label_recall=per_label_recall,
        per_label_f1=per_label_f1,
        per_label_support=dict(support),
        early_stop_rate=round(early_stop_rate, 4),
        fallback_rate=round(fallback_rate, 4),
        avg_tool_calls=round(avg_tool_calls, 2),
    )


def _compute_ece(results: list[EvalResult], n_bins: int = _N_ECE_BINS) -> float:
    """
    Expected Calibration Error using equal-width confidence bins.

    ECE = Σ (|bin| / N) * |accuracy(bin) - confidence(bin)|

    Lower is better. A perfectly calibrated system has ECE = 0.
    """
    bin_size = 1.0 / n_bins
    bins: list[list[EvalResult]] = [[] for _ in range(n_bins)]

    for r in results:
        bin_idx = min(int(r.confidence / bin_size), n_bins - 1)
        bins[bin_idx].append(r)

    total = len(results)
    ece = 0.0

    for bin_items in bins:
        if not bin_items:
            continue
        bin_acc = sum(1 for r in bin_items if r.is_correct) / len(bin_items)
        bin_conf = sum(r.confidence for r in bin_items) / len(bin_items)
        ece += (len(bin_items) / total) * abs(bin_acc - bin_conf)

    return ece


def compare_reports(report_a: EvalReport, report_b: EvalReport) -> dict:
    """
    Compare two EvalReports (e.g. agent vs single-shot).

    Returns a dict of metric deltas: positive = report_a is better.
    """
    return {
        "accuracy_delta": round(report_a.accuracy - report_b.accuracy, 4),
        "macro_f1_delta": round(report_a.macro_f1 - report_b.macro_f1, 4),
        "ece_delta": round(report_b.ece - report_a.ece, 4),  # lower ECE is better, so flip
        "avg_llm_calls_delta": round(report_a.avg_llm_calls - report_b.avg_llm_calls, 2),
        "avg_latency_delta": round(report_a.avg_latency_seconds - report_b.avg_latency_seconds, 2),
        "mode_a": report_a.mode,
        "mode_b": report_b.mode,
    }

