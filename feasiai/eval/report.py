"""
Evaluation report formatting and persistence.

Saves eval reports as JSON and prints a human-readable summary.
Supports comparison mode (agent vs single-shot side by side).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from feasiai.eval.metrics import compare_reports
from feasiai.eval.schemas import EvalReport
from feasiai.schemas import FeasibilityLabel

logger = logging.getLogger(__name__)

_LABEL_ICONS = {
    "Available": "✅",
    "Configurable": "⚙️",
    "Requires Development": "🔨",
    "Unsupported / Unclear": "❌",
}


def format_eval_report(report: EvalReport) -> str:
    """Format an EvalReport as a human-readable console summary."""
    lines = []
    sep = "=" * 72
    thin = "─" * 72

    lines.append(sep)
    lines.append(f"  FeasiAI EVALUATION REPORT  [{report.mode.upper()}]")
    lines.append(sep)
    lines.append(f"  Model:     {report.model_name or 'N/A'}")
    lines.append(f"  Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"  Samples:   {report.total_samples}")
    lines.append("")

    # ── Core metrics ──────────────────────────────────────────────
    lines.append(thin)
    lines.append("  QUALITY METRICS")
    lines.append(thin)
    lines.append(f"  Accuracy:              {report.accuracy:.1%}")
    lines.append(f"  Macro F1:              {report.macro_f1:.1%}")
    lines.append(f"  Avg Confidence:        {report.avg_confidence:.1%}")
    lines.append(f"  ECE (calibration):     {report.ece:.4f}  {'✓ well-calibrated' if report.ece < 0.1 else '⚠ needs calibration'}")
    lines.append("")

    # ── Per-label breakdown ───────────────────────────────────────
    lines.append(thin)
    lines.append("  PER-LABEL BREAKDOWN")
    lines.append(thin)
    lines.append(f"  {'Label':<28} {'Support':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    lines.append(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for label in FeasibilityLabel:
        lv = label.value
        icon = _LABEL_ICONS.get(lv, "?")
        sup = report.per_label_support.get(lv, 0)
        if sup == 0:
            continue
        prec = report.per_label_precision.get(lv, 0.0)
        rec = report.per_label_recall.get(lv, 0.0)
        f1 = report.per_label_f1.get(lv, 0.0)
        lines.append(
            f"  {icon} {lv:<26} {sup:>7} {prec:>7.1%} {rec:>7.1%} {f1:>7.1%}"
        )
    lines.append("")

    # ── Efficiency metrics ────────────────────────────────────────
    lines.append(thin)
    lines.append("  EFFICIENCY METRICS")
    lines.append(thin)
    lines.append(f"  Avg LLM calls/req:     {report.avg_llm_calls:.1f}")
    lines.append(f"  Total LLM calls:       {report.total_llm_calls}")
    lines.append(f"  Avg latency/req:       {report.avg_latency_seconds:.2f}s")
    lines.append(f"  Total latency:         {report.total_latency_seconds:.1f}s")
    lines.append("")

    # ── Agent stats (if applicable) ───────────────────────────────
    if report.mode == "agent":
        lines.append(thin)
        lines.append("  AGENT STATS")
        lines.append(thin)
        lines.append(f"  Avg tool calls/req:    {report.avg_tool_calls:.1f}")
        lines.append(f"  Early-stop rate:       {report.early_stop_rate:.1%}")
        lines.append(f"  Fallback rate:         {report.fallback_rate:.1%}")
        lines.append("")

    # ── Per-sample results ────────────────────────────────────────
    lines.append(thin)
    lines.append("  PER-SAMPLE RESULTS")
    lines.append(thin)
    for i, r in enumerate(report.results, 1):
        status = "✓" if r.is_correct else "✗"
        req_preview = r.requirement_text[:55]
        if len(r.requirement_text) > 55:
            req_preview += "..."
        lines.append(
            f"  {status} [{i:>2}] {req_preview}"
        )
        if not r.is_correct:
            lines.append(
                f"       Expected: {r.expected_label.value} | "
                f"Got: {r.predicted_label.value} | conf={r.confidence:.0%}"
            )
        if r.error:
            lines.append(f"       ERROR: {r.error}")

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def format_comparison(report_a: EvalReport, report_b: EvalReport) -> str:
    """Format a side-by-side comparison of two eval reports."""
    delta = compare_reports(report_a, report_b)
    lines = []
    sep = "=" * 72
    thin = "─" * 72

    lines.append(sep)
    lines.append(f"  ABLATION COMPARISON: {report_a.mode.upper()} vs {report_b.mode.upper()}")
    lines.append(sep)
    lines.append(f"  {'Metric':<30} {report_a.mode:>14} {report_b.mode:>14}  {'Δ':>8}")
    lines.append(f"  {'-'*30} {'-'*14} {'-'*14}  {'-'*8}")

    def row(name, val_a, val_b, delta_v, fmt=".1%", higher_better=True):
        fa = format(val_a, fmt)
        fb = format(val_b, fmt)
        fd = f"{delta_v:+.4f}"
        arrow = "↑" if (delta_v > 0) == higher_better else "↓"
        indicator = f"{fd} {arrow}" if abs(delta_v) > 0.001 else "  —"
        lines.append(f"  {name:<30} {fa:>14} {fb:>14}  {indicator:>9}")

    row("Accuracy",        report_a.accuracy,              report_b.accuracy,              delta["accuracy_delta"])
    row("Macro F1",        report_a.macro_f1,              report_b.macro_f1,              delta["macro_f1_delta"])
    row("ECE (↓ better)",  report_a.ece,                   report_b.ece,                   delta["ece_delta"],         fmt=".4f", higher_better=True)
    row("Avg LLM calls",   report_a.avg_llm_calls,         report_b.avg_llm_calls,         delta["avg_llm_calls_delta"], fmt=".1f", higher_better=False)
    row("Avg latency (s)", report_a.avg_latency_seconds,   report_b.avg_latency_seconds,   delta["avg_latency_delta"],   fmt=".2f", higher_better=False)

    lines.append("")
    lines.append(sep)
    return "\n".join(lines)


def save_report(report: EvalReport, output_path: str) -> None:
    """Save an EvalReport to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json.loads(report.model_dump_json()), f, indent=2, default=str)
    logger.info("Eval report saved to: %s", path)

