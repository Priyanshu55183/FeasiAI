"""
Structured report assembly and formatting.

Takes the pipeline's outputs (requirements + classifications) and
produces a FeasibilityReport with aggregate statistics and a
human-readable text representation.
"""

from __future__ import annotations

import logging
from collections import Counter

from feasiai.schemas import (
    Classification,
    FeasibilityLabel,
    FeasibilityReport,
    Requirement,
)

logger = logging.getLogger(__name__)


def build_report(
    requirements: list[Requirement],
    classifications: list[Classification],
    source_document: str | None = None,
    llm_call_count: int = 0,
    total_latency_seconds: float = 0.0,
) -> FeasibilityReport:
    """
    Assemble a complete feasibility report from pipeline outputs.

    Computes aggregate statistics: counts per label, average confidence,
    ambiguous requirement count, etc.
    """
    # Compute summary statistics
    label_counts = Counter(c.label.value for c in classifications)
    confidences = [c.confidence for c in classifications]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    ambiguous_count = sum(1 for r in requirements if r.is_ambiguous)

    # Low-confidence items that need human attention
    low_confidence_count = sum(1 for c in confidences if c < 0.5)

    summary = {
        "total_requirements": len(requirements),
        "label_distribution": dict(label_counts),
        "average_confidence": round(avg_confidence, 3),
        "ambiguous_requirements": ambiguous_count,
        "low_confidence_items": low_confidence_count,
        "available_count": label_counts.get(FeasibilityLabel.AVAILABLE.value, 0),
        "configurable_count": label_counts.get(FeasibilityLabel.CONFIGURABLE.value, 0),
        "requires_dev_count": label_counts.get(
            FeasibilityLabel.REQUIRES_DEVELOPMENT.value, 0
        ),
        "unsupported_count": label_counts.get(
            FeasibilityLabel.UNSUPPORTED_UNCLEAR.value, 0
        ),
    }

    return FeasibilityReport(
        source_document=source_document,
        requirements=requirements,
        classifications=classifications,
        summary=summary,
        llm_call_count=llm_call_count,
        total_latency_seconds=round(total_latency_seconds, 2),
    )


def format_report_text(report: FeasibilityReport) -> str:
    """
    Format a FeasibilityReport as a human-readable text report.
    Suitable for printing to console or saving as a text file.
    """
    lines = []
    lines.append("=" * 80)
    lines.append("  FEASIBILITY ASSESSMENT REPORT")
    lines.append("=" * 80)
    lines.append(f"  Document: {report.source_document or 'N/A'}")
    lines.append(f"  Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"  Report ID: {report.id[:12]}...")
    lines.append("")

    # Summary section
    s = report.summary
    lines.append("─" * 80)
    lines.append("  SUMMARY")
    lines.append("─" * 80)
    lines.append(f"  Total requirements:      {s.get('total_requirements', 0)}")
    lines.append(f"  ✅ Available:             {s.get('available_count', 0)}")
    lines.append(f"  ⚙️  Configurable:          {s.get('configurable_count', 0)}")
    lines.append(f"  🔨 Requires Development:  {s.get('requires_dev_count', 0)}")
    lines.append(f"  ❌ Unsupported / Unclear:  {s.get('unsupported_count', 0)}")
    lines.append(f"  Average confidence:       {s.get('average_confidence', 0):.1%}")
    lines.append(f"  Ambiguous requirements:   {s.get('ambiguous_requirements', 0)}")
    lines.append(f"  Low-confidence items:     {s.get('low_confidence_items', 0)}")
    lines.append(f"  LLM calls used:           {report.llm_call_count}")
    lines.append(f"  Pipeline latency:         {report.total_latency_seconds:.1f}s")
    lines.append("")

    # Build requirement lookup
    req_map = {r.id: r for r in report.requirements}

    # Per-requirement details
    lines.append("─" * 80)
    lines.append("  DETAILED ASSESSMENTS")
    lines.append("─" * 80)

    label_icons = {
        FeasibilityLabel.AVAILABLE: "✅",
        FeasibilityLabel.CONFIGURABLE: "⚙️",
        FeasibilityLabel.REQUIRES_DEVELOPMENT: "🔨",
        FeasibilityLabel.UNSUPPORTED_UNCLEAR: "❌",
    }

    for i, classification in enumerate(report.classifications, 1):
        req = req_map.get(classification.requirement_id)
        req_text = req.text if req else "Unknown requirement"
        icon = label_icons.get(classification.label, "?")
        ambig = " ⚠️ [AMBIGUOUS]" if req and req.is_ambiguous else ""

        lines.append("")
        lines.append(f"  [{i}] {req_text}{ambig}")
        lines.append(f"      Label:      {icon} {classification.label.value}")
        lines.append(f"      Confidence: {classification.confidence:.0%}")

        if classification.reasoning:
            # Truncate long reasoning for readability
            reasoning = classification.reasoning[:300]
            if len(classification.reasoning) > 300:
                reasoning += "..."
            lines.append(f"      Reasoning:  {reasoning}")

        if classification.what_would_change:
            change = classification.what_would_change[:200]
            if len(classification.what_would_change) > 200:
                change += "..."
            lines.append(f"      Would change if: {change}")

        if classification.draft_response:
            draft = classification.draft_response[:250]
            if len(classification.draft_response) > 250:
                draft += "..."
            lines.append(f"      Draft response: {draft}")

        # Evidence sources
        if classification.supporting_evidence:
            sources = set(
                e.source_document or "unknown"
                for e in classification.supporting_evidence
            )
            lines.append(f"      Evidence from: {', '.join(sorted(sources))}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("  ⚠️  HUMAN REVIEW REQUIRED — This report requires sign-off")
    lines.append("     before any commitments are made to the client.")
    lines.append("=" * 80)

    return "\n".join(lines)
