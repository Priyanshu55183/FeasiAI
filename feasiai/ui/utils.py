"""
Utility functions for the Streamlit UI.

Handles report loading/saving, export formatting, and session state
management. Keeps the component modules focused on rendering.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Optional

from feasiai.schemas import (
    Classification,
    FeasibilityLabel,
    FeasibilityReport,
    Requirement,
)


# ─── Label styling ──────────────────────────────────────────────────────────

LABEL_COLORS = {
    FeasibilityLabel.AVAILABLE: "#22c55e",           # green
    FeasibilityLabel.CONFIGURABLE: "#3b82f6",        # blue
    FeasibilityLabel.REQUIRES_DEVELOPMENT: "#f59e0b", # amber
    FeasibilityLabel.UNSUPPORTED_UNCLEAR: "#ef4444",  # red
}

LABEL_ICONS = {
    FeasibilityLabel.AVAILABLE: "✅",
    FeasibilityLabel.CONFIGURABLE: "⚙️",
    FeasibilityLabel.REQUIRES_DEVELOPMENT: "🔨",
    FeasibilityLabel.UNSUPPORTED_UNCLEAR: "❌",
}

LABEL_OPTIONS = [label.value for label in FeasibilityLabel]


# ─── Report loading ─────────────────────────────────────────────────────────


def load_report_from_json(json_data: str | bytes | dict) -> FeasibilityReport:
    """
    Load a FeasibilityReport from JSON data.

    Handles both raw JSON strings and pre-parsed dicts.
    """
    if isinstance(json_data, (str, bytes)):
        data = json.loads(json_data)
    else:
        data = json_data

    # Reconstruct enums from string values
    requirements = [Requirement(**r) for r in data.get("requirements", [])]

    classifications = []
    for c in data.get("classifications", []):
        # Parse label enum
        label_str = c.get("label", "Unsupported / Unclear")
        label = _parse_label_from_string(label_str)

        # Parse evidence source enums
        evidence_list = []
        for ev in c.get("supporting_evidence", []):
            from feasiai.schemas import Evidence, EvidenceSource
            source_str = ev.get("source", "memory")
            try:
                source = EvidenceSource(source_str)
            except ValueError:
                source = EvidenceSource.MEMORY
            evidence_list.append(Evidence(**{**ev, "source": source}))

        # Parse agent trace if present
        agent_trace = None
        if c.get("agent_trace"):
            from feasiai.schemas import AgentTrace, AgentStep, AgentAction
            trace_data = c["agent_trace"]
            steps = []
            for s in trace_data.get("steps", []):
                action = None
                if s.get("action"):
                    action = AgentAction(**s["action"])
                steps.append(AgentStep(**{**s, "action": action}))
            agent_trace = AgentTrace(**{**trace_data, "steps": steps})

        classifications.append(
            Classification(
                **{
                    **{k: v for k, v in c.items()
                       if k not in ("label", "supporting_evidence", "agent_trace")},
                    "label": label,
                    "supporting_evidence": evidence_list,
                    "agent_trace": agent_trace,
                }
            )
        )

    return FeasibilityReport(
        id=data.get("id", ""),
        source_document=data.get("source_document"),
        timestamp=data.get("timestamp", datetime.now(timezone.utc)),
        requirements=requirements,
        classifications=classifications,
        summary=data.get("summary", {}),
        llm_call_count=data.get("llm_call_count", 0),
        total_latency_seconds=data.get("total_latency_seconds", 0.0),
    )


def _parse_label_from_string(label_str: str) -> FeasibilityLabel:
    """Parse a label string to enum, handling variations."""
    normalized = label_str.strip().lower()
    mapping = {
        "available": FeasibilityLabel.AVAILABLE,
        "configurable": FeasibilityLabel.CONFIGURABLE,
        "requires development": FeasibilityLabel.REQUIRES_DEVELOPMENT,
        "requires_development": FeasibilityLabel.REQUIRES_DEVELOPMENT,
        "unsupported / unclear": FeasibilityLabel.UNSUPPORTED_UNCLEAR,
        "unsupported/unclear": FeasibilityLabel.UNSUPPORTED_UNCLEAR,
        "unsupported": FeasibilityLabel.UNSUPPORTED_UNCLEAR,
        "unclear": FeasibilityLabel.UNSUPPORTED_UNCLEAR,
    }
    return mapping.get(normalized, FeasibilityLabel.UNSUPPORTED_UNCLEAR)


# ─── Export functions ────────────────────────────────────────────────────────


def export_report_json(
    report: FeasibilityReport,
    overrides: dict[str, str],
    notes: dict[str, str],
    approvals: Optional[dict[str, bool]] = None,
) -> str:
    """
    Export the report as JSON with reviewer overrides applied.

    Overrides replace the original label; notes are added as a new field.
    """
    approvals = approvals or {}
    data = json.loads(report.model_dump_json())

    # Apply overrides, notes, and approvals
    for classification in data.get("classifications", []):
        req_id = classification.get("requirement_id", "")
        if req_id in overrides and overrides[req_id]:
            classification["original_label"] = classification["label"]
            classification["label"] = overrides[req_id]
            classification["reviewer_overridden"] = True
        if req_id in notes and notes[req_id]:
            classification["reviewer_notes"] = notes[req_id]
        classification["approved_for_proposal"] = approvals.get(req_id, False)

    data["export_timestamp"] = datetime.now(timezone.utc).isoformat()
    data["has_reviewer_overrides"] = any(bool(v) for v in overrides.values())
    data["approved_count"] = sum(1 for v in approvals.values() if v)

    return json.dumps(data, indent=2, default=str)


def export_report_csv(
    report: FeasibilityReport,
    overrides: dict[str, str],
    notes: dict[str, str],
    approvals: Optional[dict[str, bool]] = None,
) -> str:
    """
    Export a flat CSV of all requirements with classifications.
    """
    req_map = {r.id: r for r in report.requirements}
    approvals = approvals or {}

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    writer.writerow([
        "Requirement #",
        "Section",
        "Requirement Text",
        "AI Label",
        "Reviewer Label",
        "Approved for Proposal",
        "Confidence",
        "Is Ambiguous",
        "Conflicts",
        "Reasoning",
        "What Would Change",
        "Draft Response",
        "Reviewer Notes",
        "Evidence Sources",
    ])

    for i, c in enumerate(report.classifications, 1):
        req = req_map.get(c.requirement_id)
        override_label = overrides.get(c.requirement_id, "")
        note = notes.get(c.requirement_id, "")
        approved = approvals.get(c.requirement_id, False)
        evidence_sources = ", ".join(sorted(set(
            e.source_document or "unknown"
            for e in c.supporting_evidence
        )))
        # Conflict summary
        conflict_summary = ""
        if c.conflicts:
            severities = [w.severity for w in c.conflicts]
            conflict_summary = f"{len(c.conflicts)} conflict(s): {', '.join(severities)}"

        writer.writerow([
            i,
            req.original_section if req else "",
            req.text if req else "Unknown",
            c.label.value,
            override_label if override_label else "",
            "Yes" if approved else "No",
            f"{c.confidence:.0%}",
            "Yes" if req and req.is_ambiguous else "No",
            conflict_summary,
            c.reasoning,
            c.what_would_change,
            c.draft_response,
            note,
            evidence_sources,
        ])

    return output.getvalue()


# ─── Session state helpers ───────────────────────────────────────────────────


def init_session_state(st_module: Any) -> None:
    """Initialize session state with defaults if not already set."""
    defaults = {
        "report": None,
        "overrides": {},       # {requirement_id: new_label_string}
        "notes": {},           # {requirement_id: reviewer_note}
        "approvals": {},       # {requirement_id: bool} — proposal-approved gate
        "filter_labels": [],   # label values to show
        "filter_min_conf": 0.0,
        "filter_max_conf": 1.0,
        "filter_ambiguous_only": False,
        "filter_conflicts_only": False,
        "filter_search": "",
        "selected_req_idx": None,
        "pipeline_running": False,
    }
    for key, default in defaults.items():
        if key not in st_module.session_state:
            st_module.session_state[key] = default

