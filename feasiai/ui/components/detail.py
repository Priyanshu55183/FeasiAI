"""
Requirement detail component: evidence viewer, conflict panel,
agent trace, label overrides, approval gate, and reviewer notes.

Design decisions:
1. CONFLICT PANEL: Shows each ConflictWarning with severity badge and
   description. Always visible — never collapsed by default when conflicts exist.
2. APPROVAL GATE: A prominent "Approve for Proposal" button that must be
   clicked explicitly. Approval is blocked if HIGH-severity conflicts exist
   and have not been acknowledged by the reviewer.
3. TAB ORDER: Evidence → Conflicts → Reasoning → Agent Trace → Draft Response.
   Conflicts tab is always shown; badge indicates count.
"""

from __future__ import annotations

import streamlit as st

from feasiai.schemas import (
    Classification,
    FeasibilityLabel,
    FeasibilityReport,
    Requirement,
)
from feasiai.ui.utils import LABEL_COLORS, LABEL_ICONS, LABEL_OPTIONS, init_session_state

_CONFLICT_SEVERITY_COLORS = {
    "high": "#ef4444",
    "medium": "#f59e0b",
    "low": "#6b7280",
}

_CONFLICT_SEVERITY_ICONS = {
    "high": "🔴",
    "medium": "🟡",
    "low": "⚪",
}


def render_detail(report: FeasibilityReport) -> None:
    """Render expandable detail views for each requirement."""
    init_session_state(st)

    req_map = {r.id: r for r in report.requirements}

    for i, classification in enumerate(report.classifications):
        req = req_map.get(classification.requirement_id)
        if not req:
            continue

        label = classification.label
        icon = LABEL_ICONS[label]
        conf = classification.confidence

        flags = []
        if req.is_ambiguous:
            flags.append("⚠️ Ambiguous")
        if conf < 0.5:
            flags.append("🔍 Low confidence")
        if classification.requirement_id in st.session_state.overrides and st.session_state.overrides[classification.requirement_id]:
            flags.append("📝 Overridden")
        if classification.conflicts:
            worst = max(
                classification.conflicts,
                key=lambda w: {"high": 2, "medium": 1, "low": 0}.get(w.severity, 0)
            )
            clr = _CONFLICT_SEVERITY_COLORS.get(worst.severity, "#6b7280")
            flags.append(f"⚡ {len(classification.conflicts)} conflict(s)")
        if st.session_state.approvals.get(classification.requirement_id, False):
            flags.append("✅ Approved")

        flag_str = f" — {', '.join(flags)}" if flags else ""
        header = f"{icon} [{i+1}] {req.text[:80]}{'...' if len(req.text) > 80 else ''}{flag_str}"

        with st.expander(header, expanded=False):
            _render_requirement_detail(i, req, classification)


def _render_requirement_detail(
    idx: int,
    req: Requirement,
    classification: Classification,
) -> None:
    """Render the full detail view for a single requirement."""
    req_id = classification.requirement_id

    # ── Requirement info ──────────────────────────────────────────────
    st.markdown(f"**Requirement:** {req.text}")

    if req.original_section:
        st.markdown(f"**Section:** `{req.original_section}`")

    if req.is_ambiguous:
        st.warning(
            f"⚠️ **Ambiguous requirement** — {req.ambiguity_reason or 'No reason given'}",
            icon="⚠️",
        )

    st.divider()

    # ── Classification + Override ─────────────────────────────────────
    class_col, override_col = st.columns(2)

    with class_col:
        st.markdown("##### AI Classification")
        label = classification.label
        color = LABEL_COLORS[label]
        icon = LABEL_ICONS[label]
        st.markdown(
            f'<div style="background-color:{color}; color:white; '
            f'padding:12px 16px; border-radius:8px; font-size:1.1rem; '
            f'font-weight:700; display:inline-block;">'
            f'{icon} {label.value} — {classification.confidence:.0%} confidence'
            f'</div>',
            unsafe_allow_html=True,
        )

    with override_col:
        st.markdown("##### Reviewer Override")
        current_override = st.session_state.overrides.get(req_id, "")

        options = ["(No override)"] + LABEL_OPTIONS
        default_idx = 0
        if current_override in LABEL_OPTIONS:
            default_idx = LABEL_OPTIONS.index(current_override) + 1

        new_override = st.selectbox(
            "Override label",
            options=options,
            index=default_idx,
            key=f"override_{req_id}",
            label_visibility="collapsed",
        )

        if new_override == "(No override)":
            st.session_state.overrides[req_id] = ""
        else:
            st.session_state.overrides[req_id] = new_override

    # ── Reviewer notes ────────────────────────────────────────────────
    current_note = st.session_state.notes.get(req_id, "")
    new_note = st.text_area(
        "Reviewer notes",
        value=current_note,
        key=f"note_{req_id}",
        placeholder="Add your notes here...",
        height=68,
    )
    st.session_state.notes[req_id] = new_note

    st.divider()

    # ── APPROVAL GATE ─────────────────────────────────────────────────
    _render_approval_gate(req_id, classification)

    st.divider()

    # ── Tabs: Evidence | Conflicts | Reasoning | Agent Trace | Draft ──
    conflict_tab_label = (
        f"⚡ Conflicts ({len(classification.conflicts)})"
        if classification.conflicts
        else "⚡ Conflicts"
    )
    tabs = st.tabs([
        "📄 Evidence",
        conflict_tab_label,
        "💭 Reasoning",
        "🤖 Agent Trace",
        "✏️ Draft Response",
    ])

    with tabs[0]:
        _render_evidence(classification)

    with tabs[1]:
        _render_conflicts(classification)

    with tabs[2]:
        _render_reasoning(classification)

    with tabs[3]:
        _render_agent_trace(classification)

    with tabs[4]:
        _render_draft_response(classification)


def _render_approval_gate(req_id: str, classification: Classification) -> None:
    """
    Render the proposal approval gate with conflict-awareness.

    HIGH-severity conflicts display a warning before allowing approval.
    """
    approved = st.session_state.approvals.get(req_id, False)
    high_conflicts = [w for w in classification.conflicts if w.severity == "high"]

    gate_cols = st.columns([2, 3])

    with gate_cols[0]:
        if high_conflicts and not approved:
            st.warning(
                f"⚡ **{len(high_conflicts)} HIGH-severity conflict(s) detected.** "
                "Review the Conflicts tab before approving.",
                icon="⚠️",
            )

        if approved:
            st.success("✅ **Approved for proposal**", icon="✅")
            if st.button(
                "Revoke Approval",
                key=f"revoke_{req_id}",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state.approvals[req_id] = False
                st.rerun()
        else:
            button_label = (
                "⚠️ Approve Despite Conflicts" if high_conflicts
                else "✅ Approve for Proposal"
            )
            button_type = "secondary" if high_conflicts else "primary"
            if st.button(
                button_label,
                key=f"approve_detail_{req_id}",
                type=button_type,
                use_container_width=True,
                help="Mark this requirement as proposal-ready",
            ):
                st.session_state.approvals[req_id] = True
                st.rerun()

    with gate_cols[1]:
        if approved:
            final_label = (
                st.session_state.overrides.get(req_id)
                or classification.label.value
            )
            st.info(
                f"This requirement will appear in the proposal as: **{final_label}**"
            )


def _render_conflicts(classification: Classification) -> None:
    """Render detected conflicts between memory and live data."""
    if not classification.conflicts:
        st.success(
            "✅ **No conflicts detected** between institutional memory and live data.",
            icon="✅",
        )
        return

    st.markdown(f"**{len(classification.conflicts)} conflict(s) detected:**")
    st.caption(
        "These conflicts mean live operational data contradicts "
        "what institutional memory says about this capability."
    )

    for i, warning in enumerate(classification.conflicts, 1):
        severity = warning.severity
        sev_color = _CONFLICT_SEVERITY_COLORS.get(severity, "#6b7280")
        sev_icon = _CONFLICT_SEVERITY_ICONS.get(severity, "⚪")

        st.markdown(
            f'<div style="border-left: 4px solid {sev_color}; '
            f'padding: 10px 16px; margin: 8px 0; background: #f9fafb; border-radius: 4px;">'
            f'<div style="font-weight:700; color:{sev_color};">'
            f'{sev_icon} {severity.upper()} SEVERITY</div>'
            f'<div style="margin-top:6px;">{warning.description}</div>'
            f'{"<div style=\\'margin-top:4px; font-size:0.85rem; color:#6b7280;\\'>Live status: <code>" + warning.live_status + "</code></div>" if warning.live_status else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )


def _render_evidence(classification: Classification) -> None:
    """Render supporting evidence chunks grouped by source type."""
    evidences = classification.supporting_evidence

    if not evidences:
        st.info("No evidence was retrieved for this requirement.")
        return

    # Group by source type
    memory_evs = [e for e in evidences if e.source.value == "memory"]
    live_evs = [e for e in evidences if e.source.value == "live_data"]
    other_evs = [e for e in evidences if e.source.value not in ("memory", "live_data")]

    st.markdown(f"**{len(evidences)} evidence chunk(s):**")

    if memory_evs:
        st.markdown("🧠 **Institutional Memory**")
        for ev in memory_evs:
            _render_evidence_item(ev)

    if live_evs:
        st.markdown("🔴 **Live Data**")
        for ev in live_evs:
            _render_evidence_item(ev)

    if other_evs:
        st.markdown("📄 **Other**")
        for ev in other_evs:
            _render_evidence_item(ev)


def _render_evidence_item(ev) -> None:
    """Render one evidence chunk."""
    source = ev.source_document or "unknown"
    score = ev.relevance_score

    if score >= 0.8:
        score_color = "#22c55e"
    elif score >= 0.5:
        score_color = "#f59e0b"
    else:
        score_color = "#ef4444"

    with st.container():
        ev_cols = st.columns([3, 1])
        with ev_cols[0]:
            st.markdown(f"**📄 {source}**")
        with ev_cols[1]:
            st.markdown(
                f'<span style="color:{score_color}; font-weight:600;">'
                f'Relevance: {score:.0%}</span>',
                unsafe_allow_html=True,
            )
        st.code(ev.content, language=None)


def _render_reasoning(classification: Classification) -> None:
    """Render the AI's reasoning and what-would-change."""
    if classification.reasoning:
        st.markdown("**Reasoning:**")
        st.markdown(classification.reasoning)
    else:
        st.info("No reasoning provided.")

    if classification.what_would_change:
        st.markdown("**What would change this assessment:**")
        st.info(classification.what_would_change)


def _render_agent_trace(classification: Classification) -> None:
    """Render the agent's step-by-step reasoning trace."""
    trace = classification.agent_trace

    if trace is None:
        st.info("No agent trace — this was classified using single-shot mode (Phase 1).")
        return

    if trace.fallback_used:
        st.warning("⚡ **Fallback to single-shot** — the agent encountered an error and fell back to Phase 1 mode.")
        return

    if trace.early_stopped:
        st.success("⚡ **Early stop** — the agent was confident enough on initial evidence (no tool calls needed).")

    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.metric("Tool Calls", trace.total_tool_calls)
    with stat_cols[1]:
        st.metric("LLM Calls", trace.total_llm_calls)
    with stat_cols[2]:
        st.metric("Steps", len(trace.steps))

    if trace.steps:
        st.markdown("**Agent reasoning trace:**")
        for step in trace.steps:
            if step.action:
                st.markdown(
                    f"**Step {step.step_number}: 🔧 Tool Call**\n\n"
                    f"- 💭 *Thought:* {step.thought}\n"
                    f"- 🛠️ `{step.action.tool}(\"{step.action.query}\")`\n"
                    f"- 📊 → {step.action.result_count} result(s)"
                )
            else:
                st.markdown(
                    f"**Step {step.step_number}: ✅ Classify**\n\n"
                    f"- 💭 *Thought:* {step.thought}"
                )
            st.divider()


def _render_draft_response(classification: Classification) -> None:
    """Render the draft client-facing response."""
    if classification.draft_response:
        st.markdown("**Draft client-facing response:**")
        st.info(classification.draft_response)
        if st.button(
            "📋 Copy to clipboard",
            key=f"copy_{classification.requirement_id}",
        ):
            st.toast("Draft response copied! (use Ctrl+C on the text above)")
    else:
        st.info("No draft response generated.")
