"""
Requirement detail component: evidence viewer, agent trace,
label overrides, and reviewer notes.
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

        # Flags for the header
        flags = []
        if req.is_ambiguous:
            flags.append("⚠️ Ambiguous")
        if conf < 0.5:
            flags.append("🔍 Low confidence")
        if classification.requirement_id in st.session_state.overrides and st.session_state.overrides[classification.requirement_id]:
            flags.append("📝 Overridden")
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

    # ── Requirement info ─────────────────────────────────────────────
    st.markdown(f"**Requirement:** {req.text}")

    if req.original_section:
        st.markdown(f"**Section:** `{req.original_section}`")

    if req.is_ambiguous:
        st.warning(
            f"⚠️ **Ambiguous requirement** — {req.ambiguity_reason or 'No reason given'}",
            icon="⚠️",
        )

    st.divider()

    # ── Classification + Override ────────────────────────────────────
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

        # Find index for selectbox
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

    # ── Reviewer notes ───────────────────────────────────────────────
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

    # ── Reasoning tabs ───────────────────────────────────────────────
    tabs = st.tabs(["💭 Reasoning", "📄 Evidence", "🤖 Agent Trace", "✏️ Draft Response"])

    with tabs[0]:
        _render_reasoning(classification)

    with tabs[1]:
        _render_evidence(classification)

    with tabs[2]:
        _render_agent_trace(classification)

    with tabs[3]:
        _render_draft_response(classification)


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


def _render_evidence(classification: Classification) -> None:
    """Render the supporting evidence chunks."""
    evidences = classification.supporting_evidence

    if not evidences:
        st.info("No evidence was retrieved for this requirement.")
        return

    st.markdown(f"**{len(evidences)} evidence chunk(s):**")

    for j, ev in enumerate(evidences):
        source = ev.source_document or "unknown"
        score = ev.relevance_score

        # Color-code relevance
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


def _render_agent_trace(classification: Classification) -> None:
    """Render the agent's step-by-step reasoning trace."""
    trace = classification.agent_trace

    if trace is None:
        st.info("No agent trace — this was classified using single-shot mode (Phase 1).")
        return

    # Summary
    if trace.fallback_used:
        st.warning("⚡ **Fallback to single-shot** — the agent encountered an error and fell back to Phase 1 mode.")
        return

    if trace.early_stopped:
        st.success("⚡ **Early stop** — the agent was confident enough on initial evidence (no tool calls needed).")

    # Stats
    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.metric("Tool Calls", trace.total_tool_calls)
    with stat_cols[1]:
        st.metric("LLM Calls", trace.total_llm_calls)
    with stat_cols[2]:
        st.metric("Steps", len(trace.steps))

    # Step timeline
    if trace.steps:
        st.markdown("**Agent reasoning trace:**")
        for step in trace.steps:
            if step.action:
                # Tool call step
                st.markdown(
                    f"**Step {step.step_number}: 🔧 Tool Call**\n\n"
                    f"- 💭 *Thought:* {step.thought}\n"
                    f"- 🛠️ `{step.action.tool}(\"{step.action.query}\")`\n"
                    f"- 📊 → {step.action.result_count} result(s)"
                )
            else:
                # Classification step
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
    else:
        st.info("No draft response generated.")

