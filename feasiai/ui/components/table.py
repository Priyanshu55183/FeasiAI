"""
Requirements table component with filtering, conflict badges, and approval status.

Design decisions:
1. CONFLICT BADGE: Inline HTML badge shows worst-severity conflict per row.
   Color-coded: red=high, amber=medium, gray=low. Never silent.
2. APPROVAL CHECKBOX: One checkbox per row lets reviewers gate which
   requirements are proposal-ready. State stored in session_state.approvals.
3. SUMMARY BAR: Shows approved count vs total to track review progress.
4. FILTER: Added "Has conflicts" filter to focus attention on problem items.
"""

from __future__ import annotations

import streamlit as st

from feasiai.schemas import FeasibilityLabel, FeasibilityReport
from feasiai.ui.utils import LABEL_COLORS, LABEL_ICONS, LABEL_OPTIONS, init_session_state

# Conflict severity badge colours
_CONFLICT_COLORS = {
    "high": "#ef4444",
    "medium": "#f59e0b",
    "low": "#6b7280",
}


def render_table(report: FeasibilityReport) -> None:
    """Render the filterable requirements table."""
    init_session_state(st)

    req_map = {r.id: r for r in report.requirements}

    # ── Filters ──────────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=False):
        filter_cols = st.columns([2, 1, 1, 1, 2])

        with filter_cols[0]:
            selected_labels = st.multiselect(
                "Filter by label",
                options=LABEL_OPTIONS,
                default=[],
                placeholder="All labels",
            )

        with filter_cols[1]:
            min_conf = st.number_input(
                "Min confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                format="%.1f",
            )

        with filter_cols[2]:
            ambiguous_only = st.checkbox("Ambiguous only")

        with filter_cols[3]:
            conflicts_only = st.checkbox("Has conflicts")

        with filter_cols[4]:
            search_text = st.text_input(
                "Search requirements",
                placeholder="Type to search...",
            )

    # ── Apply filters ────────────────────────────────────────────────────
    filtered_items = []
    for i, c in enumerate(report.classifications):
        req = req_map.get(c.requirement_id)
        if not req:
            continue

        if selected_labels and c.label.value not in selected_labels:
            continue
        if c.confidence < min_conf:
            continue
        if ambiguous_only and not req.is_ambiguous:
            continue
        if conflicts_only and not c.conflicts:
            continue
        if search_text:
            search_lower = search_text.lower()
            if (
                search_lower not in req.text.lower()
                and search_lower not in (req.original_section or "").lower()
                and search_lower not in c.reasoning.lower()
            ):
                continue

        filtered_items.append((i, req, c))

    # ── Summary bar ──────────────────────────────────────────────────────
    total = len(report.classifications)
    shown = len(filtered_items)
    approved_total = sum(
        1 for c in report.classifications
        if st.session_state.approvals.get(c.requirement_id, False)
    )

    summary_cols = st.columns([2, 2, 2])
    with summary_cols[0]:
        caption = f"Showing {shown} of {total} requirements" if shown < total else f"{total} requirements"
        st.caption(caption)
    with summary_cols[1]:
        conflict_total = sum(1 for c in report.classifications if c.conflicts)
        if conflict_total:
            st.caption(f"⚡ {conflict_total} requirement(s) have conflicts")
    with summary_cols[2]:
        st.caption(f"✅ {approved_total}/{total} approved for proposal")

    # ── Column headers ───────────────────────────────────────────────────
    if filtered_items:
        hdr = st.columns([0.4, 3.5, 1.2, 1, 0.5, 0.5])
        with hdr[0]: st.caption("#")
        with hdr[1]: st.caption("Requirement")
        with hdr[2]: st.caption("Label")
        with hdr[3]: st.caption("Confidence")
        with hdr[4]: st.caption("⚠️")
        with hdr[5]: st.caption("✅")

    # ── Render table ─────────────────────────────────────────────────────
    if not filtered_items:
        st.info("No requirements match the current filters.")
        return

    for idx, req, classification in filtered_items:
        _render_row(idx, req, classification)


def _render_row(idx: int, req, classification) -> None:
    """Render a single requirement row with conflict badge and approval toggle."""
    label = classification.label
    icon = LABEL_ICONS[label]
    color = LABEL_COLORS[label]
    conf = classification.confidence
    req_id = classification.requirement_id
    override = st.session_state.overrides.get(req_id, "")
    approved = st.session_state.approvals.get(req_id, False)

    needs_attention = conf < 0.5 or req.is_ambiguous

    # Conflict badge HTML
    conflict_badge = ""
    if classification.conflicts:
        severities = [w.severity for w in classification.conflicts]
        worst = "high" if "high" in severities else "medium" if "medium" in severities else "low"
        clr = _CONFLICT_COLORS.get(worst, "#6b7280")
        conflict_badge = (
            f' <span style="background:{clr}; color:white; font-size:0.68rem; '
            f'padding:2px 6px; border-radius:10px; font-weight:600;">'
            f'⚡ {len(classification.conflicts)}</span>'
        )

    row_cols = st.columns([0.4, 3.5, 1.2, 1, 0.5, 0.5])

    with row_cols[0]:
        st.markdown(f"**{idx + 1}**")

    with row_cols[1]:
        section_tag = f"`{req.original_section}` " if req.original_section else ""
        ambig_tag = " ⚠️" if req.is_ambiguous else ""
        override_tag = " 📝" if override else ""
        text_preview = req.text[:110] + ("..." if len(req.text) > 110 else "")
        st.markdown(
            f"{section_tag}{text_preview}{ambig_tag}{override_tag}{conflict_badge}",
            unsafe_allow_html=True,
        )

    with row_cols[2]:
        if override:
            st.markdown(
                f'<span style="text-decoration:line-through; color:#888;">'
                f'{icon} {label.value}</span><br/>'
                f'<span style="font-weight:600;">→ {override}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span style="color:{color}; font-weight:600;">'
                f'{icon} {label.value}</span>',
                unsafe_allow_html=True,
            )

    with row_cols[3]:
        bar_color = (
            "#22c55e" if conf >= 0.8
            else "#f59e0b" if conf >= 0.5
            else "#ef4444"
        )
        st.markdown(
            f'<div style="background:#e5e7eb; border-radius:4px; height:20px; width:100%;">'
            f'<div style="background:{bar_color}; width:{conf*100:.0f}%; '
            f'height:100%; border-radius:4px; text-align:center; '
            f'color:white; font-size:0.75rem; line-height:20px;">'
            f'{conf:.0%}</div></div>',
            unsafe_allow_html=True,
        )

    with row_cols[4]:
        if needs_attention:
            st.markdown("🔍")

    with row_cols[5]:
        new_approved = st.checkbox(
            "approve",
            value=approved,
            key=f"approve_tbl_{req_id}",
            help="Approve this requirement for the proposal",
            label_visibility="collapsed",
        )
        st.session_state.approvals[req_id] = new_approved
