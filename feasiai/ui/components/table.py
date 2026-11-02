"""
Requirements table component with filtering.
"""

from __future__ import annotations

import streamlit as st

from feasiai.schemas import FeasibilityLabel, FeasibilityReport
from feasiai.ui.utils import LABEL_COLORS, LABEL_ICONS, LABEL_OPTIONS, init_session_state


def render_table(report: FeasibilityReport) -> None:
    """Render the filterable requirements table."""
    init_session_state(st)

    req_map = {r.id: r for r in report.requirements}

    # ── Filters ──────────────────────────────────────────────────────
    with st.expander("🔍 Filters", expanded=False):
        filter_cols = st.columns([2, 1, 1, 2])

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
            search_text = st.text_input(
                "Search requirements",
                placeholder="Type to search...",
            )

    # ── Apply filters ────────────────────────────────────────────────
    filtered_items = []
    for i, c in enumerate(report.classifications):
        req = req_map.get(c.requirement_id)
        if not req:
            continue

        # Label filter
        if selected_labels and c.label.value not in selected_labels:
            continue

        # Confidence filter
        if c.confidence < min_conf:
            continue

        # Ambiguous filter
        if ambiguous_only and not req.is_ambiguous:
            continue

        # Text search
        if search_text:
            search_lower = search_text.lower()
            if (
                search_lower not in req.text.lower()
                and search_lower not in (req.original_section or "").lower()
                and search_lower not in c.reasoning.lower()
            ):
                continue

        filtered_items.append((i, req, c))

    # ── Results count ────────────────────────────────────────────────
    total = len(report.classifications)
    shown = len(filtered_items)
    if shown < total:
        st.caption(f"Showing {shown} of {total} requirements")
    else:
        st.caption(f"{total} requirements")

    # ── Render table ─────────────────────────────────────────────────
    if not filtered_items:
        st.info("No requirements match the current filters.")
        return

    for idx, req, classification in filtered_items:
        _render_row(idx, req, classification)


def _render_row(idx: int, req, classification) -> None:
    """Render a single requirement row."""
    label = classification.label
    icon = LABEL_ICONS[label]
    color = LABEL_COLORS[label]
    conf = classification.confidence
    override = st.session_state.overrides.get(classification.requirement_id, "")

    # Determine if this item needs attention
    needs_attention = conf < 0.5 or req.is_ambiguous

    # Build the row
    row_cols = st.columns([0.4, 4, 1.2, 1, 0.6])

    with row_cols[0]:
        st.markdown(f"**{idx + 1}**")

    with row_cols[1]:
        # Requirement text with section tag
        section_tag = ""
        if req.original_section:
            section_tag = f"`{req.original_section}` "
        ambig_tag = " ⚠️" if req.is_ambiguous else ""
        override_tag = " 📝" if override else ""

        text_preview = req.text[:120]
        if len(req.text) > 120:
            text_preview += "..."

        st.markdown(f"{section_tag}{text_preview}{ambig_tag}{override_tag}")

    with row_cols[2]:
        if override:
            # Show override label
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
        # Confidence bar
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

