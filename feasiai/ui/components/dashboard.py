"""
Dashboard component: summary statistics and charts.
"""

from __future__ import annotations

import streamlit as st

from feasiai.schemas import FeasibilityLabel, FeasibilityReport
from feasiai.ui.utils import LABEL_COLORS, LABEL_ICONS


def render_dashboard(report: FeasibilityReport) -> None:
    """Render the dashboard with summary stats and charts."""
    s = report.summary

    # ── Attention banner ─────────────────────────────────────────────
    low_conf = s.get("low_confidence_items", 0)
    ambiguous = s.get("ambiguous_requirements", 0)
    attention_items = low_conf + ambiguous
    if attention_items > 0:
        st.warning(
            f"⚠️ **{attention_items} item(s) need human review** — "
            f"{low_conf} low-confidence, {ambiguous} ambiguous",
            icon="⚠️",
        )

    # ── Key metrics row ──────────────────────────────────────────────
    cols = st.columns(6)

    with cols[0]:
        st.metric("Total Requirements", s.get("total_requirements", 0))
    with cols[1]:
        avg_conf = s.get("average_confidence", 0)
        st.metric("Avg Confidence", f"{avg_conf:.0%}")
    with cols[2]:
        st.metric("✅ Available", s.get("available_count", 0))
    with cols[3]:
        st.metric("⚙️ Configurable", s.get("configurable_count", 0))
    with cols[4]:
        st.metric("🔨 Requires Dev", s.get("requires_dev_count", 0))
    with cols[5]:
        st.metric("❌ Unsupported", s.get("unsupported_count", 0))

    # ── Charts row ───────────────────────────────────────────────────
    chart_col1, chart_col2 = st.columns([3, 2])

    with chart_col1:
        st.markdown("##### Label Distribution")
        _render_label_bar_chart(report)

    with chart_col2:
        st.markdown("##### Pipeline Stats")
        st.markdown(
            f"- 📄 **Document:** {report.source_document or 'N/A'}\n"
            f"- 🤖 **LLM calls:** {report.llm_call_count}\n"
            f"- ⏱️ **Latency:** {report.total_latency_seconds:.1f}s\n"
            f"- ⚠️ **Ambiguous:** {s.get('ambiguous_requirements', 0)}\n"
            f"- 🔍 **Low confidence:** {s.get('low_confidence_items', 0)}"
        )


def _render_label_bar_chart(report: FeasibilityReport) -> None:
    """Render a horizontal stacked bar using Streamlit columns and markdown."""
    total = len(report.classifications)
    if total == 0:
        st.info("No classifications to display.")
        return

    label_order = [
        FeasibilityLabel.AVAILABLE,
        FeasibilityLabel.CONFIGURABLE,
        FeasibilityLabel.REQUIRES_DEVELOPMENT,
        FeasibilityLabel.UNSUPPORTED_UNCLEAR,
    ]

    label_counts = {}
    for c in report.classifications:
        label_counts[c.label] = label_counts.get(c.label, 0) + 1

    # Build the bar using columns proportional to counts
    segments = []
    for label in label_order:
        count = label_counts.get(label, 0)
        if count > 0:
            segments.append((label, count))

    if not segments:
        return

    bar_cols = st.columns([count for _, count in segments])
    for col, (label, count) in zip(bar_cols, segments):
        color = LABEL_COLORS[label]
        icon = LABEL_ICONS[label]
        pct = count / total * 100
        with col:
            st.markdown(
                f'<div style="background-color:{color}; color:white; '
                f'padding:8px; border-radius:4px; text-align:center; '
                f'font-size:0.85rem; font-weight:600;">'
                f'{icon} {count} ({pct:.0f}%)</div>',
                unsafe_allow_html=True,
            )

    # Legend
    legend_parts = []
    for label in label_order:
        count = label_counts.get(label, 0)
        color = LABEL_COLORS[label]
        icon = LABEL_ICONS[label]
        legend_parts.append(
            f'<span style="color:{color}; font-weight:600;">'
            f'{icon} {label.value}: {count}</span>'
        )
    st.markdown(
        " &nbsp;·&nbsp; ".join(legend_parts),
        unsafe_allow_html=True,
    )

