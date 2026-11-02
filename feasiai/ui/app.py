"""
FeasiAI — Streamlit Review UI

Main app entry point. Run with:
    streamlit run feasiai/ui/app.py

Or from the project root:
    python -m streamlit run feasiai/ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from feasiai.ui.utils import (
    export_report_csv,
    export_report_json,
    init_session_state,
)
from feasiai.ui.components.sidebar import render_sidebar
from feasiai.ui.components.dashboard import render_dashboard
from feasiai.ui.components.table import render_table
from feasiai.ui.components.detail import render_detail

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="FeasiAI — Feasibility Review",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Initialize ───────────────────────────────────────────────────────────

init_session_state(st)


# ── Custom CSS ───────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Tighter spacing for requirement rows */
    .stColumns {
        margin-bottom: -0.5rem;
    }
    /* Expander header styling */
    .streamlit-expanderHeader {
        font-size: 0.95rem;
    }
    /* Metric styling */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ──────────────────────────────────────────────────────────────

render_sidebar()


# ── Main content ─────────────────────────────────────────────────────────

report = st.session_state.report

if report is None:
    # Welcome screen
    st.markdown(
        """
        # 🤖 FeasiAI — Feasibility Review UI

        Welcome to the **FeasiAI Review Interface**. This tool lets you:

        - 📊 **View** feasibility assessment reports with interactive charts
        - 🔍 **Filter & search** requirements by label, confidence, or text
        - 📄 **Drill into** evidence and agent reasoning traces
        - 📝 **Override** AI labels and add reviewer notes
        - 📥 **Export** finalized reports as JSON or CSV

        ### Getting Started

        Use the **sidebar** to either:
        1. **Upload** a JSON report from `run_pipeline.py --json-output`
        2. **Run the pipeline** directly on an RFP document
        3. **Load the demo** to explore the UI with sample data

        ---

        *FeasiAI — AI Solution Feasibility & Proposal Copilot*
        """
    )
    st.stop()


# ── Report loaded — render the UI ────────────────────────────────────────

st.title(f"📋 {report.source_document or 'Feasibility Report'}")
st.caption(
    f"Report ID: `{report.id[:12]}...` | "
    f"Generated: {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}"
)

# Dashboard
render_dashboard(report)

st.divider()

# View toggle
view_mode = st.radio(
    "View mode",
    ["📋 Table View", "🔍 Detail View"],
    horizontal=True,
    label_visibility="collapsed",
)

if view_mode == "📋 Table View":
    render_table(report)
else:
    render_detail(report)

# ── Export bar ───────────────────────────────────────────────────────────

st.divider()

export_cols = st.columns([1, 1, 4])

with export_cols[0]:
    overrides = st.session_state.overrides
    notes = st.session_state.notes

    json_export = export_report_json(report, overrides, notes)
    st.download_button(
        "📥 Export JSON",
        data=json_export,
        file_name=f"feasiai_report_{report.id[:8]}.json",
        mime="application/json",
        use_container_width=True,
    )

with export_cols[1]:
    csv_export = export_report_csv(report, overrides, notes)
    st.download_button(
        "📥 Export CSV",
        data=csv_export,
        file_name=f"feasiai_report_{report.id[:8]}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_cols[2]:
    # Override summary
    override_count = sum(1 for v in overrides.values() if v)
    note_count = sum(1 for v in notes.values() if v)
    if override_count > 0 or note_count > 0:
        st.info(
            f"📝 {override_count} label override(s), "
            f"{note_count} reviewer note(s) will be included in exports."
        )

