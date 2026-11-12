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
        - ⚡ **Inspect conflicts** between memory and live data sources
        - 📄 **Drill into** evidence and agent reasoning traces
        - 📝 **Override** AI labels and add reviewer notes
        - ✅ **Approve** requirements for inclusion in the proposal
        - 📥 **Export** finalized reports as JSON or CSV
        - 💾 **Save feedback** for closed-loop memory improvement

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

# ── Approval summary ─────────────────────────────────────────────────────

approvals = st.session_state.approvals
approved_ids = [rid for rid, v in approvals.items() if v]
total_reqs = len(report.classifications)
approved_count = len(approved_ids)

if approved_count > 0:
    st.divider()
    with st.expander(
        f"✅ Proposal Gate — {approved_count}/{total_reqs} requirements approved",
        expanded=False,
    ):
        st.info(
            f"**{approved_count}** requirement(s) are marked as proposal-ready. "
            f"**{total_reqs - approved_count}** still need review."
        )
        req_map = {r.id: r for r in report.requirements}

        # Show approved items
        for c in report.classifications:
            if c.requirement_id in approved_ids:
                req = req_map.get(c.requirement_id)
                final_label = approvals.get(c.requirement_id) or c.label.value
                note = st.session_state.notes.get(c.requirement_id, "")
                st.markdown(
                    f"- **{final_label}** — {req.text[:90] if req else '?'}{'...' if req and len(req.text) > 90 else ''}"
                    + (f" *(note: {note})*" if note else "")
                )

# ── Export bar ───────────────────────────────────────────────────────────

st.divider()

export_cols = st.columns([1, 1, 1, 3])

overrides = st.session_state.overrides
notes = st.session_state.notes

with export_cols[0]:
    json_export = export_report_json(report, overrides, notes, approvals)
    st.download_button(
        "📥 Export JSON",
        data=json_export,
        file_name=f"feasiai_report_{report.id[:8]}.json",
        mime="application/json",
        use_container_width=True,
    )

with export_cols[1]:
    csv_export = export_report_csv(report, overrides, notes, approvals)
    st.download_button(
        "📥 Export CSV",
        data=csv_export,
        file_name=f"feasiai_report_{report.id[:8]}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with export_cols[2]:
    # Save feedback to SQLite
    if st.button("💾 Save Feedback", use_container_width=True, type="primary"):
        _save_feedback(report, overrides, notes, approvals)

with export_cols[3]:
    override_count = sum(1 for v in overrides.values() if v)
    note_count = sum(1 for v in notes.values() if v)
    summary_parts = []
    if approved_count > 0:
        summary_parts.append(f"✅ {approved_count} approved")
    if override_count > 0:
        summary_parts.append(f"📝 {override_count} override(s)")
    if note_count > 0:
        summary_parts.append(f"🗒️ {note_count} note(s)")
    if summary_parts:
        st.info("  |  ".join(summary_parts))


# ── Memory write-back panel ───────────────────────────────────────────────

if approved_count > 0:
    st.divider()
    with st.expander("🔄 Closed-Loop Memory Write-back", expanded=False):
        st.markdown(
            """
            After the proposal is **delivered**, record the outcome to improve future assessments.
            This writes lessons and corrections back into the institutional memory store.
            """
        )

        wb_cols = st.columns([1, 1, 2])
        with wb_cols[0]:
            outcome = st.selectbox(
                "Proposal outcome",
                ["won", "delivered", "lost", "cancelled"],
                key="wb_outcome",
            )
        with wb_cols[1]:
            effort = st.number_input(
                "Total effort (days, optional)",
                min_value=0.0,
                value=0.0,
                step=0.5,
                key="wb_effort",
            )
        with wb_cols[2]:
            vertical = st.text_input(
                "Client vertical (optional)",
                placeholder="e.g. healthcare, fintech, government",
                key="wb_vertical",
            )

        outcome_notes = st.text_area(
            "Outcome notes",
            placeholder="What went wrong? What was easier than expected? Any gaps discovered?",
            height=80,
            key="wb_notes",
        )

        if st.button("🔄 Write to Memory", type="primary", use_container_width=False):
            _trigger_writeback(
                report=report,
                overrides=overrides,
                notes=notes,
                approvals=approvals,
                outcome=outcome,
                effort=effort if effort > 0 else None,
                vertical=vertical or None,
                outcome_notes=outcome_notes or None,
            )


def _save_feedback(report, overrides, notes, approvals) -> None:
    """Persist reviewer decisions to the feedback SQLite store."""
    try:
        from feasiai.feedback import FeedbackStore, FeedbackRecord, ReviewDecision

        store = FeedbackStore()
        req_map = {r.id: r for r in report.requirements}
        saved = 0

        for c in report.classifications:
            rid = c.requirement_id
            req = req_map.get(rid)
            if not req:
                continue

            override = overrides.get(rid, "")
            note = notes.get(rid, "")
            approved = approvals.get(rid, False)

            if not (override or note or approved):
                continue  # Nothing to save for this requirement

            final_label = override if override else c.label.value
            if approved:
                decision = ReviewDecision.APPROVED
            elif override:
                decision = ReviewDecision.OVERRIDDEN
            elif note:
                decision = ReviewDecision.FLAGGED
            else:
                continue

            record = FeedbackRecord(
                report_id=report.id,
                requirement_id=rid,
                requirement_text=req.text,
                ai_label=c.label.value,
                ai_confidence=c.confidence,
                final_label=final_label,
                decision=decision,
                reviewer_notes=note,
                draft_response_approved=approved,
            )
            store.save(record)
            saved += 1

        if saved > 0:
            st.success(f"💾 Saved {saved} feedback record(s) to local store.")
        else:
            st.info("No new feedback to save. Make overrides, notes, or approvals first.")

    except Exception as e:
        st.error(f"Failed to save feedback: {e}")


def _trigger_writeback(
    report, overrides, notes, approvals,
    outcome, effort, vertical, outcome_notes
) -> None:
    """Derive write-back records from approved feedback and write to memory."""
    try:
        from feasiai.feedback import FeedbackStore, FeedbackRecord, ReviewDecision
        from feasiai.writeback import MemoryWritebackService

        store = FeedbackStore()
        approved_feedback = store.get_approved_for_proposal(report.id)

        if not approved_feedback:
            # Create feedback on-the-fly from current session state
            req_map = {r.id: r for r in report.requirements}
            approved_feedback = []
            for c in report.classifications:
                rid = c.requirement_id
                if not approvals.get(rid, False):
                    continue
                req = req_map.get(rid)
                if not req:
                    continue
                override = overrides.get(rid, "")
                note = notes.get(rid, "")
                approved_feedback.append(FeedbackRecord(
                    report_id=report.id,
                    requirement_id=rid,
                    requirement_text=req.text,
                    ai_label=c.label.value,
                    ai_confidence=c.confidence,
                    final_label=override if override else c.label.value,
                    decision=ReviewDecision.APPROVED,
                    reviewer_notes=note,
                    draft_response_approved=True,
                ))

        service = MemoryWritebackService()
        path = service.process_feedback_batch(
            feedback_records=approved_feedback,
            outcome=outcome,
            actual_effort_days=effort,
            outcome_notes=outcome_notes,
            client_vertical=vertical,
        )

        if path:
            st.success(
                f"✅ **Memory updated!** {len(approved_feedback)} feedback records "
                f"written to `{path.name}`. Re-run `pipeline.initialize()` to ingest."
            )
            st.code(str(path), language=None)
        else:
            st.info("No write-back records were generated from the approved feedback.")

    except Exception as e:
        st.error(f"Write-back failed: {e}")
