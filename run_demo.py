"""
End-to-end demo script for FeasiAI.

Runs the full pipeline on the sample RFP and prints a report.
No API key needed for the basic smoke test (uses mocked LLM).

Usage:
    # Quick smoke test (mocked LLM — no API key):
    python run_demo.py --mock

    # Full demo with real LLM (requires GROQ_API_KEY in .env):
    python run_demo.py

    # With roadmap live data connector:
    python run_demo.py --live-data

    # Save report to file:
    python run_demo.py --output demo_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FeasiAI end-to-end demo")
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use mocked LLM (no API key required)",
    )
    p.add_argument(
        "--live-data",
        action="store_true",
        help="Enable roadmap live data connector",
    )
    p.add_argument(
        "--rfp",
        default=str(PROJECT_ROOT / "data" / "sample_rfps" / "sample_rfp_01.md"),
        help="Path to RFP document",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Save JSON report to this file",
    )
    p.add_argument(
        "--no-agent",
        action="store_true",
        help="Disable planner agent (Phase 1 single-shot mode)",
    )
    return p.parse_args()


def _make_mock_llm():
    """Return a mock LLM client that produces deterministic responses."""
    from unittest.mock import MagicMock
    from feasiai.llm.client import LLMClient

    llm = MagicMock(spec=LLMClient)
    llm.call_count = 0

    # Extraction response
    extract_response = json.dumps({
        "requirements": [
            {
                "text": "The system must support SSO via SAML 2.0",
                "original_section": "Security",
                "is_ambiguous": False,
            },
            {
                "text": "The system must provide HIPAA-compliant data storage",
                "original_section": "Compliance",
                "is_ambiguous": False,
            },
            {
                "text": "The system must integrate with Shopify",
                "original_section": "Integrations",
                "is_ambiguous": False,
            },
        ]
    })

    # Classification response
    classify_response = json.dumps({
        "label": "Available",
        "confidence": 0.88,
        "reasoning": "SSO via SAML 2.0 is confirmed in product capabilities.",
        "what_would_change": "Custom SAML attribute mapping requirements.",
        "draft_response": "Our platform fully supports SSO via SAML 2.0 and OIDC.",
    })

    llm.chat.return_value = extract_response
    llm.chat_json.return_value = {
        "action": "classify",
        "label": "Available",
        "confidence": 0.88,
        "reasoning": "Evidence confirms this capability.",
        "what_would_change": "Edge cases in custom configuration.",
        "draft_response": "Yes, this is supported.",
    }
    llm.reset_budget = MagicMock()
    return llm


def main() -> None:
    args = _parse_args()

    print("=" * 60)
    print("🤖  FeasiAI — AI Solution Feasibility & Proposal Copilot")
    print("=" * 60)
    print()

    # ── Setup ────────────────────────────────────────────────────────
    print("⚙️  Setting up pipeline...")

    if args.mock:
        print("   Mode: MOCK (no API key required)")
        llm = _make_mock_llm()
    else:
        print("   Mode: REAL (using GROQ_API_KEY from .env)")
        from feasiai.llm.client import LLMClient
        llm = LLMClient()

    use_agent = not args.no_agent
    print(f"   Agent: {'enabled' if use_agent else 'disabled (single-shot)'}")

    # ── Live data ────────────────────────────────────────────────────
    live_registry = None
    if args.live_data:
        from feasiai.live import LiveDataRegistry, RoadmapConnector
        live_registry = LiveDataRegistry()
        roadmap_connector = RoadmapConnector()
        live_registry.register(roadmap_connector)
        health = live_registry.health_check_all()
        print(f"   Live data: enabled — {live_registry.active_connectors}")
        for name, ok in health.items():
            status = "✅" if ok else "❌"
            print(f"     {status} {name}")
    else:
        print("   Live data: disabled")

    # ── Memory store ─────────────────────────────────────────────────
    from feasiai.retrieval.memory_store import MemoryStore
    from feasiai.pipeline import FeasibilityPipeline

    if args.mock:
        memory = MemoryStore(ephemeral=True)
    else:
        memory = MemoryStore()

    pipeline = FeasibilityPipeline(
        llm_client=llm,
        memory_store=memory,
        use_agent=use_agent,
        live_registry=live_registry,
    )

    print()
    print("📚  Initializing memory store...")
    t0 = time.perf_counter()

    if args.mock:
        # Skip file ingest for mock mode
        pipeline._initialized = True
        if use_agent:
            from feasiai.agent.planner import PlannerAgent
            pipeline._agent = PlannerAgent(llm, memory, live_registry=live_registry)
    else:
        pipeline.initialize()

    print(f"   Done in {time.perf_counter() - t0:.1f}s")

    # ── Run pipeline ─────────────────────────────────────────────────
    t1 = time.perf_counter()

    if args.mock:
        # In mock mode, inject requirements directly to avoid mocked extractor
        print()
        print("📄  Running pipeline on: [mock inline RFP text]")
        from feasiai.schemas import Requirement, Classification, FeasibilityLabel
        from feasiai.report.generator import build_report

        mock_reqs = [
            Requirement(text="The system must support SSO via SAML 2.0", original_section="Security"),
            Requirement(text="The system must provide HIPAA-compliant data storage", original_section="Compliance"),
            Requirement(text="The system must integrate with Shopify", original_section="Integrations"),
        ]
        label_cycle = [
            FeasibilityLabel.AVAILABLE,
            FeasibilityLabel.CONFIGURABLE,
            FeasibilityLabel.REQUIRES_DEVELOPMENT,
        ]
        mock_cls = [
            Classification(
                requirement_id=req.id,
                label=label_cycle[i % 3],
                confidence=0.85 - i * 0.1,
                reasoning=f"Mock reasoning for requirement {i + 1}.",
                what_would_change="N/A (mock mode)",
                draft_response=f"Mock draft response for: {req.text[:60]}",
            )
            for i, req in enumerate(mock_reqs)
        ]
        report = build_report(
            requirements=mock_reqs,
            classifications=mock_cls,
            source_document="[mock RFP]",
            llm_call_count=3,
            total_latency_seconds=time.perf_counter() - t1,
        )
    else:
        rfp_path = args.rfp
        if not Path(rfp_path).exists():
            print(f"❌  RFP not found: {rfp_path}")
            sys.exit(1)
        print()
        print(f"📄  Running pipeline on: {rfp_path}")
        report = pipeline.run(input_path=rfp_path)

    elapsed = time.perf_counter() - t1

    # ── Print report ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("📋  FEASIBILITY REPORT")
    print("=" * 60)
    print(pipeline.format_report(report))

    # ── Stats ────────────────────────────────────────────────────────
    print()
    print("📊  Statistics")
    print(f"   Requirements:  {len(report.requirements)}")
    print(f"   LLM calls:     {report.llm_call_count}")
    print(f"   Latency:       {elapsed:.1f}s")

    # Conflict summary
    total_conflicts = sum(len(c.conflicts) for c in report.classifications)
    if total_conflicts:
        print(f"   ⚡ Conflicts:   {total_conflicts} detected")
        for c in report.classifications:
            if c.conflicts:
                req = next((r for r in report.requirements if r.id == c.requirement_id), None)
                req_text = req.text[:60] if req else "?"
                for w in c.conflicts:
                    print(f"     [{w.severity.upper()}] {req_text}... → {w.description[:80]}")

    # ── Save output ──────────────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        print()
        print(f"💾  Report saved to: {output_path}")

    print()
    print("✅  Demo complete.")
    print()
    print("Next steps:")
    print("  streamlit run feasiai/ui/app.py    — open the review UI")
    print("  python run_eval.py --limit 5       — run the evaluation harness")
    print("  pytest tests/ -q                   — run all 231 tests")


if __name__ == "__main__":
    main()
