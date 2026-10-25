#!/usr/bin/env python3
"""
FeasiAI — Run the feasibility assessment pipeline.

Usage:
    python run_pipeline.py                          # Use default sample RFP
    python run_pipeline.py path/to/rfp.pdf          # Use a specific document
    python run_pipeline.py --text "Your RFP text"   # Use inline text

The pipeline will:
1. Load institutional knowledge into the memory store
2. Parse the RFP document
3. Extract atomic requirements
4. Retrieve evidence from memory
5. Classify each requirement
6. Print a structured feasibility report
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from feasiai.pipeline import FeasibilityPipeline


def main():
    parser = argparse.ArgumentParser(
        description="FeasiAI — AI Solution Feasibility & Proposal Copilot"
    )
    parser.add_argument(
        "document",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "sample_rfps" / "sample_rfp_01.md"),
        help="Path to RFP document (PDF, DOCX, TXT, MD)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="RFP text directly (alternative to file input)",
    )
    parser.add_argument(
        "--memory-dir",
        type=str,
        default=None,
        help="Directory containing institutional knowledge documents",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to save the report as JSON",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("  🤖 FeasiAI — Feasibility Assessment Pipeline")
    print("=" * 60 + "\n")

    # Initialize pipeline
    pipeline = FeasibilityPipeline()
    pipeline.initialize(memory_docs_dir=args.memory_dir)

    # Run pipeline
    if args.text:
        report = pipeline.run(input_text=args.text, source_name="cli_input")
    else:
        doc_path = args.document
        if not Path(doc_path).exists():
            print(f"❌ File not found: {doc_path}")
            sys.exit(1)
        report = pipeline.run(input_path=doc_path)

    # Print human-readable report
    print("\n" + pipeline.format_report(report))

    # Save JSON if requested
    if args.json_output:
        output_path = Path(args.json_output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
        print(f"\n📄 JSON report saved to: {output_path}")

    # Print LLM usage summary
    print(f"\n📊 LLM Usage: {report.llm_call_count} calls | "
          f"{report.total_latency_seconds:.1f}s total")

    return 0


if __name__ == "__main__":
    sys.exit(main())
