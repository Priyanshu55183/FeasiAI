#!/usr/bin/env python3
"""
FeasiAI Evaluation Harness CLI.

Runs the feasibility pipeline against a labeled dataset and reports
accuracy, macro-F1, ECE, and efficiency metrics.

Usage:
    # Run agent vs single-shot comparison (default)
    python run_eval.py

    # Run only one mode
    python run_eval.py --mode agent
    python run_eval.py --mode single_shot

    # Use a custom dataset
    python run_eval.py --dataset path/to/dataset.jsonl

    # Save results to JSON
    python run_eval.py --output-dir eval/results/

    # Run a subset (for quick smoke test)
    python run_eval.py --limit 5
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from feasiai.eval.dataset import load_dataset
from feasiai.eval.metrics import compute_metrics
from feasiai.eval.report import format_comparison, format_eval_report, save_report
from feasiai.eval.runner import EvalRunner
from feasiai.llm.client import LLMClient
from feasiai.retrieval.memory_store import MemoryStore


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FeasiAI Evaluation Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["agent", "single_shot", "both"],
        default="both",
        help="Which pipeline mode to evaluate (default: both)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to JSONL eval dataset (default: eval/dataset.jsonl)",
    )
    parser.add_argument(
        "--memory-dir",
        type=str,
        default=None,
        help="Directory with institutional knowledge docs (default: data/memory/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save JSON result files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples (for quick smoke tests)",
    )
    parser.add_argument(
        "--agent-max-steps",
        type=int,
        default=3,
        help="Max tool calls per requirement in agent mode (default: 3)",
    )
    parser.add_argument(
        "--agent-threshold",
        type=float,
        default=0.85,
        help="Agent early-stop confidence threshold (default: 0.85)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 72)
    print("  🧪 FeasiAI Evaluation Harness")
    print("=" * 72 + "\n")

    # Load dataset
    print("📂 Loading dataset...")
    samples = load_dataset(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
        print(f"   (Limited to {len(samples)} samples)")
    print(f"   {len(samples)} samples loaded\n")

    # Initialize shared components
    print("🔧 Initializing pipeline...")
    llm = LLMClient()
    model_name = llm._model

    memory = MemoryStore(ephemeral=True)
    memory.ingest_directory(args.memory_dir)
    print("   Memory store ready\n")

    reports = {}

    # ── Run evaluations ──────────────────────────────────────────
    if args.mode in ("agent", "both"):
        print("🤖 Running AGENT mode evaluation...")
        runner = EvalRunner(
            llm_client=llm,
            memory_store=memory,
            use_agent=True,
            agent_max_steps=args.agent_max_steps,
            agent_confidence_threshold=args.agent_threshold,
        )
        results = runner.run(samples)
        agent_report = compute_metrics(results, mode="agent", model_name=model_name)
        reports["agent"] = agent_report
        print(format_eval_report(agent_report))

        if args.output_dir:
            path = Path(args.output_dir) / "eval_agent.json"
            save_report(agent_report, str(path))

    if args.mode in ("single_shot", "both"):
        print("⚡ Running SINGLE-SHOT mode evaluation...")
        # Reuse the same LLM client — runner resets budget per sample
        runner = EvalRunner(
            llm_client=llm,
            memory_store=memory,
            use_agent=False,
        )
        results = runner.run(samples)
        ss_report = compute_metrics(results, mode="single_shot", model_name=model_name)
        reports["single_shot"] = ss_report
        print(format_eval_report(ss_report))

        if args.output_dir:
            path = Path(args.output_dir) / "eval_single_shot.json"
            save_report(ss_report, str(path))

    # ── Side-by-side comparison ───────────────────────────────────
    if "agent" in reports and "single_shot" in reports:
        print(format_comparison(reports["agent"], reports["single_shot"]))

    return 0


if __name__ == "__main__":
    sys.exit(main())

