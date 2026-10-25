"""
End-to-end pipeline orchestrator.

This is the Phase 1 single-shot pipeline:
  Document → Parse → Extract Requirements → Retrieve Evidence → Classify → Report

Design decisions:
- Linear pipeline (no agentic planning yet — that's Phase 2).
- Each step is independently testable.
- The pipeline tracks total LLM calls and latency for evaluation.
- Failures in individual requirement processing are caught and logged
  rather than aborting the entire pipeline. Partial results are still
  valuable.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from feasiai import config
from feasiai.classification.classifier import classify_requirement
from feasiai.extraction.requirement_extractor import extract_requirements
from feasiai.ingestion.parser import parse_document, parse_text
from feasiai.llm.client import LLMClient
from feasiai.report.generator import build_report, format_report_text
from feasiai.retrieval.memory_store import MemoryStore
from feasiai.schemas import Classification, FeasibilityReport, Requirement

logger = logging.getLogger(__name__)


class FeasibilityPipeline:
    """
    End-to-end feasibility assessment pipeline.

    Usage:
        pipeline = FeasibilityPipeline()
        pipeline.initialize()  # Load memory store
        report = pipeline.run("path/to/rfp.pdf")
        print(pipeline.format_report(report))
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        memory_store: Optional[MemoryStore] = None,
    ):
        self._llm = llm_client or LLMClient()
        self._memory = memory_store or MemoryStore()
        self._initialized = False

    def initialize(self, memory_docs_dir: Optional[str] = None) -> None:
        """
        Initialize the pipeline: ingest institutional knowledge documents
        into the memory store.

        Args:
            memory_docs_dir: Directory containing memory documents.
                             Uses config default if not specified.
        """
        logger.info("Initializing memory store...")
        chunk_count = self._memory.ingest_directory(memory_docs_dir)
        logger.info(
            "Memory store initialized with %d chunks", chunk_count
        )
        self._initialized = True

    def run(
        self,
        input_path: Optional[str] = None,
        input_text: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> FeasibilityReport:
        """
        Run the full pipeline on an RFP document or text.

        Args:
            input_path: Path to PDF/DOCX/TXT file.
            input_text: Raw text input (alternative to file).
            source_name: Document name for provenance.

        Returns:
            FeasibilityReport with all classifications and statistics.
        """
        if not self._initialized:
            logger.warning(
                "Pipeline not initialized. Call initialize() first. "
                "Proceeding with empty memory store."
            )

        start_time = time.perf_counter()
        self._llm.reset_budget()

        # Step 1: Parse document
        logger.info("Step 1: Parsing document...")
        if input_path:
            text = parse_document(input_path)
            source_name = source_name or Path(input_path).name
        elif input_text:
            text = parse_text(input_text)
            source_name = source_name or "direct_input"
        else:
            raise ValueError("Must provide either input_path or input_text")

        logger.info("Parsed %d characters from %s", len(text), source_name)

        # Step 2: Extract requirements
        logger.info("Step 2: Extracting requirements...")
        extraction = extract_requirements(
            text, self._llm, source_document=source_name
        )
        logger.info(
            "Extracted %d requirements (%d ambiguous)",
            len(extraction.requirements),
            sum(1 for r in extraction.requirements if r.is_ambiguous),
        )

        if not extraction.requirements:
            logger.warning("No requirements extracted. Returning empty report.")
            return build_report(
                requirements=[],
                classifications=[],
                source_document=source_name,
                llm_call_count=self._llm.call_count,
                total_latency_seconds=time.perf_counter() - start_time,
            )

        # Step 3 & 4: Retrieve evidence and classify each requirement
        logger.info("Steps 3-4: Retrieving evidence and classifying...")
        classifications: list[Classification] = []

        for i, req in enumerate(extraction.requirements, 1):
            logger.info(
                "Processing requirement %d/%d: %.60s...",
                i,
                len(extraction.requirements),
                req.text,
            )

            try:
                # Retrieve evidence from memory
                evidences = self._memory.retrieve(req.text)
                logger.debug(
                    "  Retrieved %d evidence chunks", len(evidences)
                )

                # Classify
                classification = classify_requirement(
                    req, evidences, self._llm
                )
                classifications.append(classification)

                logger.info(
                    "  → %s (%.0f%% confidence)",
                    classification.label.value,
                    classification.confidence * 100,
                )

            except Exception as e:
                logger.error(
                    "Failed to process requirement %s: %s",
                    req.id[:8],
                    e,
                )
                # Create an explicit failure classification
                from feasiai.schemas import FeasibilityLabel

                classifications.append(
                    Classification(
                        requirement_id=req.id,
                        label=FeasibilityLabel.UNSUPPORTED_UNCLEAR,
                        confidence=0.0,
                        supporting_evidence=[],
                        reasoning=f"Processing failed: {e}",
                        what_would_change="Resolve the processing error and retry.",
                        draft_response=(
                            "We are unable to assess this requirement at this "
                            "time due to a processing error."
                        ),
                    )
                )

        # Step 5: Build report
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Step 5: Building report... (%d LLM calls, %.1fs elapsed)",
            self._llm.call_count,
            elapsed,
        )

        report = build_report(
            requirements=extraction.requirements,
            classifications=classifications,
            source_document=source_name,
            llm_call_count=self._llm.call_count,
            total_latency_seconds=elapsed,
        )

        return report

    def format_report(self, report: FeasibilityReport) -> str:
        """Format a report as human-readable text."""
        return format_report_text(report)

    @property
    def llm_call_log(self):
        """Access the LLM call log for post-run analysis."""
        return self._llm.call_log
