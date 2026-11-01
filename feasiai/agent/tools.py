"""
Agent tools: wrappers around MemoryStore for focused retrieval.

Design decisions:
1. THREE TOOL TYPES: Each tool searches the same MemoryStore but with
   different metadata filters and result framing. This lets the agent
   choose *what kind* of evidence to look for, not just *what query*.

2. CATEGORY FILTERING: Tools like check_gaps and search_past_proposals
   boost results from specific source documents by filtering on
   metadata category. This is cheaper than re-indexing.

3. EVIDENCE DEDUPLICATION: The AgentToolkit tracks all evidence returned
   across tool calls and deduplicates by chunk_id. This prevents the
   classifier from seeing duplicate context.

4. RESULT CAPPING: Each tool returns at most top_k results to keep
   the agent's context window manageable.
"""

from __future__ import annotations

import logging
from typing import Optional

from feasiai import config
from feasiai.retrieval.memory_store import MemoryStore
from feasiai.schemas import Evidence

logger = logging.getLogger(__name__)

# Source document names → tool categories
_GAP_KEYWORDS = {"known_gaps", "known gaps", "gaps"}
_PROPOSAL_KEYWORDS = {"past_proposals", "past proposals", "proposals"}


class AgentToolkit:
    """
    Provides tools for the planner agent to call during reasoning.

    Wraps the MemoryStore with focused retrieval strategies and
    tracks all evidence returned for deduplication.
    """

    AVAILABLE_TOOLS = {
        "search_memory": "Search all institutional memory with a custom query. Use for general evidence gathering.",
        "check_gaps": "Search specifically for known gaps and limitations. Use to find counter-evidence or confirm unsupported capabilities.",
        "search_past_proposals": "Search past proposals for precedent. Use to find similar requirements from previous RFPs and their outcomes.",
    }

    def __init__(
        self,
        memory_store: MemoryStore,
        top_k: Optional[int] = None,
    ):
        self._memory = memory_store
        self._top_k = top_k or config.RETRIEVAL_TOP_K
        self._seen_chunk_ids: set[str] = set()
        self._all_evidence: list[Evidence] = []

    def execute(self, tool_name: str, query: str) -> list[Evidence]:
        """
        Execute a tool by name and return evidence.

        Args:
            tool_name: One of the AVAILABLE_TOOLS keys.
            query: Search query formulated by the agent.

        Returns:
            List of Evidence objects (deduplicated across calls).

        Raises:
            ValueError: If tool_name is not recognized.
        """
        if tool_name not in self.AVAILABLE_TOOLS:
            raise ValueError(
                f"Unknown tool '{tool_name}'. "
                f"Available: {list(self.AVAILABLE_TOOLS.keys())}"
            )

        logger.info("Agent tool call: %s(query='%s')", tool_name, query[:80])

        if tool_name == "search_memory":
            results = self._search_memory(query)
        elif tool_name == "check_gaps":
            results = self._check_gaps(query)
        elif tool_name == "search_past_proposals":
            results = self._search_past_proposals(query)
        else:
            results = []

        # Deduplicate against previously seen evidence
        new_results = self._deduplicate(results)
        logger.info(
            "  → %d results (%d new, %d already seen)",
            len(results),
            len(new_results),
            len(results) - len(new_results),
        )
        return new_results

    @property
    def all_evidence(self) -> list[Evidence]:
        """All unique evidence accumulated across all tool calls."""
        return list(self._all_evidence)

    def reset(self) -> None:
        """Reset accumulated evidence (call between requirements)."""
        self._seen_chunk_ids.clear()
        self._all_evidence.clear()

    # ── Tool implementations ──────────────────────────────────────────

    def _search_memory(self, query: str) -> list[Evidence]:
        """General search across all institutional memory."""
        return self._memory.retrieve(query, top_k=self._top_k)

    def _check_gaps(self, query: str) -> list[Evidence]:
        """
        Search with emphasis on known gaps.

        Strategy: retrieve more results then boost those from gap documents.
        """
        results = self._memory.retrieve(query, top_k=self._top_k * 2)

        # Partition into gap-related and other evidence
        gap_results = []
        other_results = []
        for ev in results:
            category = ev.metadata.get("category", "").lower()
            source = (ev.source_document or "").lower().replace(".md", "").replace(".txt", "")
            if any(kw in category or kw in source for kw in _GAP_KEYWORDS):
                gap_results.append(ev)
            else:
                other_results.append(ev)

        # Return gap results first, then fill with others up to top_k
        combined = gap_results + other_results
        return combined[: self._top_k]

    def _search_past_proposals(self, query: str) -> list[Evidence]:
        """
        Search with emphasis on past proposals.

        Strategy: retrieve more results then boost those from proposal documents.
        """
        results = self._memory.retrieve(query, top_k=self._top_k * 2)

        # Partition into proposal-related and other evidence
        proposal_results = []
        other_results = []
        for ev in results:
            category = ev.metadata.get("category", "").lower()
            source = (ev.source_document or "").lower().replace(".md", "").replace(".txt", "")
            if any(kw in category or kw in source for kw in _PROPOSAL_KEYWORDS):
                proposal_results.append(ev)
            else:
                other_results.append(ev)

        combined = proposal_results + other_results
        return combined[: self._top_k]

    # ── Deduplication ─────────────────────────────────────────────────

    def _deduplicate(self, evidences: list[Evidence]) -> list[Evidence]:
        """Remove evidence already seen in previous tool calls."""
        new = []
        for ev in evidences:
            chunk_id = ev.chunk_id or ev.id
            if chunk_id not in self._seen_chunk_ids:
                self._seen_chunk_ids.add(chunk_id)
                self._all_evidence.append(ev)
                new.append(ev)
        return new

