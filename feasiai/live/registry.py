"""
Registry that aggregates multiple live data connectors.

Design decisions:
1. AGGREGATION: retrieve_all() fans out to all registered connectors and
   merges results, deduplicated by chunk_id. This is the single entry point
   for the agent's search_live_data tool.
2. PER-CONNECTOR BUDGET: Each connector gets top_k results independently,
   then results are merged and re-ranked by relevance_score. This prevents
   one dominant connector from crowding out others.
3. HEALTH VISIBILITY: health_check_all() provides a dict of connector_name →
   bool so the UI/logs can surface which live sources are reachable.
4. FAIL-SAFE FANOUT: If one connector throws an unexpected exception during
   retrieve_all(), it is logged and skipped — the registry never propagates
   individual connector failures.
5. EMPTY REGISTRY: When no connectors are registered, retrieve_all() returns
   [] and the pipeline is identical to Phase 1.
"""

from __future__ import annotations

import logging
from typing import Optional

from feasiai.live.base import LiveDataConnector
from feasiai.schemas import Evidence

logger = logging.getLogger(__name__)


class LiveDataRegistry:
    """
    Central registry for all live data connectors.

    Connectors are registered by name and queried together via retrieve_all().
    The registry is designed to be injected into the pipeline and agent toolkit
    as an optional dependency — if not provided, all live data features are skipped.
    """

    def __init__(self) -> None:
        self._connectors: dict[str, LiveDataConnector] = {}

    def register(self, connector: LiveDataConnector) -> None:
        """
        Register a connector with the registry.

        Args:
            connector: A LiveDataConnector implementation.

        Raises:
            ValueError: If a connector with the same name is already registered.
        """
        if connector.name in self._connectors:
            raise ValueError(
                f"A connector named '{connector.name}' is already registered. "
                "Call unregister() first or use a unique name."
            )
        self._connectors[connector.name] = connector
        logger.info("LiveDataRegistry: registered connector '%s'", connector.name)

    def unregister(self, name: str) -> None:
        """
        Remove a connector from the registry.

        Args:
            name: The connector name to remove.
        """
        if name in self._connectors:
            del self._connectors[name]
            logger.info("LiveDataRegistry: unregistered connector '%s'", name)

    @property
    def active_connectors(self) -> list[str]:
        """Names of all currently registered connectors."""
        return list(self._connectors.keys())

    def retrieve_all(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Evidence]:
        """
        Query all registered connectors and return merged, ranked evidence.

        Each connector is queried for up to top_k results, then all results
        are merged and sorted by relevance_score descending. The final list
        is capped at top_k * len(connectors) to avoid context explosion.

        Args:
            query: Free-text query string.
            top_k: Maximum results *per connector*.

        Returns:
            Deduplicated list of Evidence sorted by relevance_score descending.
        """
        if not self._connectors:
            return []

        all_evidences: list[Evidence] = []
        seen_chunk_ids: set[str] = set()

        for name, connector in self._connectors.items():
            try:
                results = connector.retrieve(query, top_k=top_k)
                for ev in results:
                    chunk_id = ev.chunk_id or ev.id
                    if chunk_id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk_id)
                        all_evidences.append(ev)
                logger.debug(
                    "LiveDataRegistry: connector '%s' returned %d results",
                    name,
                    len(results),
                )
            except Exception as exc:
                logger.warning(
                    "LiveDataRegistry: connector '%s' raised an unexpected error — %s",
                    name,
                    exc,
                )

        # Sort by relevance descending
        all_evidences.sort(key=lambda ev: ev.relevance_score, reverse=True)
        return all_evidences

    def health_check_all(self) -> dict[str, bool]:
        """
        Run health checks for all registered connectors.

        Returns:
            Dict mapping connector name → health status (True = healthy).
        """
        results: dict[str, bool] = {}
        for name, connector in self._connectors.items():
            try:
                results[name] = connector.health_check()
            except Exception as exc:
                logger.warning(
                    "LiveDataRegistry: health_check for '%s' raised — %s", name, exc
                )
                results[name] = False
        return results

    def __len__(self) -> int:
        return len(self._connectors)

    def __bool__(self) -> bool:
        return bool(self._connectors)

