"""
Abstract base class for all live data connectors.

Design decisions:
1. INTERFACE CONTRACT: All connectors must implement retrieve() and health_check().
   This guarantees the registry can call them polymorphically.
2. NAME PROPERTY: Each connector declares its own name so Evidence.source_document
   can be prefixed with the connector name for provenance tracing.
3. GRACEFUL DEGRADATION: Subclasses should catch their own exceptions and return []
   rather than propagating — live data is best-effort, not critical path.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from feasiai.schemas import Evidence


class LiveDataConnector(ABC):
    """
    Abstract base class for all live data connectors.

    Implementations must be safe to call even when credentials are absent
    (return empty list) and must not block the pipeline if the external
    service is unreachable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Short identifier for this connector (e.g. 'jira', 'roadmap').

        Used as prefix in Evidence.source_document: "<name>:<resource_id>".
        """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[Evidence]:
        """
        Search this data source for evidence relevant to *query*.

        Args:
            query: Free-text search query (from agent or pipeline).
            top_k: Maximum number of Evidence objects to return.

        Returns:
            List of Evidence objects with source=EvidenceSource.LIVE_DATA.
            Returns [] if credentials are missing or the source is unavailable.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the connector can reach its data source.

        Returns:
            True if the source is reachable and credentials are valid.
            False otherwise (must not raise).
        """

