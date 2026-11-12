"""
Live data connectors — external sources for EvidenceSource.LIVE_DATA.

Design decisions:
1. ABSTRACT BASE: All connectors implement LiveDataConnector, making them
   interchangeable and testable via dependency injection.
2. OPT-IN / FAIL-SAFE: Connectors return [] if credentials/files are missing.
   The pipeline works identically to Phase 1 when no connectors are registered.
3. REGISTRY PATTERN: LiveDataRegistry aggregates multiple connectors, so the
   agent can search all live sources with a single call.
4. EVIDENCE PROVENANCE: All live evidence uses EvidenceSource.LIVE_DATA and
   source_document formatted as "<connector_name>:<resource_id>" for auditability.
"""

from feasiai.live.base import LiveDataConnector
from feasiai.live.registry import LiveDataRegistry
from feasiai.live.jira import JiraConnector
from feasiai.live.roadmap import RoadmapConnector

__all__ = [
    "LiveDataConnector",
    "LiveDataRegistry",
    "JiraConnector",
    "RoadmapConnector",
]

