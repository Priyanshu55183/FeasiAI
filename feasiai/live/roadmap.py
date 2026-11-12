"""
Roadmap / feature-flag file connector.

Design decisions:
1. LOCAL FILE: Reads a YAML or JSON file — no external service required.
   This makes it usable in tests and offline environments with zero config.
2. KEYWORD MATCHING: Performs case-insensitive substring search across all
   text fields (feature name, notes). Simple but effective for structured data.
3. STATUS-AWARE SCORING: Items with status 'in_progress' or 'shipped' get a
   slightly higher relevance score than 'planned' items; 'cancelled' items
   still appear so the agent can detect conflicts.
4. GRACEFUL MISSING FILE: Returns [] if the file doesn't exist — opt-in behavior.
5. ETA INCLUDED: ETA is embedded in the evidence content so the LLM can reason
   about delivery timelines when assessing feasibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from feasiai import config
from feasiai.live.base import LiveDataConnector
from feasiai.schemas import Evidence, EvidenceSource

logger = logging.getLogger(__name__)

# Relevance scores by feature status
_STATUS_SCORES = {
    "shipped": 1.0,
    "in_progress": 0.9,
    "planned": 0.7,
    "cancelled": 0.6,
}


class RoadmapConnector(LiveDataConnector):
    """
    Reads a local YAML or JSON roadmap file and keyword-searches its entries.

    The roadmap file should be a dict with a top-level 'features' list.
    Each feature entry should have at minimum a 'feature' (name) field.
    Optional fields: status, eta, notes, tags.

    Example YAML structure:
        features:
          - feature: "Shopify Integration"
            status: "planned"
            eta: "Q3 2027"
            notes: "Connector planned, not yet funded"
    """

    def __init__(self, file_path: Optional[str] = None):
        """
        Args:
            file_path: Path to the roadmap YAML or JSON file.
                       Defaults to ROADMAP_FILE_PATH config var.
        """
        self._file_path = Path(file_path or config.ROADMAP_FILE_PATH)
        self._features: Optional[list[dict]] = None  # lazy-loaded cache

    @property
    def name(self) -> str:
        return "roadmap"

    def _load(self) -> list[dict]:
        """
        Load and parse the roadmap file (cached after first load).

        Returns:
            List of feature dicts. Returns [] if file doesn't exist or is invalid.
        """
        if self._features is not None:
            return self._features

        if not self._file_path.exists():
            logger.debug(
                "RoadmapConnector: file not found at %s — returning []",
                self._file_path,
            )
            self._features = []
            return self._features

        try:
            suffix = self._file_path.suffix.lower()
            text = self._file_path.read_text(encoding="utf-8")

            if suffix in {".yaml", ".yml"}:
                try:
                    import yaml  # optional dependency
                    data = yaml.safe_load(text)
                except ImportError:
                    # Fallback: try JSON (the user might have used .yaml extension but JSON content)
                    data = json.loads(text)
            elif suffix == ".json":
                data = json.loads(text)
            else:
                logger.warning(
                    "RoadmapConnector: unsupported file format '%s' — returning []", suffix
                )
                self._features = []
                return self._features

            if isinstance(data, dict):
                features = data.get("features", [])
            elif isinstance(data, list):
                features = data
            else:
                features = []

            self._features = [f for f in features if isinstance(f, dict)]
            logger.info(
                "RoadmapConnector: loaded %d features from %s",
                len(self._features),
                self._file_path,
            )
        except Exception as exc:
            logger.warning("RoadmapConnector: failed to load file — %s", exc)
            self._features = []

        return self._features

    def _matches(self, feature: dict, query: str) -> bool:
        """
        Check if a feature entry matches the query via case-insensitive substring search.

        Args:
            feature: Feature dict from the roadmap file.
            query: Search query string.

        Returns:
            True if any text field in the feature contains a query term.
        """
        query_lower = query.lower()
        query_terms = query_lower.split()

        # Fields to search
        searchable = " ".join(
            str(feature.get(field, "")).lower()
            for field in ("feature", "notes", "tags", "eta")
        )

        return any(term in searchable for term in query_terms)

    def _format_feature(self, feature: dict) -> str:
        """
        Format a feature dict into human-readable Evidence content.

        Args:
            feature: Feature dict with name, status, eta, notes fields.

        Returns:
            Formatted string for inclusion in Evidence.content.
        """
        name = feature.get("feature", "(unnamed feature)")
        status = feature.get("status", "unknown").title()
        eta = feature.get("eta", "")
        notes = feature.get("notes", "")

        lines = [f"Feature: {name}", f"Status: {status}"]
        if eta:
            lines.append(f"ETA: {eta}")
        if notes:
            lines.append(f"Notes: {notes}")
        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int = 5) -> list[Evidence]:
        """
        Search roadmap features for entries relevant to query.

        Args:
            query: Free-text query string.
            top_k: Maximum evidence items to return.

        Returns:
            List of Evidence with source=LIVE_DATA and source_document='roadmap:<feature_name>'.
        """
        try:
            features = self._load()
            if not features:
                return []

            matched: list[tuple[float, dict]] = []
            for feature in features:
                if self._matches(feature, query):
                    status = feature.get("status", "").lower()
                    score = _STATUS_SCORES.get(status, 0.5)
                    matched.append((score, feature))

            # Sort by relevance score descending
            matched.sort(key=lambda x: x[0], reverse=True)

            evidences = []
            for score, feature in matched[:top_k]:
                name = feature.get("feature", "unknown")
                content = self._format_feature(feature)
                # Sanitize name for chunk_id
                chunk_key = name.lower().replace(" ", "_")[:50]
                ev = Evidence(
                    content=content,
                    source=EvidenceSource.LIVE_DATA,
                    source_document=f"roadmap:{chunk_key}",
                    chunk_id=f"roadmap:{chunk_key}",
                    relevance_score=score,
                    metadata={
                        "connector": "roadmap",
                        "feature_name": name,
                        "status": feature.get("status", "unknown"),
                        "eta": feature.get("eta", ""),
                    },
                )
                evidences.append(ev)

            return evidences

        except Exception as exc:
            logger.warning("RoadmapConnector: retrieve() failed — %s", exc)
            return []

    def health_check(self) -> bool:
        """
        Verify the roadmap file exists and is parseable.

        Returns:
            True if file loads at least one feature successfully.
        """
        try:
            features = self._load()
            return len(features) > 0
        except Exception:
            return False

    def reload(self) -> None:
        """Force re-read the roadmap file on next retrieve() call."""
        self._features = None

