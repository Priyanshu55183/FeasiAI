"""
Tests for Phase 3A: Live Data Layer.

Coverage:
- RoadmapConnector: load YAML, search, return Evidence with LIVE_DATA source
- JiraConnector: mock requests.Session, verify JQL, Evidence format, no-credentials
- LiveDataRegistry: register/unregister, retrieve_all, health_check_all, dedup
- AgentToolkit: search_live_data tool present when registry non-empty, absent otherwise
- ConflictDetection: memory says Available + live status cancelled → HIGH conflict
- PlannerAgent: accepts live_registry, passes it to toolkit
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from feasiai.live.base import LiveDataConnector
from feasiai.live.jira import JiraConnector
from feasiai.live.registry import LiveDataRegistry
from feasiai.live.roadmap import RoadmapConnector
from feasiai.live.conflict import detect_conflicts, ConflictSeverity
from feasiai.schemas import (
    Classification,
    ConflictWarning,
    Evidence,
    EvidenceSource,
    FeasibilityLabel,
    Requirement,
)


# ─── Fixtures ───────────────────────────────────────────────────────────────


SAMPLE_ROADMAP_YAML = """\
features:
  - feature: "Shopify Integration"
    status: "planned"
    eta: "Q3 2027"
    notes: "Connector planned, not yet funded"
  - feature: "HIPAA Compliance"
    status: "in_progress"
    eta: "Q1 2027"
    notes: "BAA process underway"
  - feature: "Mobile App"
    status: "cancelled"
    notes: "Deprioritized in favor of responsive web"
  - feature: "SSO Integration"
    status: "shipped"
    eta: "Q4 2026"
    notes: "SAML 2.0 and OIDC support shipped"
"""

SAMPLE_ROADMAP_JSON = json.dumps({
    "features": [
        {"feature": "Shopify Integration", "status": "planned", "eta": "Q3 2027"},
        {"feature": "Mobile App", "status": "cancelled"},
    ]
})


@pytest.fixture
def roadmap_yaml_file(tmp_path):
    """Create a temporary YAML roadmap file."""
    p = tmp_path / "roadmap.yaml"
    p.write_text(SAMPLE_ROADMAP_YAML, encoding="utf-8")
    return str(p)


@pytest.fixture
def roadmap_json_file(tmp_path):
    """Create a temporary JSON roadmap file."""
    p = tmp_path / "roadmap.json"
    p.write_text(SAMPLE_ROADMAP_JSON, encoding="utf-8")
    return str(p)


@pytest.fixture
def roadmap_connector(roadmap_yaml_file):
    """RoadmapConnector pointed at the fixture YAML."""
    return RoadmapConnector(file_path=roadmap_yaml_file)


def _make_live_evidence(
    content: str = "Feature: Test\nStatus: Planned",
    status: str = "planned",
    feature_name: str = "Test Feature",
    source_doc: str = "roadmap:test",
) -> Evidence:
    return Evidence(
        content=content,
        source=EvidenceSource.LIVE_DATA,
        source_document=source_doc,
        chunk_id=source_doc,
        relevance_score=0.7,
        metadata={"connector": "roadmap", "status": status, "feature_name": feature_name},
    )


def _make_memory_evidence(content: str = "SSO is supported") -> Evidence:
    return Evidence(
        content=content,
        source=EvidenceSource.MEMORY,
        source_document="capabilities.md",
        chunk_id="cap-sso-001",
        relevance_score=0.9,
    )


def _make_classification(
    label: FeasibilityLabel = FeasibilityLabel.AVAILABLE,
    req_id: str = "req-001",
    memory_ev: Optional[Evidence] = None,
) -> Classification:
    evs = [memory_ev] if memory_ev else [_make_memory_evidence()]
    return Classification(
        requirement_id=req_id,
        label=label,
        confidence=0.85,
        supporting_evidence=evs,
        reasoning="Evidence supports availability.",
        what_would_change="Nothing significant.",
        draft_response="Yes, this is supported.",
    )


# ─── Abstract Base Tests ─────────────────────────────────────────────────────


class TestLiveDataConnectorBase:
    """Verify the abstract interface contract."""

    def test_cannot_instantiate_abstract(self):
        """LiveDataConnector cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LiveDataConnector()

    def test_concrete_must_implement_retrieve(self):
        """Subclass without retrieve() raises TypeError."""
        class BadConnector(LiveDataConnector):
            @property
            def name(self):
                return "bad"
            def health_check(self):
                return False

        with pytest.raises(TypeError):
            BadConnector()

    def test_concrete_must_implement_health_check(self):
        """Subclass without health_check() raises TypeError."""
        class BadConnector(LiveDataConnector):
            @property
            def name(self):
                return "bad"
            def retrieve(self, query, top_k=5):
                return []

        with pytest.raises(TypeError):
            BadConnector()


# ─── RoadmapConnector Tests ──────────────────────────────────────────────────


class TestRoadmapConnector:
    def test_name_is_roadmap(self, roadmap_connector):
        assert roadmap_connector.name == "roadmap"

    def test_loads_features_from_yaml(self, roadmap_connector):
        features = roadmap_connector._load()
        assert len(features) == 4

    def test_retrieve_returns_evidence_with_live_data_source(self, roadmap_connector):
        results = roadmap_connector.retrieve("Shopify")
        assert len(results) == 1
        assert results[0].source == EvidenceSource.LIVE_DATA

    def test_retrieve_content_contains_feature_name(self, roadmap_connector):
        results = roadmap_connector.retrieve("Shopify")
        assert "Shopify Integration" in results[0].content

    def test_retrieve_content_contains_status(self, roadmap_connector):
        results = roadmap_connector.retrieve("Shopify")
        assert "Planned" in results[0].content or "planned" in results[0].content.lower()

    def test_retrieve_content_contains_eta(self, roadmap_connector):
        results = roadmap_connector.retrieve("Shopify")
        assert "Q3 2027" in results[0].content

    def test_retrieve_source_document_format(self, roadmap_connector):
        results = roadmap_connector.retrieve("Shopify")
        assert results[0].source_document.startswith("roadmap:")

    def test_retrieve_metadata_has_status(self, roadmap_connector):
        results = roadmap_connector.retrieve("HIPAA")
        assert results[0].metadata["status"] == "in_progress"

    def test_retrieve_shipped_has_highest_score(self, roadmap_connector):
        results = roadmap_connector.retrieve("SSO")
        assert len(results) == 1
        assert results[0].relevance_score == 1.0  # shipped = 1.0

    def test_retrieve_cancelled_still_returned(self, roadmap_connector):
        results = roadmap_connector.retrieve("Mobile")
        assert len(results) == 1
        assert results[0].metadata["status"] == "cancelled"

    def test_retrieve_top_k_respected(self, roadmap_connector):
        # Query that matches all features via broad term
        # Use a term present in multiple feature notes
        results = roadmap_connector.retrieve("Integration Compliance App SSO", top_k=2)
        assert len(results) <= 2

    def test_retrieve_missing_file_returns_empty(self, tmp_path):
        connector = RoadmapConnector(file_path=str(tmp_path / "nonexistent.yaml"))
        assert connector.retrieve("anything") == []

    def test_retrieve_empty_query_returns_empty_or_all(self, roadmap_connector):
        """Empty query shouldn't crash — returns [] since no terms match."""
        results = roadmap_connector.retrieve("")
        # With no query terms, _matches returns False for all
        assert isinstance(results, list)

    def test_loads_json_file(self, roadmap_json_file):
        connector = RoadmapConnector(file_path=roadmap_json_file)
        features = connector._load()
        assert len(features) == 2

    def test_health_check_true_when_file_exists(self, roadmap_connector):
        assert roadmap_connector.health_check() is True

    def test_health_check_false_when_file_missing(self, tmp_path):
        connector = RoadmapConnector(file_path=str(tmp_path / "missing.yaml"))
        assert connector.health_check() is False

    def test_reload_forces_re_read(self, roadmap_yaml_file):
        connector = RoadmapConnector(file_path=roadmap_yaml_file)
        # First load
        f1 = connector._load()
        # Reload
        connector.reload()
        f2 = connector._load()
        assert f1 == f2  # Same data, just re-read

    def test_retrieve_returns_list_evidence_objects(self, roadmap_connector):
        results = roadmap_connector.retrieve("HIPAA")
        for ev in results:
            assert isinstance(ev, Evidence)


# ─── JiraConnector Tests ─────────────────────────────────────────────────────


def _make_jira_response(issues: list[dict]) -> MagicMock:
    """Create a mock requests response for Jira."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"issues": issues}
    resp.raise_for_status = MagicMock()
    return resp


def _jira_issue(key: str, summary: str, status: str = "In Progress", description: str = "") -> dict:
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "status": {"name": status},
            "description": description or None,
        },
    }


class TestJiraConnector:
    def _make_configured_connector(self, session=None):
        return JiraConnector(
            base_url="https://example.atlassian.net",
            email="user@example.com",
            api_token="token123",
            requests_session=session,
        )

    def test_name_is_jira(self):
        connector = self._make_configured_connector()
        assert connector.name == "jira"

    def test_retrieve_returns_evidence_list(self):
        session = MagicMock()
        session.post.return_value = _make_jira_response([
            _jira_issue("ENG-1", "SAML SSO support"),
        ])
        connector = self._make_configured_connector(session=session)
        results = connector.retrieve("SSO")
        assert len(results) == 1
        assert isinstance(results[0], Evidence)

    def test_retrieve_evidence_has_live_data_source(self):
        session = MagicMock()
        session.post.return_value = _make_jira_response([
            _jira_issue("ENG-1", "SAML SSO support"),
        ])
        connector = self._make_configured_connector(session=session)
        results = connector.retrieve("SSO")
        assert results[0].source == EvidenceSource.LIVE_DATA

    def test_retrieve_source_document_format(self):
        session = MagicMock()
        session.post.return_value = _make_jira_response([
            _jira_issue("ENG-42", "Feature X"),
        ])
        connector = self._make_configured_connector(session=session)
        results = connector.retrieve("feature")
        assert results[0].source_document == "jira:ENG-42"
        assert results[0].chunk_id == "jira:ENG-42"

    def test_retrieve_content_format(self):
        session = MagicMock()
        session.post.return_value = _make_jira_response([
            _jira_issue("ENG-1", "SAML SSO", status="Done", description="Shipped to prod"),
        ])
        connector = self._make_configured_connector(session=session)
        results = connector.retrieve("SSO")
        content = results[0].content
        assert "[ENG-1]" in content
        assert "SAML SSO" in content
        assert "Done" in content

    def test_retrieve_jql_construction(self):
        session = MagicMock()
        session.post.return_value = _make_jira_response([])
        connector = self._make_configured_connector(session=session)
        connector.retrieve("mobile app requirements")

        call_kwargs = session.post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs.get("json", {})
        assert "mobile app requirements" in payload["jql"]

    def test_retrieve_project_keys_in_jql(self):
        session = MagicMock()
        session.post.return_value = _make_jira_response([])
        connector = JiraConnector(
            base_url="https://example.atlassian.net",
            email="user@example.com",
            api_token="token123",
            project_keys="ENG,INFRA",
            requests_session=session,
        )
        connector.retrieve("SSO")

        call_kwargs = session.post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs.get("json", {})
        assert "ENG" in payload["jql"]
        assert "INFRA" in payload["jql"]

    def test_retrieve_top_k_in_payload(self):
        session = MagicMock()
        session.post.return_value = _make_jira_response([])
        connector = self._make_configured_connector(session=session)
        connector.retrieve("test", top_k=3)

        call_kwargs = session.post.call_args
        payload = call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs.kwargs.get("json", {})
        assert payload["maxResults"] == 3

    def test_retrieve_http_error_returns_empty(self):
        session = MagicMock()
        from requests import HTTPError
        session.post.side_effect = HTTPError("403 Forbidden")
        connector = self._make_configured_connector(session=session)
        results = connector.retrieve("test")
        assert results == []

    def test_health_check_returns_true_on_200(self):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 200
        session.get.return_value = resp
        connector = self._make_configured_connector(session=session)
        assert connector.health_check() is True

    def test_health_check_returns_false_on_401(self):
        session = MagicMock()
        resp = MagicMock()
        resp.status_code = 401
        session.get.return_value = resp
        connector = self._make_configured_connector(session=session)
        assert connector.health_check() is False

    def test_health_check_returns_false_on_exception(self):
        session = MagicMock()
        session.get.side_effect = Exception("Network error")
        connector = self._make_configured_connector(session=session)
        assert connector.health_check() is False


class TestJiraConnectorNoCredentials:
    """JiraConnector returns [] gracefully when env vars are missing."""

    def test_missing_base_url_returns_empty(self):
        connector = JiraConnector(base_url="", email="a@b.com", api_token="tok")
        assert connector.retrieve("SSO") == []

    def test_missing_email_returns_empty(self):
        connector = JiraConnector(base_url="https://x.atlassian.net", email="", api_token="tok")
        assert connector.retrieve("SSO") == []

    def test_missing_token_returns_empty(self):
        connector = JiraConnector(base_url="https://x.atlassian.net", email="a@b.com", api_token="")
        assert connector.retrieve("SSO") == []

    def test_health_check_false_when_not_configured(self):
        connector = JiraConnector(base_url="", email="", api_token="")
        assert connector.health_check() is False

    def test_is_configured_property(self):
        configured = JiraConnector(
            base_url="https://x.atlassian.net", email="a@b.com", api_token="tok"
        )
        assert configured._is_configured is True

        unconfigured = JiraConnector(base_url="", email="a@b.com", api_token="tok")
        assert unconfigured._is_configured is False


# ─── LiveDataRegistry Tests ──────────────────────────────────────────────────


class MockConnector(LiveDataConnector):
    """Simple mock connector for registry tests."""

    def __init__(self, name: str, results: list[Evidence] = None, healthy: bool = True):
        self._name = name
        self._results = results or []
        self._healthy = healthy

    @property
    def name(self) -> str:
        return self._name

    def retrieve(self, query: str, top_k: int = 5) -> list[Evidence]:
        return self._results[:top_k]

    def health_check(self) -> bool:
        return self._healthy


class TestLiveDataRegistry:
    def test_register_connector(self):
        registry = LiveDataRegistry()
        registry.register(MockConnector("test"))
        assert "test" in registry.active_connectors

    def test_register_duplicate_raises(self):
        registry = LiveDataRegistry()
        registry.register(MockConnector("test"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(MockConnector("test"))

    def test_unregister_connector(self):
        registry = LiveDataRegistry()
        registry.register(MockConnector("test"))
        registry.unregister("test")
        assert "test" not in registry.active_connectors

    def test_active_connectors_returns_names(self):
        registry = LiveDataRegistry()
        registry.register(MockConnector("a"))
        registry.register(MockConnector("b"))
        assert set(registry.active_connectors) == {"a", "b"}

    def test_retrieve_all_returns_merged_results(self):
        ev1 = _make_live_evidence(source_doc="roadmap:feat1")
        ev2 = _make_live_evidence(source_doc="jira:ENG-1")
        registry = LiveDataRegistry()
        registry.register(MockConnector("roadmap", results=[ev1]))
        registry.register(MockConnector("jira", results=[ev2]))

        results = registry.retrieve_all("test query")
        assert len(results) == 2
        source_docs = {r.source_document for r in results}
        assert "roadmap:feat1" in source_docs
        assert "jira:ENG-1" in source_docs

    def test_retrieve_all_deduplicates_by_chunk_id(self):
        # Same chunk_id from two connectors
        ev1 = _make_live_evidence(source_doc="roadmap:feat1")
        ev2 = Evidence(
            content="Duplicate",
            source=EvidenceSource.LIVE_DATA,
            source_document="roadmap:feat1",
            chunk_id="roadmap:feat1",  # same chunk_id
            relevance_score=0.5,
        )
        registry = LiveDataRegistry()
        registry.register(MockConnector("a", results=[ev1]))
        registry.register(MockConnector("b", results=[ev2]))

        results = registry.retrieve_all("test")
        assert len(results) == 1

    def test_retrieve_all_sorted_by_relevance(self):
        ev_low = Evidence(
            content="Low relevance",
            source=EvidenceSource.LIVE_DATA,
            source_document="a:1",
            chunk_id="a:1",
            relevance_score=0.3,
        )
        ev_high = Evidence(
            content="High relevance",
            source=EvidenceSource.LIVE_DATA,
            source_document="b:1",
            chunk_id="b:1",
            relevance_score=0.9,
        )
        registry = LiveDataRegistry()
        registry.register(MockConnector("a", results=[ev_low]))
        registry.register(MockConnector("b", results=[ev_high]))

        results = registry.retrieve_all("test")
        assert results[0].relevance_score >= results[1].relevance_score

    def test_retrieve_all_empty_registry_returns_empty(self):
        registry = LiveDataRegistry()
        assert registry.retrieve_all("test") == []

    def test_retrieve_all_handles_connector_exception(self):
        """Registry skips connectors that raise unexpectedly."""
        class BrokenConnector(LiveDataConnector):
            @property
            def name(self):
                return "broken"
            def retrieve(self, query, top_k=5):
                raise RuntimeError("Network down")
            def health_check(self):
                return False

        ev = _make_live_evidence()
        registry = LiveDataRegistry()
        registry.register(BrokenConnector())
        registry.register(MockConnector("good", results=[ev]))

        results = registry.retrieve_all("test")
        assert len(results) == 1  # Only good connector's result

    def test_health_check_all_returns_dict(self):
        registry = LiveDataRegistry()
        registry.register(MockConnector("healthy", healthy=True))
        registry.register(MockConnector("sick", healthy=False))

        status = registry.health_check_all()
        assert status["healthy"] is True
        assert status["sick"] is False

    def test_len_returns_connector_count(self):
        registry = LiveDataRegistry()
        assert len(registry) == 0
        registry.register(MockConnector("a"))
        assert len(registry) == 1

    def test_bool_true_when_has_connectors(self):
        registry = LiveDataRegistry()
        assert not registry
        registry.register(MockConnector("a"))
        assert registry


# ─── AgentToolkit Live Data Tool Tests ───────────────────────────────────────


class TestAgentToolkitWithLiveData:
    def _make_memory_store(self):
        from feasiai.retrieval.memory_store import MemoryStore
        store = MemoryStore(ephemeral=True)
        return store

    def test_search_live_data_not_in_tools_without_registry(self):
        from feasiai.agent.tools import AgentToolkit
        store = self._make_memory_store()
        toolkit = AgentToolkit(memory_store=store, live_registry=None)
        assert "search_live_data" not in toolkit.AVAILABLE_TOOLS

    def test_search_live_data_not_in_tools_with_empty_registry(self):
        from feasiai.agent.tools import AgentToolkit
        store = self._make_memory_store()
        registry = LiveDataRegistry()  # empty
        toolkit = AgentToolkit(memory_store=store, live_registry=registry)
        assert "search_live_data" not in toolkit.AVAILABLE_TOOLS

    def test_search_live_data_in_tools_with_non_empty_registry(self):
        from feasiai.agent.tools import AgentToolkit
        store = self._make_memory_store()
        registry = LiveDataRegistry()
        registry.register(MockConnector("roadmap"))
        toolkit = AgentToolkit(memory_store=store, live_registry=registry)
        assert "search_live_data" in toolkit.AVAILABLE_TOOLS

    def test_execute_search_live_data_calls_registry(self):
        from feasiai.agent.tools import AgentToolkit
        store = self._make_memory_store()
        ev = _make_live_evidence()
        registry = LiveDataRegistry()
        registry.register(MockConnector("roadmap", results=[ev]))
        toolkit = AgentToolkit(memory_store=store, live_registry=registry)
        results = toolkit.execute("search_live_data", "Shopify integration")
        assert len(results) == 1
        assert results[0].source == EvidenceSource.LIVE_DATA

    def test_execute_search_live_data_without_registry_raises(self):
        """search_live_data not in AVAILABLE_TOOLS without registry → ValueError."""
        from feasiai.agent.tools import AgentToolkit
        store = self._make_memory_store()
        toolkit = AgentToolkit(memory_store=store, live_registry=None)
        with pytest.raises(ValueError, match="Unknown tool"):
            toolkit.execute("search_live_data", "test")

    def test_base_tools_always_available(self):
        from feasiai.agent.tools import AgentToolkit
        store = self._make_memory_store()
        toolkit = AgentToolkit(memory_store=store)
        for tool in ("search_memory", "check_gaps", "search_past_proposals"):
            assert tool in toolkit.AVAILABLE_TOOLS

    def test_live_data_evidence_deduplicated(self):
        """Live data evidence is deduplicated with memory evidence."""
        from feasiai.agent.tools import AgentToolkit
        store = self._make_memory_store()
        ev = _make_live_evidence(source_doc="roadmap:feat1")
        registry = LiveDataRegistry()
        registry.register(MockConnector("roadmap", results=[ev]))
        toolkit = AgentToolkit(memory_store=store, live_registry=registry)

        # Execute once
        r1 = toolkit.execute("search_live_data", "shopify")
        assert len(r1) == 1

        # Execute again — same evidence should be deduped
        r2 = toolkit.execute("search_live_data", "shopify")
        assert len(r2) == 0


# ─── Conflict Detection Tests ─────────────────────────────────────────────────


class TestConflictDetection:
    def test_no_conflict_when_no_live_evidence(self):
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        warnings = detect_conflicts(cls, [])
        assert warnings == []

    def test_high_conflict_available_plus_cancelled(self):
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        live_ev = _make_live_evidence(status="cancelled", feature_name="Mobile App")
        warnings = detect_conflicts(cls, [live_ev])
        assert len(warnings) == 1
        assert warnings[0].severity == ConflictSeverity.HIGH.value
        assert warnings[0].live_status == "cancelled"

    def test_high_conflict_configurable_plus_cancelled(self):
        cls = _make_classification(label=FeasibilityLabel.CONFIGURABLE)
        live_ev = _make_live_evidence(status="cancelled", feature_name="Feature X")
        warnings = detect_conflicts(cls, [live_ev])
        assert len(warnings) == 1
        assert warnings[0].severity == ConflictSeverity.HIGH.value

    def test_medium_conflict_available_plus_planned(self):
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        live_ev = _make_live_evidence(status="planned", feature_name="Shopify Integration")
        warnings = detect_conflicts(cls, [live_ev])
        assert len(warnings) == 1
        assert warnings[0].severity == ConflictSeverity.MEDIUM.value

    def test_medium_conflict_contains_eta(self):
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        ev = Evidence(
            content="Feature: X\nStatus: Planned\nETA: Q3 2027",
            source=EvidenceSource.LIVE_DATA,
            source_document="roadmap:x",
            chunk_id="roadmap:x",
            relevance_score=0.7,
            metadata={"status": "planned", "feature_name": "X", "eta": "Q3 2027"},
        )
        warnings = detect_conflicts(cls, [ev])
        assert "Q3 2027" in warnings[0].description

    def test_low_conflict_requires_dev_plus_shipped(self):
        cls = _make_classification(label=FeasibilityLabel.REQUIRES_DEVELOPMENT)
        live_ev = _make_live_evidence(status="shipped", feature_name="SSO")
        warnings = detect_conflicts(cls, [live_ev])
        assert len(warnings) == 1
        assert warnings[0].severity == ConflictSeverity.LOW.value

    def test_no_conflict_available_plus_shipped(self):
        """Available + shipped = consistent, no conflict."""
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        live_ev = _make_live_evidence(status="shipped")
        warnings = detect_conflicts(cls, [live_ev])
        assert warnings == []

    def test_no_conflict_unclear_plus_cancelled(self):
        """Unsupported/Unclear + cancelled = consistent, no conflict."""
        cls = _make_classification(label=FeasibilityLabel.UNSUPPORTED_UNCLEAR)
        live_ev = _make_live_evidence(status="cancelled")
        warnings = detect_conflicts(cls, [live_ev])
        assert warnings == []

    def test_conflict_has_requirement_id(self):
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE, req_id="req-xyz")
        live_ev = _make_live_evidence(status="cancelled")
        warnings = detect_conflicts(cls, [live_ev])
        assert warnings[0].requirement_id == "req-xyz"

    def test_conflict_has_live_evidence_id(self):
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        live_ev = _make_live_evidence(status="cancelled")
        warnings = detect_conflicts(cls, [live_ev])
        assert warnings[0].live_evidence_id == live_ev.id

    def test_conflict_has_memory_evidence_id(self):
        mem_ev = _make_memory_evidence()
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE, memory_ev=mem_ev)
        live_ev = _make_live_evidence(status="cancelled")
        warnings = detect_conflicts(cls, [live_ev])
        assert warnings[0].memory_evidence_id == mem_ev.id

    def test_multiple_live_evidences_multiple_conflicts(self):
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        ev1 = _make_live_evidence(status="cancelled", source_doc="roadmap:a", feature_name="A")
        ev2 = Evidence(
            content="Feature: B\nStatus: Cancelled",
            source=EvidenceSource.LIVE_DATA,
            source_document="roadmap:b",
            chunk_id="roadmap:b",
            relevance_score=0.6,
            metadata={"status": "cancelled", "feature_name": "B"},
        )
        warnings = detect_conflicts(cls, [ev1, ev2])
        assert len(warnings) == 2

    def test_live_evidence_without_status_skipped(self):
        """Evidence with no status metadata produces no warning."""
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        ev = Evidence(
            content="Some live data",
            source=EvidenceSource.LIVE_DATA,
            source_document="jira:ENG-1",
            chunk_id="jira:ENG-1",
            relevance_score=0.8,
            metadata={"connector": "jira"},  # no 'status' key
        )
        warnings = detect_conflicts(cls, [ev])
        assert warnings == []

    def test_conflict_warning_is_pydantic_model(self):
        cls = _make_classification(label=FeasibilityLabel.AVAILABLE)
        live_ev = _make_live_evidence(status="cancelled")
        warnings = detect_conflicts(cls, [live_ev])
        assert isinstance(warnings[0], ConflictWarning)


# ─── PlannerAgent with Live Registry ─────────────────────────────────────────


class TestPlannerAgentWithLiveRegistry:
    def _make_llm(self):
        llm = MagicMock()
        llm.chat_json.return_value = {
            "action": "classify",
            "thought": "Evidence is sufficient.",
            "label": "Available",
            "confidence": 0.9,
            "reasoning": "SSO is supported.",
            "what_would_change": "Nothing.",
            "draft_response": "Yes, SSO is supported.",
        }
        llm.call_count = 1
        return llm

    def test_planner_accepts_live_registry(self):
        from feasiai.agent.planner import PlannerAgent
        from feasiai.retrieval.memory_store import MemoryStore

        llm = self._make_llm()
        store = MemoryStore(ephemeral=True)
        registry = LiveDataRegistry()
        registry.register(MockConnector("roadmap"))

        # Should not raise
        agent = PlannerAgent(llm, store, live_registry=registry)
        assert agent is not None

    def test_planner_toolkit_has_live_tool_with_registry(self):
        from feasiai.agent.planner import PlannerAgent
        from feasiai.retrieval.memory_store import MemoryStore

        llm = self._make_llm()
        store = MemoryStore(ephemeral=True)
        registry = LiveDataRegistry()
        registry.register(MockConnector("roadmap"))

        agent = PlannerAgent(llm, store, live_registry=registry)
        assert "search_live_data" in agent._toolkit.AVAILABLE_TOOLS

    def test_planner_toolkit_no_live_tool_without_registry(self):
        from feasiai.agent.planner import PlannerAgent
        from feasiai.retrieval.memory_store import MemoryStore

        llm = self._make_llm()
        store = MemoryStore(ephemeral=True)

        agent = PlannerAgent(llm, store, live_registry=None)
        assert "search_live_data" not in agent._toolkit.AVAILABLE_TOOLS


# ─── Schema: ConflictWarning in Classification ───────────────────────────────


class TestConflictWarningSchema:
    def test_classification_has_conflicts_field(self):
        cls = _make_classification()
        assert hasattr(cls, "conflicts")
        assert isinstance(cls.conflicts, list)

    def test_classification_conflicts_default_empty(self):
        cls = _make_classification()
        assert cls.conflicts == []

    def test_classification_with_conflicts(self):
        warning = ConflictWarning(
            requirement_id="req-001",
            severity="high",
            description="Conflict detected",
            live_status="cancelled",
        )
        cls = Classification(
            requirement_id="req-001",
            label=FeasibilityLabel.AVAILABLE,
            confidence=0.8,
            supporting_evidence=[],
            reasoning="test",
            what_would_change="test",
            draft_response="test",
            conflicts=[warning],
        )
        assert len(cls.conflicts) == 1
        assert cls.conflicts[0].severity == "high"

    def test_conflict_warning_is_frozen(self):
        warning = ConflictWarning(
            requirement_id="req-001",
            severity="high",
            description="Conflict",
            live_status="cancelled",
        )
        with pytest.raises(Exception):  # ValidationError or AttributeError
            warning.severity = "low"  # type: ignore

