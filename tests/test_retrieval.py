"""Tests for hybrid retrieval (memory store) — uses real Chroma + BM25."""

import pytest
from pathlib import Path

from feasiai.retrieval.memory_store import MemoryStore


@pytest.fixture
def memory_store(tmp_path):
    """Create a fresh memory store with temp persistence."""
    store = MemoryStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_memory",
    )
    return store


@pytest.fixture
def populated_store(memory_store, sample_memory_dir):
    """Memory store loaded with sample documents."""
    memory_store.ingest_directory(str(sample_memory_dir))
    return memory_store


class TestIngestion:
    def test_ingest_directory(self, memory_store, sample_memory_dir):
        count = memory_store.ingest_directory(str(sample_memory_dir))
        assert count > 0
        assert memory_store.chunk_count > 0

    def test_ingest_empty_directory(self, memory_store, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        count = memory_store.ingest_directory(str(empty_dir))
        assert count == 0

    def test_ingest_nonexistent_directory(self, memory_store):
        count = memory_store.ingest_directory("/nonexistent/path")
        assert count == 0

    def test_idempotent_ingestion(self, memory_store, sample_memory_dir):
        """Ingesting the same directory twice should not duplicate chunks."""
        count1 = memory_store.ingest_directory(str(sample_memory_dir))
        count2 = memory_store.ingest_directory(str(sample_memory_dir))
        assert count1 == count2
        # ChromaDB upserts, so chunk count should stay the same
        assert memory_store.chunk_count == count1


class TestRetrieval:
    def test_retrieve_relevant(self, populated_store):
        """Should find evidence for a supported capability."""
        results = populated_store.retrieve("SAML 2.0 SSO authentication")
        assert len(results) > 0
        # Should find content about SSO
        all_content = " ".join(e.content for e in results)
        assert "SAML" in all_content or "SSO" in all_content

    def test_retrieve_gap(self, populated_store):
        """Should find content about known gaps."""
        results = populated_store.retrieve("Shopify e-commerce integration")
        assert len(results) > 0
        all_content = " ".join(e.content for e in results).lower()
        assert "shopify" in all_content or "e-commerce" in all_content

    def test_retrieve_irrelevant(self, populated_store):
        """Query about something completely unrelated returns low scores."""
        results = populated_store.retrieve("quantum computing blockchain")
        # Should still return something (nearest neighbors) but with low scores
        if results:
            assert all(e.relevance_score < 0.5 for e in results)

    def test_retrieve_empty_store(self, memory_store):
        """Empty store should return empty results."""
        results = memory_store.retrieve("any query")
        assert results == []

    def test_evidence_provenance(self, populated_store):
        """Retrieved evidence should carry source metadata."""
        results = populated_store.retrieve("SSO SAML authentication")
        for ev in results:
            assert ev.source_document is not None
            assert ev.chunk_id is not None
            assert ev.source.value == "memory"

    def test_top_k_limit(self, populated_store):
        """Should respect top_k parameter."""
        results = populated_store.retrieve("authentication", top_k=2)
        assert len(results) <= 2


class TestChunking:
    def test_chunk_text_short(self):
        store = MemoryStore.__new__(MemoryStore)
        chunks = store._chunk_text("Short text")
        assert len(chunks) == 1

    def test_chunk_text_long(self):
        store = MemoryStore.__new__(MemoryStore)
        long_text = "This is a sentence. " * 200  # ~4000 chars
        chunks = store._chunk_text(long_text)
        assert len(chunks) > 1

    def test_chunk_text_empty(self):
        store = MemoryStore.__new__(MemoryStore)
        chunks = store._chunk_text("")
        assert len(chunks) == 0


class TestClear:
    def test_clear_store(self, populated_store):
        assert populated_store.chunk_count > 0
        populated_store.clear()
        assert populated_store.chunk_count == 0
