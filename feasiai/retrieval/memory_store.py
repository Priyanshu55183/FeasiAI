"""
Hybrid memory store: ChromaDB (dense) + BM25 (sparse) with RRF fusion.

Design decisions:
1. HYBRID RETRIEVAL: Dense (semantic) and sparse (lexical) retrieval
   are complementary. Dense catches paraphrases; BM25 catches exact
   terms (product names, acronyms). Reciprocal Rank Fusion (RRF)
   merges their rankings without needing score normalization.

2. CHUNKING STRATEGY: Documents are split into ~500-char chunks with
   100-char overlap. This is a pragmatic default — small enough for
   precise retrieval, large enough for context. The overlap prevents
   losing information at chunk boundaries.

3. METADATA PRESERVATION: Each chunk carries its source document name
   and category. This powers provenance tracking downstream.

4. IDEMPOTENT INGESTION: Documents are identified by name + chunk index.
   Re-ingesting the same document updates rather than duplicates.

5. CONNECTOR PATTERN: The MemoryStore class is the "memory connector."
   In Phase 2, LiveDataStore will follow the same interface pattern,
   making connectors swappable.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import chromadb
from rank_bm25 import BM25Okapi

from feasiai import config
from feasiai.retrieval.embeddings import embed_query, embed_texts
from feasiai.schemas import Evidence, EvidenceSource

logger = logging.getLogger(__name__)

# Chunking parameters
_CHUNK_SIZE = 500        # characters per chunk
_CHUNK_OVERLAP = 100     # overlap between chunks


class MemoryStore:
    """
    Hybrid retrieval over institutional knowledge documents.

    Combines ChromaDB (dense vector search) with BM25 (sparse lexical search)
    using Reciprocal Rank Fusion for result merging.
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        ephemeral: bool = False,
    ):
        self._persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self._collection_name = collection_name or config.MEMORY_COLLECTION_NAME

        # Initialize ChromaDB — use EphemeralClient for tests (avoids
        # singleton conflicts) and PersistentClient for production.
        if ephemeral:
            self._chroma_client = chromadb.EphemeralClient()
        else:
            self._chroma_client = chromadb.PersistentClient(
                path=self._persist_dir,
            )
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index (rebuilt on each load — acceptable for <10k chunks)
        self._bm25_corpus: list[str] = []  # chunk texts
        self._bm25_ids: list[str] = []      # corresponding chunk IDs
        self._bm25_metadata: list[dict] = []  # corresponding metadata
        self._bm25_index: Optional[BM25Okapi] = None

    # ── Ingestion ───────────────────────────────────────────────────────

    def ingest_directory(self, docs_dir: Optional[str] = None) -> int:
        """
        Ingest all .md and .txt files from a directory into the memory store.

        Returns:
            Number of chunks ingested.
        """
        directory = Path(docs_dir or config.MEMORY_DOCS_DIR)
        if not directory.exists():
            logger.warning("Memory docs directory not found: %s", directory)
            return 0

        total_chunks = 0
        for file_path in sorted(directory.iterdir()):
            if file_path.suffix.lower() in (".md", ".txt", ".text", ".markdown"):
                n = self.ingest_document(file_path)
                total_chunks += n
                logger.info("Ingested %s → %d chunks", file_path.name, n)

        # Rebuild BM25 index after all documents are ingested
        self._rebuild_bm25_index()
        logger.info("Total chunks in memory store: %d", total_chunks)
        return total_chunks

    def ingest_document(self, file_path: Path) -> int:
        """
        Ingest a single document: chunk → embed → store in ChromaDB.

        Returns:
            Number of chunks created.
        """
        text = file_path.read_text(encoding="utf-8")
        if not text.strip():
            return 0

        # Extract category from filename (e.g., "product_capabilities" from
        # "product_capabilities.md")
        category = file_path.stem.replace("_", " ").replace("-", " ")

        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        # Generate stable IDs for idempotent ingestion
        ids = [
            self._chunk_id(file_path.name, i)
            for i in range(len(chunks))
        ]

        # Embed all chunks
        embeddings = embed_texts(chunks)

        # Metadata for each chunk
        metadatas = [
            {
                "source_document": file_path.name,
                "category": category,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        # Upsert into ChromaDB (idempotent)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        return len(chunks)

    # ── Retrieval ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        dense_weight: Optional[float] = None,
    ) -> list[Evidence]:
        """
        Hybrid retrieval: dense (ChromaDB) + sparse (BM25) with RRF fusion.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            dense_weight: Weight for dense vs sparse (0–1).

        Returns:
            List of Evidence objects ranked by fused relevance.
        """
        k = top_k or config.RETRIEVAL_TOP_K
        w = dense_weight if dense_weight is not None else config.DENSE_WEIGHT

        # Dense retrieval from ChromaDB
        dense_results = self._dense_retrieve(query, k * 2)

        # Sparse retrieval from BM25
        sparse_results = self._sparse_retrieve(query, k * 2)

        # Fuse with Reciprocal Rank Fusion
        fused = self._rrf_fuse(dense_results, sparse_results, w, k)

        return fused

    def _dense_retrieve(
        self, query: str, n_results: int
    ) -> list[tuple[str, str, dict, float]]:
        """
        Dense vector search via ChromaDB.

        Returns:
            List of (chunk_id, text, metadata, score) tuples.
        """
        if self._collection.count() == 0:
            return []

        query_embedding = embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        items = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"][0]):
                text = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                # ChromaDB returns distance; convert to similarity score
                distance = results["distances"][0][i] if results["distances"] else 1.0
                score = max(0.0, 1.0 - distance)
                items.append((chunk_id, text, meta, score))

        return items

    def _sparse_retrieve(
        self, query: str, n_results: int
    ) -> list[tuple[str, str, dict, float]]:
        """
        Sparse BM25 retrieval.

        Returns:
            List of (chunk_id, text, metadata, score) tuples.
        """
        if self._bm25_index is None:
            self._rebuild_bm25_index()

        if not self._bm25_corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokenized_query)

        # Get top-N indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:n_results]

        items = []
        for idx in top_indices:
            if scores[idx] > 0:
                items.append((
                    self._bm25_ids[idx],
                    self._bm25_corpus[idx],
                    self._bm25_metadata[idx],
                    float(scores[idx]),
                ))

        return items

    def _rrf_fuse(
        self,
        dense_results: list[tuple[str, str, dict, float]],
        sparse_results: list[tuple[str, str, dict, float]],
        dense_weight: float,
        top_k: int,
    ) -> list[Evidence]:
        """
        Reciprocal Rank Fusion (RRF) to merge dense and sparse rankings.

        RRF score = Σ 1 / (k + rank) across rankings.
        We weight the contributions by dense_weight.
        k=60 is the standard constant from the RRF paper.
        """
        rrf_k = 60  # Standard RRF constant
        scores: dict[str, float] = {}
        texts: dict[str, str] = {}
        metas: dict[str, dict] = {}

        # Score dense results
        for rank, (chunk_id, text, meta, _) in enumerate(dense_results):
            rrf_score = dense_weight / (rrf_k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            texts[chunk_id] = text
            metas[chunk_id] = meta

        # Score sparse results
        sparse_weight = 1.0 - dense_weight
        for rank, (chunk_id, text, meta, _) in enumerate(sparse_results):
            rrf_score = sparse_weight / (rrf_k + rank + 1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + rrf_score
            texts.setdefault(chunk_id, text)
            metas.setdefault(chunk_id, meta)

        # Sort by fused score and return top_k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        evidences = []
        for chunk_id, fused_score in ranked:
            meta = metas.get(chunk_id, {})
            evidences.append(
                Evidence(
                    content=texts[chunk_id],
                    source=EvidenceSource.MEMORY,
                    source_document=meta.get("source_document", "unknown"),
                    chunk_id=chunk_id,
                    relevance_score=round(fused_score, 4),
                    metadata=meta,
                )
            )

        return evidences

    # ── BM25 Index Management ───────────────────────────────────────────

    def _rebuild_bm25_index(self) -> None:
        """Rebuild the BM25 index from all documents in ChromaDB."""
        all_data = self._collection.get(include=["documents", "metadatas"])

        if not all_data or not all_data["ids"]:
            self._bm25_index = None
            return

        self._bm25_ids = all_data["ids"]
        self._bm25_corpus = all_data["documents"] or []
        self._bm25_metadata = all_data["metadatas"] or []

        # Tokenize corpus for BM25
        tokenized_corpus = [self._tokenize(doc) for doc in self._bm25_corpus]

        if tokenized_corpus:
            self._bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self._bm25_index = None

    # ── Utilities ───────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str) -> list[str]:
        """
        Split text into chunks of ~500 chars with 100-char overlap.
        Splits on paragraph boundaries when possible.
        """
        if len(text) <= _CHUNK_SIZE:
            return [text.strip()] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + _CHUNK_SIZE

            if end < len(text):
                # Try to break at a paragraph or sentence boundary
                for sep in ["\n\n", "\n", ". ", ", "]:
                    break_pos = text.rfind(sep, start + _CHUNK_SIZE // 2, end)
                    if break_pos > start:
                        end = break_pos + len(sep)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = max(start + 1, end - _CHUNK_OVERLAP)

        return chunks

    @staticmethod
    def _chunk_id(filename: str, index: int) -> str:
        """Generate a stable, deterministic ID for a chunk."""
        raw = f"{filename}::chunk_{index}"
        return hashlib.md5(raw.encode()).hexdigest()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer for BM25."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    # ── Diagnostics ─────────────────────────────────────────────────────

    @property
    def chunk_count(self) -> int:
        """Number of chunks currently in the store."""
        return self._collection.count()

    def clear(self) -> None:
        """Remove all data from the store (useful in tests)."""
        self._chroma_client.delete_collection(self._collection_name)
        self._collection = self._chroma_client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._bm25_corpus = []
        self._bm25_ids = []
        self._bm25_metadata = []
        self._bm25_index = None
