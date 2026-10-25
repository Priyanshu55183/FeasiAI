"""
FeasiAI configuration module.

All configuration is centralized here and loaded from environment variables
with sensible defaults. This avoids scattering os.getenv() across modules
and makes testing easier (override via monkeypatch or .env).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ─── LLM Provider ───────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# Temperature for classification (low = deterministic)
LLM_TEMPERATURE_CLASSIFY: float = float(os.getenv("LLM_TEMPERATURE_CLASSIFY", "0.1"))
# Temperature for extraction (slightly higher to handle diverse formats)
LLM_TEMPERATURE_EXTRACT: float = float(os.getenv("LLM_TEMPERATURE_EXTRACT", "0.2"))

# ─── Paths ───────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR", str(_PROJECT_ROOT / "chroma_db")
)
MEMORY_DOCS_DIR: str = os.getenv(
    "MEMORY_DOCS_DIR", str(_PROJECT_ROOT / "data" / "memory")
)

# ─── Embedding model ────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ─── Retrieval tuning ───────────────────────────────────────────────────────
RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", "0.5"))

# ─── LLM call budget ────────────────────────────────────────────────────────
MAX_LLM_CALLS_PER_RUN: int = int(os.getenv("MAX_LLM_CALLS_PER_RUN", "50"))

# ─── ChromaDB collection name ───────────────────────────────────────────────
MEMORY_COLLECTION_NAME: str = "institutional_memory"
