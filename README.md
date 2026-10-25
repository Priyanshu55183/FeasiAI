# FeasiAI — AI Solution Feasibility & Proposal Copilot

An agentic RAG system that reads an RFP and assesses, per requirement, whether your company can deliver it. Designed as a high-signal engineering project demonstrating systems thinking, measurement discipline, and uncertainty-aware AI.

## What It Does

Given an RFP document (PDF/DOCX/text), the system:

1. **Extracts** atomic, evaluable requirements (structured JSON; flags ambiguous ones)
2. **Retrieves** evidence from institutional memory (product capabilities, past proposals, known gaps) using hybrid dense + BM25 search
3. **Classifies** each requirement as: `Available` | `Configurable` | `Requires Development` | `Unsupported / Unclear`
4. **Generates** a structured feasibility report with confidence scores, evidence provenance, and draft client-facing responses

## Quick Start

### 1. Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free tier)

### 2. Setup

```bash
# Clone the repository
cd FeasiAI

# Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Configure API key
copy .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 3. Run the Pipeline

```bash
# Run with the included sample RFP
python run_pipeline.py

# Run with your own document
python run_pipeline.py path/to/your/rfp.pdf

# Run with inline text
python run_pipeline.py --text "The system must support SSO via SAML 2.0"

# Save JSON report
python run_pipeline.py --json-output report.json

# Verbose logging
python run_pipeline.py -v
```

### 4. Run Tests

```bash
# Run all tests (no API key needed — tests use mocks)
pytest tests/ -v

# Run specific test file
pytest tests/test_schemas.py -v

# Run with coverage
pytest tests/ -v --tb=short
```

## Project Structure

```
FeasiAI/
├── feasiai/                    # Core package
│   ├── config.py               # Central configuration (env vars)
│   ├── schemas.py              # Pydantic data models
│   ├── pipeline.py             # End-to-end orchestrator
│   ├── llm/
│   │   └── client.py           # Unified LLM wrapper (logging + budget)
│   ├── ingestion/
│   │   └── parser.py           # PDF/DOCX/text document parsing
│   ├── extraction/
│   │   └── requirement_extractor.py  # LLM-powered requirement extraction
│   ├── retrieval/
│   │   ├── embeddings.py       # Embedding model wrapper
│   │   └── memory_store.py     # Hybrid ChromaDB + BM25 retrieval
│   ├── classification/
│   │   └── classifier.py       # Feasibility classifier
│   └── report/
│       └── generator.py        # Report assembly + formatting
├── data/
│   ├── memory/                 # Institutional knowledge documents
│   │   ├── product_capabilities.md
│   │   ├── past_proposals.md
│   │   └── known_gaps.md
│   └── sample_rfps/
│       └── sample_rfp_01.md    # Sample RFP for testing
├── tests/                      # pytest test suite
├── run_pipeline.py             # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Architecture

```
RFP Document
    │
    ▼
┌──────────────────┐
│ Document Parser   │  PDF/DOCX/Text → clean text
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Requirement       │  LLM-powered extraction
│ Extractor         │  Flags ambiguous requirements
└────────┬─────────┘  Prompt injection defense
         │
         ▼
┌──────────────────┐   ┌─────────────────────┐
│ Hybrid Retrieval  │◄──│ Memory Store         │
│ (Dense + BM25     │   │ • Product capabilities│
│  + RRF Fusion)    │   │ • Past proposals      │
└────────┬─────────┘   │ • Known gaps          │
         │              └─────────────────────┘
         ▼
┌──────────────────┐
│ Feasibility       │  Single-shot classification
│ Classifier        │  Confidence + evidence-grounded
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Structured Report │  Summary + per-requirement details
│ + Human Review    │  ⚠️ Requires sign-off
└──────────────────┘
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Hybrid retrieval (dense + BM25 + RRF) | Dense catches paraphrases; BM25 catches exact terms. RRF merges without score normalization. |
| Prompt injection defense | RFP text is untrusted input. XML delimiters + explicit ignore instructions. |
| "Unsupported / Unclear" as first-class label | Admitting uncertainty is better than over-promising. |
| Unified LLM wrapper with call budget | Every call logged. Hard limit prevents runaway cost. |
| Evidence-only generation | Classifier can only cite retrieved evidence; no hallucination. |
| Frozen Pydantic models | Immutable data objects prevent accidental mutation across pipeline steps. |

## Current Status: Phase 1 (Foundation)

✅ Implemented:
- Document ingestion (PDF, DOCX, text)
- LLM-powered requirement extraction with ambiguity flagging
- Hybrid retrieval (ChromaDB dense + BM25 sparse + RRF fusion)
- Single-shot feasibility classification
- Structured reporting with provenance
- Full test suite (no API key needed)
- Prompt injection defenses

🔜 Coming in later phases:
- Planner Agent with tool-use budget & early stopping
- Live data layer (roadmap, tickets, feature flags)
- Conflict detection between memory and live data
- Evaluation harness with labeled dataset & ablations
- Streamlit review UI
- Closed-loop memory updates from delivery outcomes

## License

MIT
