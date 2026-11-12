"""
Memory write-back — closed-loop updates from delivery outcomes.

After a proposal is delivered (won/lost/done), engineers can record
the actual outcome. This module ingests those outcomes back into the
institutional memory store so the next RFP benefits from real data.

Design decisions:
1. MARKDOWN DOCUMENTS: Write-back generates new .md files in data/memory/
   rather than directly modifying ChromaDB. This keeps the write path
   human-readable, version-controllable, and auditable.
2. THREE SIGNAL TYPES:
   - 'lesson' — a generalizable finding from this delivery
   - 'do_not_claim' — a capability that turned out to be false/overstated
   - 'effort_calibration' — actual engineering cost for future estimates
3. RE-INGEST TRIGGER: After writing the file, the caller should call
   MemoryStore.ingest_directory() to pick up the new document.
4. IDEMPOTENT: Writing the same record twice produces the same file —
   safe to replay on restart.
"""

from feasiai.writeback.writer import MemoryWritebackService
from feasiai.writeback.schemas import WritebackRecord, WritebackSignalType

__all__ = ["MemoryWritebackService", "WritebackRecord", "WritebackSignalType"]

