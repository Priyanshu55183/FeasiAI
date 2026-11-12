"""
Feedback capture module — persists reviewer decisions for closed-loop learning.

Design decisions:
1. SQLITE BACKEND: Lightweight, file-based, no server required. Suitable
   for a solo/small-team review workflow. Can be replaced with Postgres
   by swapping the FeedbackStore implementation.
2. APPEND-ONLY LOG: Feedback records are never updated in-place — only
   new records are inserted. This provides a full audit trail of how
   reviewer opinions evolved over time.
3. SEPARATION FROM SCHEMAS: FeedbackRecord lives here, not in core schemas,
   because it describes human actions, not pipeline outputs.
"""

from feasiai.feedback.store import FeedbackStore
from feasiai.feedback.schemas import FeedbackRecord, ReviewDecision

__all__ = ["FeedbackStore", "FeedbackRecord", "ReviewDecision"]

