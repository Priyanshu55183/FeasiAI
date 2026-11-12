"""
Jira connector for live issue/ticket data.

Design decisions:
1. REST API v3: Uses Jira Cloud REST API v3 with Basic Auth (email + API token).
   This avoids requiring the official jira library as a dependency.
2. JQL SEARCH: Converts free-text queries to JQL text search across summary,
   description, and comments fields for maximum coverage.
3. PROJECT SCOPING: JIRA_PROJECT_KEYS limits search to configured projects,
   preventing leakage of unrelated team tickets.
4. GRACEFUL NO-CREDENTIALS: Returns [] immediately if any required env var is
   missing — never fails the pipeline due to Jira being unconfigured.
5. DESCRIPTION TRUNCATION: Descriptions are capped at 300 chars to keep
   Evidence content concise and within LLM context budgets.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

from feasiai import config
from feasiai.live.base import LiveDataConnector
from feasiai.schemas import Evidence, EvidenceSource

logger = logging.getLogger(__name__)

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


class JiraConnector(LiveDataConnector):
    """
    Fetches Jira issues matching a text query via the Jira Cloud REST API v3.

    Configuration (via environment variables or config.py):
        JIRA_BASE_URL      e.g. https://yourcompany.atlassian.net
        JIRA_EMAIL         Atlassian account email
        JIRA_API_TOKEN     API token from https://id.atlassian.com/manage-profile/security/api-tokens
        JIRA_PROJECT_KEYS  Comma-separated project keys, e.g. "ENG,INFRA" (optional)
    """

    _MAX_DESCRIPTION_CHARS = 300

    def __init__(
        self,
        base_url: Optional[str] = None,
        email: Optional[str] = None,
        api_token: Optional[str] = None,
        project_keys: Optional[str] = None,
        requests_session: Optional[object] = None,
    ):
        """
        Args:
            base_url: Jira base URL (defaults to JIRA_BASE_URL env var).
            email: Atlassian account email (defaults to JIRA_EMAIL env var).
            api_token: API token (defaults to JIRA_API_TOKEN env var).
            project_keys: Comma-separated project keys (defaults to JIRA_PROJECT_KEYS env var).
            requests_session: Optional pre-configured requests.Session for testing.
        """
        self._base_url = (base_url or config.JIRA_BASE_URL).rstrip("/")
        self._email = email or config.JIRA_EMAIL
        self._api_token = api_token or config.JIRA_API_TOKEN
        self._project_keys = project_keys or config.JIRA_PROJECT_KEYS
        self._session = requests_session  # injected for testing

    @property
    def name(self) -> str:
        return "jira"

    @property
    def _is_configured(self) -> bool:
        """True when all required credentials are present."""
        return bool(self._base_url and self._email and self._api_token)

    @property
    def _auth_header(self) -> str:
        """HTTP Basic Auth header value (email:token base64-encoded)."""
        credentials = f"{self._email}:{self._api_token}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    def _build_jql(self, query: str) -> str:
        """
        Construct a JQL query string from free-text query.

        Args:
            query: Free-text search query.

        Returns:
            JQL string scoped to configured projects (if any).
        """
        # Escape double quotes in the query string
        safe_query = query.replace('"', '\\"')
        jql = f'text ~ "{safe_query}" ORDER BY updated DESC'

        if self._project_keys:
            keys = [k.strip() for k in self._project_keys.split(",") if k.strip()]
            if keys:
                project_clause = " OR ".join(f'project = "{k}"' for k in keys)
                jql = f"({project_clause}) AND {jql}"

        return jql

    def _format_issue(self, issue: dict) -> str:
        """
        Format a Jira issue dict into a readable Evidence content string.

        Args:
            issue: Raw issue dict from the Jira REST API response.

        Returns:
            Formatted string with key, summary, status, and description preview.
        """
        key = issue.get("key", "UNKNOWN")
        fields = issue.get("fields", {})
        summary = fields.get("summary", "(no summary)")
        status = fields.get("status", {}).get("name", "Unknown")
        description_obj = fields.get("description", None)

        # Jira v3 returns description as Atlassian Document Format (ADF), or plain str
        if isinstance(description_obj, dict):
            # Try to extract text from ADF content blocks
            desc = _extract_adf_text(description_obj)
        elif isinstance(description_obj, str):
            desc = description_obj
        else:
            desc = ""

        desc_preview = desc[: self._MAX_DESCRIPTION_CHARS]
        if len(desc) > self._MAX_DESCRIPTION_CHARS:
            desc_preview += "..."

        lines = [f"[{key}] {summary}", f"Status: {status}"]
        if desc_preview:
            lines.append(f"Description: {desc_preview}")
        return "\n".join(lines)

    def retrieve(self, query: str, top_k: int = 5) -> list[Evidence]:
        """
        Search Jira for issues relevant to query.

        Args:
            query: Free-text query string.
            top_k: Maximum issues to return.

        Returns:
            List of Evidence with source=LIVE_DATA, or [] if unconfigured/unavailable.
        """
        if not self._is_configured:
            logger.debug("JiraConnector: not configured — skipping (set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN)")
            return []

        if not _REQUESTS_AVAILABLE and self._session is None:
            logger.warning("JiraConnector: 'requests' library not available and no session injected")
            return []

        try:
            return self._search(query, top_k)
        except Exception as exc:
            logger.warning("JiraConnector: search failed — %s", exc)
            return []

    def _search(self, query: str, top_k: int) -> list[Evidence]:
        """Internal search — may raise; callers should catch."""
        jql = self._build_jql(query)
        url = f"{self._base_url}/rest/api/3/issue/search"
        headers = {
            "Authorization": self._auth_header,
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "jql": jql,
            "maxResults": top_k,
            "fields": ["summary", "status", "description", "key"],
        }

        session = self._session
        if session is None:
            import requests
            response = requests.post(url, json=payload, headers=headers, timeout=10)
        else:
            response = session.post(url, json=payload, headers=headers, timeout=10)

        response.raise_for_status()
        data = response.json()

        evidences = []
        for issue in data.get("issues", []):
            key = issue.get("key", "unknown")
            content = self._format_issue(issue)
            ev = Evidence(
                content=content,
                source=EvidenceSource.LIVE_DATA,
                source_document=f"jira:{key}",
                chunk_id=f"jira:{key}",
                relevance_score=1.0,
                metadata={"connector": "jira", "issue_key": key},
            )
            evidences.append(ev)

        return evidences

    def health_check(self) -> bool:
        """
        Verify Jira credentials by calling the myself endpoint.

        Returns:
            True if credentials are valid and Jira is reachable.
        """
        if not self._is_configured:
            return False
        try:
            url = f"{self._base_url}/rest/api/3/myself"
            headers = {
                "Authorization": self._auth_header,
                "Accept": "application/json",
            }
            session = self._session
            if session is None:
                import requests
                resp = requests.get(url, headers=headers, timeout=5)
            else:
                resp = session.get(url, headers=headers, timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


def _extract_adf_text(adf: dict) -> str:
    """
    Recursively extract plain text from an Atlassian Document Format (ADF) node.

    Args:
        adf: ADF content node dict.

    Returns:
        Concatenated plain text string.
    """
    if not isinstance(adf, dict):
        return ""

    text_parts: list[str] = []
    if adf.get("type") == "text":
        text_parts.append(adf.get("text", ""))

    for child in adf.get("content", []):
        text_parts.append(_extract_adf_text(child))

    return " ".join(t for t in text_parts if t).strip()

