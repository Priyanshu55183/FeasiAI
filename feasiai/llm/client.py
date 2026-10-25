"""
Unified LLM client wrapper.

Design decisions:
1. EVERY LLM call flows through this wrapper — no direct Groq calls
   elsewhere. This guarantees consistent logging, budget enforcement,
   and makes swapping providers trivial (just change this file).

2. The wrapper logs: prompt, response, latency, token counts, and
   purpose tag. This is the foundation for the evaluation harness
   (latency, cost, call-count metrics).

3. A global call counter enforces the per-run budget. If exceeded,
   the pipeline fails loudly rather than silently racking up cost.

4. We use the chat completions API with JSON mode where available.
   For structured output we rely on prompting + Pydantic parsing
   rather than function-calling, because Groq's Llama models have
   inconsistent function-calling support.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

from groq import Groq

from feasiai import config
from feasiai.schemas import LLMCallRecord

logger = logging.getLogger(__name__)


class LLMBudgetExceeded(Exception):
    """Raised when the per-run LLM call budget is exhausted."""


class LLMClient:
    """
    Thread-unsafe (by design for single-pipeline runs) LLM client.
    Tracks call count and stores call records for post-run analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_calls: Optional[int] = None,
    ):
        self._api_key = api_key or config.GROQ_API_KEY
        self._model = model or config.LLM_MODEL
        self._max_calls = max_calls or config.MAX_LLM_CALLS_PER_RUN
        self._call_count = 0
        self._call_log: list[LLMCallRecord] = []

        if not self._api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env "
                "and add your key from https://console.groq.com"
            )

        self._client = Groq(api_key=self._api_key)

    # ── Public API ──────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        *,
        purpose: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        response_format: Optional[dict] = None,
    ) -> str:
        """
        Send a chat completion request and return the assistant's text.

        Args:
            messages: OpenAI-style message list.
            purpose: Tag for logging (e.g. 'extraction', 'classification').
            temperature: Sampling temperature.
            max_tokens: Max response tokens.
            response_format: Optional {"type": "json_object"} for JSON mode.

        Returns:
            The assistant's response text.

        Raises:
            LLMBudgetExceeded: If call budget is exhausted.
        """
        if self._call_count >= self._max_calls:
            raise LLMBudgetExceeded(
                f"LLM call budget exhausted ({self._max_calls} calls). "
                "Increase MAX_LLM_CALLS_PER_RUN or optimize the pipeline."
            )

        self._call_count += 1
        start = time.perf_counter()
        error_msg = None
        raw_response = ""
        usage = {}

        try:
            kwargs = {
                "model": self._model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if response_format:
                kwargs["response_format"] = response_format

            completion = self._client.chat.completions.create(**kwargs)
            raw_response = completion.choices[0].message.content or ""
            usage = {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0,
            }
        except Exception as e:
            error_msg = str(e)
            logger.error("LLM call failed [%s]: %s", purpose, error_msg)
            raise
        finally:
            latency = time.perf_counter() - start
            record = LLMCallRecord(
                model=self._model,
                purpose=purpose,
                prompt_messages=messages,
                raw_response=raw_response,
                latency_seconds=round(latency, 3),
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                error=error_msg,
            )
            self._call_log.append(record)
            logger.info(
                "LLM call #%d [%s] | %.2fs | %d tokens | model=%s",
                self._call_count,
                purpose,
                latency,
                usage.get("total_tokens", 0),
                self._model,
            )

        return raw_response

    def chat_json(
        self,
        messages: list[dict],
        *,
        purpose: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> dict:
        """
        Send a chat request expecting JSON output.
        Parses the response and returns a dict.
        Falls back to extracting JSON from markdown code fences if needed.
        """
        raw = self.chat(
            messages,
            purpose=purpose,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return self._parse_json_response(raw)

    # ── Accessors ───────────────────────────────────────────────────────

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def call_log(self) -> list[LLMCallRecord]:
        return list(self._call_log)

    def reset_budget(self) -> None:
        """Reset call counter (e.g. between pipeline runs in tests)."""
        self._call_count = 0

    # ── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_response(raw: str) -> dict:
        """
        Parse JSON from LLM response. Handles:
        1. Clean JSON
        2. JSON wrapped in ```json ... ``` code fences
        3. JSON embedded in surrounding text
        """
        text = raw.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Strip code fences
        if "```" in text:
            # Find content between first ``` and last ```
            parts = text.split("```")
            for part in parts[1:]:
                # Remove optional language tag (e.g. "json\n")
                clean = part.strip()
                if clean.lower().startswith("json"):
                    clean = clean[4:].strip()
                try:
                    return json.loads(clean)
                except json.JSONDecodeError:
                    continue

        # Last resort: find first { ... last }
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace != -1:
            try:
                return json.loads(text[first_brace : last_brace + 1])
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse JSON from LLM response:\n{raw[:500]}")
