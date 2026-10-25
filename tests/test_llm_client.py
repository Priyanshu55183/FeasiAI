"""Tests for LLM client wrapper."""

import pytest
import json
from unittest.mock import MagicMock, patch

from feasiai.llm.client import LLMClient, LLMBudgetExceeded


class TestLLMClientInit:
    def test_missing_api_key_raises(self):
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            LLMClient(api_key="")

    def test_custom_budget(self):
        with patch("feasiai.llm.client.Groq"):
            client = LLMClient(api_key="test-key", max_calls=5)
            assert client._max_calls == 5


class TestJsonParsing:
    """Test the JSON parsing logic (no API calls needed)."""

    def test_clean_json(self):
        raw = '{"key": "value", "number": 42}'
        result = LLMClient._parse_json_response(raw)
        assert result == {"key": "value", "number": 42}

    def test_json_in_code_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        result = LLMClient._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_in_plain_code_fence(self):
        raw = '```\n{"key": "value"}\n```'
        result = LLMClient._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_json_with_surrounding_text(self):
        raw = 'Here is the result:\n{"key": "value"}\nEnd of result.'
        result = LLMClient._parse_json_response(raw)
        assert result == {"key": "value"}

    def test_unparseable_raises(self):
        with pytest.raises(ValueError, match="Could not parse JSON"):
            LLMClient._parse_json_response("This is not JSON at all.")

    def test_nested_json(self):
        raw = '{"outer": {"inner": [1, 2, 3]}, "flag": true}'
        result = LLMClient._parse_json_response(raw)
        assert result["outer"]["inner"] == [1, 2, 3]


class TestBudgetEnforcement:
    def test_budget_exceeded(self):
        with patch("feasiai.llm.client.Groq"):
            client = LLMClient(api_key="test-key", max_calls=2)
            # Simulate exhausting the budget
            client._call_count = 2
            with pytest.raises(LLMBudgetExceeded):
                client.chat([{"role": "user", "content": "test"}])

    def test_reset_budget(self):
        with patch("feasiai.llm.client.Groq"):
            client = LLMClient(api_key="test-key", max_calls=5)
            client._call_count = 5
            client.reset_budget()
            assert client.call_count == 0
