"""
Shared pytest fixtures for FeasiAI tests.

Provides reusable fixtures for LLM client mocking, memory store
setup, and sample data generation.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_rfp_text():
    """A short, representative RFP text for testing."""
    return """
    # Request for Proposal: Test Platform

    ## Authentication Requirements

    1. The system must support Single Sign-On via SAML 2.0.
    2. Role-Based Access Control with at least 3 levels of hierarchy.
    3. The system should be user-friendly and intuitive.

    ## Integration Requirements

    4. REST API with OpenAPI 3.0 documentation.
    5. Native Shopify integration for e-commerce order management.
    """


@pytest.fixture
def sample_memory_dir(tmp_path):
    """Create a temporary directory with sample memory documents."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    (memory_dir / "product_capabilities.md").write_text(
        """# Product Capabilities
## Authentication
- SSO via SAML 2.0 and OIDC supported.
- RBAC with up to 5 levels of role nesting.
- MFA via TOTP and WebAuthn/FIDO2.

## Integration
- REST API with OpenAPI 3.0 spec covering 95% of features.
- Native Salesforce integration (bi-directional).
- Webhook support with retry logic.
""",
        encoding="utf-8",
    )

    (memory_dir / "known_gaps.md").write_text(
        """# Known Gaps
## E-Commerce
- No Shopify, WooCommerce, or Magento integration.
- No payment processing capabilities.
- No product catalog management.
""",
        encoding="utf-8",
    )

    return memory_dir


@pytest.fixture
def mock_llm_client():
    """
    A mock LLM client that returns predefined responses.
    Does not make actual API calls.
    """
    client = MagicMock()
    client.call_count = 0
    client.call_log = []
    client.reset_budget = MagicMock()

    # Default extraction response
    client.chat_json.return_value = {
        "requirements": [
            {
                "text": "The system must support SSO via SAML 2.0",
                "original_section": "Authentication",
                "is_ambiguous": False,
                "ambiguity_reason": None,
            },
            {
                "text": "The system should be user-friendly",
                "original_section": "General",
                "is_ambiguous": True,
                "ambiguity_reason": "No measurable criteria for user-friendliness",
            },
        ]
    }

    return client


@pytest.fixture
def mock_classification_response():
    """A mock classification JSON response."""
    return {
        "label": "Available",
        "confidence": 0.9,
        "reasoning": "SSO via SAML 2.0 is confirmed in product capabilities.",
        "what_would_change": "If custom SAML attribute mapping is required.",
        "draft_response": "Our platform fully supports SSO via SAML 2.0.",
    }
