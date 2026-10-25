"""Tests for document parser — PDF, DOCX, and text parsing."""

import pytest
from pathlib import Path

from feasiai.ingestion.parser import parse_document, parse_text


class TestParseText:
    def test_basic_cleaning(self):
        text = "Hello\r\nWorld\r\n"
        result = parse_text(text)
        assert result == "Hello\nWorld"

    def test_null_byte_removal(self):
        text = "Hello\x00World"
        result = parse_text(text)
        assert result == "HelloWorld"

    def test_preserves_tabs_and_newlines(self):
        text = "Line 1\n\tIndented\nLine 3"
        result = parse_text(text)
        assert "\t" in result
        assert "\n" in result

    def test_empty_string(self):
        assert parse_text("") == ""
        assert parse_text("   ") == ""


class TestParseDocument:
    def test_txt_file(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Test content\nLine 2", encoding="utf-8")
        result = parse_document(str(txt_file))
        assert "Test content" in result
        assert "Line 2" in result

    def test_md_file(self, tmp_path):
        md_file = tmp_path / "test.md"
        md_file.write_text("# Heading\n\nParagraph text", encoding="utf-8")
        result = parse_document(str(md_file))
        assert "Heading" in result
        assert "Paragraph text" in result

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_document("/nonexistent/file.txt")

    def test_unsupported_format(self, tmp_path):
        bad_file = tmp_path / "test.xyz"
        bad_file.write_text("content")
        with pytest.raises(ValueError, match="Unsupported file format"):
            parse_document(str(bad_file))

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        result = parse_document(str(empty))
        assert result == ""

    def test_actual_sample_rfp(self):
        """Test with the actual sample RFP in the project."""
        rfp_path = Path(__file__).parent.parent / "data" / "sample_rfps" / "sample_rfp_01.md"
        if rfp_path.exists():
            result = parse_document(str(rfp_path))
            assert len(result) > 100
            assert "NovaTech" in result
            assert "SAML 2.0" in result
