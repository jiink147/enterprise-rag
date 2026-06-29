"""
Unit tests for backend/core/document_loader.py

Run with:
    pytest tests/test_loader.py -v
"""

import sys
import os

# Allow imports from the project root when running pytest from anywhere
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.document_loader import parse_pdf

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents", "samples")


def sample_path(filename: str) -> str:
    return os.path.join(SAMPLES_DIR, filename)


def test_parse_pdf_returns_list():
    """parse_pdf should always return a list, even on success."""
    result = parse_pdf(sample_path("Fachrurrozi_Rosyadi_Resume.pdf"))
    assert isinstance(result, list)


def test_parse_pdf_has_expected_keys():
    """Each page dict must contain 'page_num' and 'text' keys."""
    result = parse_pdf(sample_path("Fachrurrozi_Rosyadi_Resume.pdf"))
    assert len(result) > 0
    for page in result:
        assert "page_num" in page
        assert "text" in page


def test_parse_pdf_page_numbers_start_at_one():
    """Page numbering should be 1-indexed (human-friendly), not 0-indexed."""
    result = parse_pdf(sample_path("Fachrurrozi_Rosyadi_Resume.pdf"))
    assert result[0]["page_num"] == 1


def test_parse_pdf_page_numbers_are_sequential():
    """Page numbers should increase monotonically by 1."""
    result = parse_pdf(sample_path("s41598-025-18947-2.pdf"))
    page_nums = [p["page_num"] for p in result]
    assert page_nums == list(range(1, len(page_nums) + 1))


def test_parse_pdf_text_based_document_has_content():
    """A normal text-based PDF (resume) should yield non-empty text on at least one page."""
    result = parse_pdf(sample_path("Fachrurrozi_Rosyadi_Resume.pdf"))
    combined_text = " ".join(p["text"] for p in result)
    assert len(combined_text.strip()) > 0


def test_parse_pdf_academic_paper_multi_page():
    """Academic paper should have multiple pages."""
    result = parse_pdf(sample_path("s41598-025-18947-2.pdf"))
    assert len(result) > 1


def test_parse_pdf_scanned_id_card_likely_empty_text():
    """
    KTP.pdf is expected to be an image-based scan with no embedded text layer.
    PyMuPDF should not crash, but extracted text per page may be empty.
    This documents the known limitation that OCR (planned for M8) is needed for scans.
    """
    result = parse_pdf(sample_path("KTP.pdf"))
    assert isinstance(result, list)
    # We don't assert text is empty (some scans embed a hidden text layer),
    # we just assert the function handled it without raising an exception.


def test_parse_pdf_invalid_path_returns_empty_list():
    """A non-existent file path should not crash; it should return an empty list."""
    result = parse_pdf(sample_path("this_file_does_not_exist.pdf"))
    assert result == []


def test_parse_pdf_financial_document_payslip():
    """Payslip should parse without error and contain some content."""
    result = parse_pdf(sample_path("payslips_2025.pdf"))
    assert isinstance(result, list)
    assert len(result) > 0


def test_parse_pdf_climate_report():
    """laporan_iklim_harian.pdf should parse without error."""
    result = parse_pdf(sample_path("laporan_iklim_harian.pdf"))
    assert isinstance(result, list)
    assert len(result) > 0
