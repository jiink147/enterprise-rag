"""
Unit tests for backend/core/document_loader.py (M3: multi-format) and backend/utils/hasher.py

Run with:
    pytest tests/test_loader.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.document_loader import parse_pdf, parse_docx, parse_xlsx, parse_txt, DocumentLoader
from backend.utils.hasher import compute_file_hash

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "documents", "samples")


def sample_path(filename):
    return os.path.join(SAMPLES_DIR, filename)


def test_parse_pdf_returns_list():
    result = parse_pdf(sample_path("Fachrurrozi_Rosyadi_Resume.pdf"))
    assert isinstance(result, list)
    assert len(result) > 0


def test_parse_docx_returns_list():
    result = parse_docx(sample_path("Fachrurrozi Rosyadi_Resume.docx"))
    assert isinstance(result, list)
    assert len(result) > 0


def test_parse_docx_has_expected_keys():
    result = parse_docx(sample_path("Fachrurrozi Rosyadi_Resume.docx"))
    for section in result:
        assert "section_title" in section
        assert "text" in section


def test_parse_docx_invalid_path_returns_empty_list():
    result = parse_docx(sample_path("does_not_exist.docx"))
    assert result == []


def test_parse_xlsx_returns_list():
    result = parse_xlsx(sample_path("Jadwal_Shift_Februari_2026_4_Minggu.xlsx"))
    assert isinstance(result, list)
    assert len(result) > 0


def test_parse_xlsx_has_expected_keys():
    result = parse_xlsx(sample_path("Jadwal_Shift_Februari_2026_4_Minggu.xlsx"))
    for sheet in result:
        assert "sheet_name" in sheet
        assert "text" in sheet


def test_parse_xlsx_invalid_path_returns_empty_list():
    result = parse_xlsx(sample_path("does_not_exist.xlsx"))
    assert result == []


def test_parse_txt_returns_list():
    result = parse_txt(sample_path("Image data.txt"))
    assert isinstance(result, list)
    assert len(result) == 1
    assert "text" in result[0]


def test_parse_txt_invalid_path_returns_empty_list():
    result = parse_txt(sample_path("does_not_exist.txt"))
    assert result == []


def test_document_loader_detects_pdf():
    loader = DocumentLoader()
    result = loader.load(sample_path("Fachrurrozi_Rosyadi_Resume.pdf"))
    assert result["file_type"] == "pdf"
    assert result["success"] is True


def test_document_loader_detects_docx():
    loader = DocumentLoader()
    result = loader.load(sample_path("Fachrurrozi Rosyadi_Resume.docx"))
    assert result["file_type"] == "docx"
    assert result["success"] is True


def test_document_loader_detects_xlsx():
    loader = DocumentLoader()
    result = loader.load(sample_path("Jadwal_Shift_Februari_2026_4_Minggu.xlsx"))
    assert result["file_type"] == "xlsx"
    assert result["success"] is True


def test_document_loader_detects_txt():
    loader = DocumentLoader()
    result = loader.load(sample_path("Image data.txt"))
    assert result["file_type"] == "txt"
    assert result["success"] is True


def test_document_loader_unsupported_extension():
    loader = DocumentLoader()
    result = loader.load(sample_path("something.zip"))
    assert result["success"] is False
    assert result["error"] is not None


def test_document_loader_missing_file():
    loader = DocumentLoader()
    result = loader.load(sample_path("ghost_file.pdf"))
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_compute_file_hash_returns_32_char_hex():
    file_hash = compute_file_hash(sample_path("Fachrurrozi Rosyadi_Resume.docx"))
    assert isinstance(file_hash, str)
    assert len(file_hash) == 32
    int(file_hash, 16)


def test_compute_file_hash_is_deterministic():
    hash1 = compute_file_hash(sample_path("Fachrurrozi Rosyadi_Resume.docx"))
    hash2 = compute_file_hash(sample_path("Fachrurrozi Rosyadi_Resume.docx"))
    assert hash1 == hash2


def test_compute_file_hash_differs_for_different_files():
    hash1 = compute_file_hash(sample_path("Fachrurrozi Rosyadi_Resume.docx"))
    hash2 = compute_file_hash(sample_path("Jadwal_Shift_Februari_2026_4_Minggu.xlsx"))
    assert hash1 != hash2
