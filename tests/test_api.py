"""
Unit tests for llm_client.py dan prompt_builder.py (M5)
Run with: pytest tests/test_api.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.prompt_builder import build_prompt, build_citation_list
from backend.core.llm_client import generate

SAMPLE_CHUNKS = [
    {
        "text": "Kebijakan cuti tahunan karyawan adalah 12 hari kerja per tahun.",
        "source": "sop_hr.pdf",
        "page_num": 3,
        "score": 0.91,
    },
    {
        "text": "Pengajuan cuti harus dilakukan minimal 3 hari sebelumnya.",
        "source": "sop_hr.pdf",
        "page_num": 4,
        "score": 0.78,
    },
]


# ---------------------------------------------------------------------------
# Prompt Builder tests
# ---------------------------------------------------------------------------

def test_build_prompt_returns_string():
    prompt = build_prompt("Berapa hari cuti?", SAMPLE_CHUNKS)
    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_build_prompt_contains_query():
    query = "Berapa hari cuti tahunan?"
    prompt = build_prompt(query, SAMPLE_CHUNKS)
    assert query in prompt


def test_build_prompt_contains_chunk_text():
    prompt = build_prompt("test query", SAMPLE_CHUNKS)
    assert "12 hari kerja" in prompt


def test_build_prompt_contains_source():
    prompt = build_prompt("test query", SAMPLE_CHUNKS)
    assert "sop_hr.pdf" in prompt


def test_build_prompt_contains_page_number():
    prompt = build_prompt("test query", SAMPLE_CHUNKS)
    assert "Halaman 3" in prompt


def test_build_prompt_empty_chunks():
    prompt = build_prompt("Pertanyaan tanpa konteks.", [])
    assert isinstance(prompt, str)
    assert "Pertanyaan tanpa konteks." in prompt


def test_build_citation_list_returns_list():
    citations = build_citation_list(SAMPLE_CHUNKS)
    assert isinstance(citations, list)
    assert len(citations) == 2


def test_build_citation_list_has_expected_keys():
    citations = build_citation_list(SAMPLE_CHUNKS)
    for c in citations:
        assert "source" in c
        assert "page_num" in c
        assert "score" in c


def test_build_citation_list_score_rounded():
    citations = build_citation_list(SAMPLE_CHUNKS)
    for c in citations:
        assert isinstance(c["score"], float)


# ---------------------------------------------------------------------------
# LLM Client tests (tidak memanggil LLM sungguhan — cukup test struktur output)
# ---------------------------------------------------------------------------

def test_generate_returns_dict():
    result = generate("Halo, siapa kamu? Jawab satu kalimat.")
    assert isinstance(result, dict)


def test_generate_has_expected_keys():
    result = generate("Halo, siapa kamu? Jawab satu kalimat.")
    assert "text" in result
    assert "model" in result
    assert "success" in result
    assert "error" in result


def test_generate_success_flag():
    result = generate("Halo, siapa kamu? Jawab satu kalimat.")
    assert result["success"] is True
    assert result["text"] != ""
    assert result["error"] is None


def test_generate_invalid_model_returns_error():
    result = generate("test", model="model-yang-tidak-ada-sama-sekali")
    assert result["success"] is False
    assert result["error"] is not None
