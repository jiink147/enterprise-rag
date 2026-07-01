"""
Unit tests for retriever.py and confidence.py (M6)
Run with: pytest tests/test_retriever.py -v
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.confidence import compute_confidence, format_answer_with_confidence
from backend.core.retriever import retrieve
from backend.core.faiss_store import FaissStore
from backend.core.embedder import embed_texts, embed_query


# ---------------------------------------------------------------------------
# Confidence Scoring tests
# ---------------------------------------------------------------------------

def test_confidence_high():
    chunks = [{"text": "cuti 12 hari", "score": 0.85, "rerank_score": 8.5}]
    result = compute_confidence(chunks)
    assert result["level"] == "HIGH"
    assert result["should_answer"] is True
    assert result["disclaimer"] is None


def test_confidence_medium():
    chunks = [{"text": "cuti tahunan", "score": 0.55, "rerank_score": 4.5}]
    result = compute_confidence(chunks)
    assert result["level"] == "MEDIUM"
    assert result["should_answer"] is True
    assert result["disclaimer"] is not None


def test_confidence_low():
    chunks = [{"text": "resep masakan", "score": 0.15, "rerank_score": 1.2}]
    result = compute_confidence(chunks)
    assert result["level"] == "LOW"
    assert result["should_answer"] is False
    assert result["disclaimer"] is not None


def test_confidence_empty_chunks():
    result = compute_confidence([])
    assert result["level"] == "LOW"
    assert result["should_answer"] is False
    assert result["score"] == 0.0


def test_confidence_without_reranker():
    chunks = [{"text": "teks relevan", "score": 0.75}]
    result = compute_confidence(chunks, use_reranker=False)
    assert result["level"] == "HIGH"
    assert result["should_answer"] is True


def test_format_answer_high_confidence():
    confidence = {"level": "HIGH", "score": 0.85, "should_answer": True, "disclaimer": None}
    citations = [{"source": "doc.pdf", "page_num": 1, "score": 0.85}]
    result = format_answer_with_confidence("Jawaban lengkap.", confidence, citations)
    assert result["answer"] == "Jawaban lengkap."
    assert result["confidence"] == "HIGH"
    assert len(result["citations"]) == 1


def test_format_answer_low_confidence_hides_citations():
    confidence = {
        "level": "LOW",
        "score": 0.15,
        "should_answer": False,
        "disclaimer": "Tidak ditemukan informasi relevan.",
    }
    citations = [{"source": "doc.pdf", "page_num": 1, "score": 0.15}]
    result = format_answer_with_confidence("Jawaban.", confidence, citations)
    assert result["citations"] == []
    assert "Tidak ditemukan" in result["answer"]


# ---------------------------------------------------------------------------
# Retriever tests
# ---------------------------------------------------------------------------

def _make_store():
    texts = [
        "Kebijakan cuti tahunan karyawan adalah 12 hari kerja per tahun.",
        "Gaji karyawan dibayarkan setiap tanggal 25 bulan berjalan.",
        "Resep nasi goreng membutuhkan bawang putih dan kecap manis.",
        "Prosedur pengajuan cuti harus dilakukan 3 hari sebelumnya.",
        "Tunjangan kesehatan mencakup rawat inap dan rawat jalan.",
    ]
    metadata = [{"text": t, "source": "sop_hr.pdf", "page_num": i + 1} for i, t in enumerate(texts)]
    vectors = embed_texts(texts)
    store = FaissStore(dim=384)
    store.add(vectors, metadata)
    return store


def test_retrieve_returns_list():
    store = _make_store()
    results = retrieve("Berapa hari cuti karyawan?", store, use_reranker=False)
    assert isinstance(results, list)
    assert len(results) > 0


def test_retrieve_top_k_respected():
    store = _make_store()
    results = retrieve("cuti karyawan", store, top_k_final=2, use_reranker=False)
    assert len(results) <= 2


def test_retrieve_empty_store_returns_empty():
    store = FaissStore(dim=384)
    results = retrieve("test query", store)
    assert results == []


def test_retrieve_result_has_text_and_score():
    store = _make_store()
    results = retrieve("kebijakan gaji", store, use_reranker=False)
    for r in results:
        assert "text" in r
        assert "score" in r


def test_retrieve_with_reranker_returns_rerank_score():
    store = _make_store()
    results = retrieve(
        "Berapa hari cuti karyawan?",
        store,
        top_k_faiss=5,
        top_k_final=2,
        use_reranker=True,
    )
    assert len(results) > 0
    for r in results:
        assert "rerank_score" in r


def test_retrieve_reranker_improves_relevance():
    store = _make_store()
    query = "Berapa hari cuti tahunan?"

    results_no_rerank = retrieve(query, store, top_k_final=1, use_reranker=False)
    results_reranked = retrieve(query, store, top_k_final=1, use_reranker=True)

    assert len(results_reranked) > 0
    assert "cuti" in results_reranked[0]["text"].lower()
