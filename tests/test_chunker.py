import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.chunker import chunk_text, chunk_document
from backend.core.embedder import embed_texts, embed_query
from backend.core.faiss_store import FaissStore


def test_chunk_text_basic():
    text = "Ini kalimat pertama. " * 30
    result = chunk_text(text, chunk_size=100, overlap=20)
    assert isinstance(result, list)
    assert len(result) > 1


def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_short_text_returns_one_chunk():
    text = "Teks ini cukup panjang untuk lolos filter min_chunk_size dan menghasilkan satu chunk saja."
    result = chunk_text(text, chunk_size=500)
    assert len(result) == 1


def test_chunk_text_min_chunk_size_filters_tiny_chunks():
    text = "Hi. " * 5
    result = chunk_text(text, chunk_size=500, min_chunk_size=100)
    for chunk in result:
        assert len(chunk) >= 100


def test_chunk_text_overlap_content():
    text = "Paragraf pertama berisi informasi penting tentang kebijakan. " * 5
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    if len(chunks) > 1:
        tail_of_first = chunks[0][-20:]
        assert tail_of_first.strip() in chunks[1]


def test_chunk_document_returns_list():
    parsed_doc = {
        "chunks_raw": [
            {"page_num": 1, "text": "Halaman pertama berisi informasi penting. " * 10},
            {"page_num": 2, "text": "Halaman kedua berisi data tambahan. " * 10},
        ]
    }
    result = chunk_document(parsed_doc, chunk_size=100, overlap=20)
    assert isinstance(result, list)
    assert len(result) > 0


def test_chunk_document_preserves_metadata():
    parsed_doc = {
        "chunks_raw": [
            {"page_num": 3, "text": "Teks halaman tiga. " * 20},
        ]
    }
    result = chunk_document(parsed_doc, chunk_size=100, overlap=20)
    for chunk in result:
        assert "chunk_index" in chunk
        assert "text" in chunk
        assert "page_num" in chunk
        assert chunk["page_num"] == 3


def test_chunk_index_is_sequential():
    parsed_doc = {
        "chunks_raw": [
            {"page_num": 1, "text": "Kalimat satu. " * 20},
            {"page_num": 2, "text": "Kalimat dua. " * 20},
        ]
    }
    result = chunk_document(parsed_doc, chunk_size=100, overlap=20)
    indices = [c["chunk_index"] for c in result]
    assert indices == list(range(len(indices)))


def test_embed_texts_shape():
    texts = ["Kebijakan cuti.", "Gaji karyawan.", "Resep masakan."]
    vectors = embed_texts(texts)
    assert vectors.shape == (3, 384)


def test_embed_texts_empty_returns_empty():
    result = embed_texts([])
    assert result.shape == (0, 384)


def test_embed_texts_normalized():
    texts = ["Teks untuk diuji normalisasinya."]
    vectors = embed_texts(texts)
    norm = np.linalg.norm(vectors[0])
    assert abs(norm - 1.0) < 1e-5


def test_embed_query_shape():
    vec = embed_query("Apa kebijakan cuti tahunan?")
    assert vec.shape == (384,)


def test_embed_semantic_similarity():
    texts = [
        "Kebijakan cuti tahunan karyawan adalah 12 hari.",
        "Annual leave policy is 12 days per year.",
        "Resep nasi goreng membutuhkan kecap manis.",
    ]
    vectors = embed_texts(texts)
    sim_related = float(np.dot(vectors[0], vectors[1]))
    sim_unrelated = float(np.dot(vectors[0], vectors[2]))
    assert sim_related > sim_unrelated


def test_faiss_store_add_and_search():
    texts = [
        "Kebijakan cuti tahunan karyawan adalah 12 hari.",
        "Gaji dibayarkan setiap tanggal 25.",
        "Resep nasi goreng membutuhkan kecap.",
    ]
    metadata = [{"text": t, "source": "test.pdf"} for t in texts]
    vectors = embed_texts(texts)

    store = FaissStore(dim=384)
    store.add(vectors, metadata)
    assert store.total_vectors == 3


def test_faiss_search_returns_relevant_result():
    texts = [
        "Kebijakan cuti tahunan karyawan adalah 12 hari.",
        "Gaji dibayarkan setiap tanggal 25.",
        "Resep nasi goreng membutuhkan kecap.",
    ]
    metadata = [{"text": t, "source": "test.pdf"} for t in texts]
    vectors = embed_texts(texts)

    store = FaissStore(dim=384)
    store.add(vectors, metadata)

    query_vec = embed_query("Berapa hari cuti karyawan?")
    results = store.search(query_vec, top_k=1)

    assert len(results) == 1
    assert "cuti" in results[0]["text"]


def test_faiss_search_empty_store_returns_empty():
    store = FaissStore(dim=384)
    query_vec = embed_query("test query")
    results = store.search(query_vec, top_k=5)
    assert results == []


def test_faiss_search_result_has_score():
    texts = ["Teks pertama untuk diuji.", "Teks kedua untuk dibandingkan."]
    metadata = [{"text": t} for t in texts]
    vectors = embed_texts(texts)

    store = FaissStore(dim=384)
    store.add(vectors, metadata)

    query_vec = embed_query("Teks pertama")
    results = store.search(query_vec, top_k=2)

    for r in results:
        assert "score" in r
        assert 0.0 <= r["score"] <= 1.1


def test_faiss_save_and_load(tmp_path):
    texts = ["Dokumen satu.", "Dokumen dua.", "Dokumen tiga."]
    metadata = [{"text": t, "source": "test.pdf"} for t in texts]
    vectors = embed_texts(texts)

    store = FaissStore(dim=384)
    store.add(vectors, metadata)

    index_path = str(tmp_path / "test.faiss")
    meta_path = str(tmp_path / "test.pkl")
    store.save(index_path, meta_path)

    store2 = FaissStore(dim=384)
    store2.load(index_path, meta_path)

    assert store2.total_vectors == 3
    query_vec = embed_query("Dokumen satu")
    results = store2.search(query_vec, top_k=1)
    assert len(results) == 1
