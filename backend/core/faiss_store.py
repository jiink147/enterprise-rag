"""
FAISS Store - Layer 2: Processing Pipeline

Manages a persistent FAISS vector index plus a parallel list of metadata
records (one per vector), so that a FAISS search result (an integer index)
can be mapped back to the original chunk text and its source document info.

Uses IndexFlatIP (inner product) over L2-normalized vectors, which is
mathematically equivalent to cosine similarity search -- appropriate for
small/medium corpora (< ~10,000 chunks), per the planning doc.
"""

import os
import pickle
import logging

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class FaissStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata = []

    def add(self, vectors, metadata_list):
        if len(vectors) != len(metadata_list):
            raise ValueError(
                f"vectors ({len(vectors)}) and metadata_list ({len(metadata_list)}) must have the same length"
            )
        if len(vectors) == 0:
            return

        vectors = np.array(vectors, dtype=np.float32)
        self.index.add(vectors)
        self.metadata.extend(metadata_list)

    def search(self, query_vector, top_k=10):
        if self.index.ntotal == 0:
            return []

        query_vector = np.array([query_vector], dtype=np.float32)
        top_k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(score)
            results.append(entry)

        return results

    def save(self, index_path, metadata_path):
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to '{index_path}'")

    def load(self, index_path, metadata_path):
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.warning(f"No existing index found at '{index_path}', starting fresh.")
            return

        self.index = faiss.read_index(index_path)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        logger.info(f"Loaded FAISS index ({self.index.ntotal} vectors) from '{index_path}'")

    @property
    def total_vectors(self):
        return self.index.ntotal


if __name__ == "__main__":
    from backend.core.embedder import embed_texts, embed_query

    texts = [
        "Kebijakan cuti tahunan karyawan adalah 12 hari per tahun.",
        "Gaji dibayarkan setiap tanggal 25 setiap bulan.",
        "Resep nasi goreng membutuhkan bawang putih dan kecap manis.",
    ]
    metadata = [
        {"text": texts[0], "source": "sop_hr.pdf", "page_num": 3},
        {"text": texts[1], "source": "sop_hr.pdf", "page_num": 5},
        {"text": texts[2], "source": "resep.docx", "page_num": None},
    ]

    print("Embedding sample texts...")
    vectors = embed_texts(texts)

    store = FaissStore(dim=384)
    store.add(vectors, metadata)
    print(f"Total vectors in store: {store.total_vectors}")

    query = "Berapa hari cuti yang didapat karyawan?"
    print(f"\nQuery: '{query}'")
    query_vec = embed_query(query)
    results = store.search(query_vec, top_k=2)

    for r in results:
        print(f"\nScore: {r['score']:.4f} | Source: {r['source']}")
        print(f"Text: {r['text']}")
