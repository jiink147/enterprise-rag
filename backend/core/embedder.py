"""
Embedder - Layer 2: Processing Pipeline

Wraps sentence-transformers to convert text chunks into normalized
384-dimensional embedding vectors, using a multilingual model that
supports Indonesian.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model = None


def get_model():
    global _model
    if _model is None:
        logger.info(f"Loading embedding model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _model


def embed_texts(texts, batch_size=64):
    if not texts:
        return np.empty((0, 384), dtype=np.float32)

    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.array(embeddings, dtype=np.float32)


def embed_query(query):
    result = embed_texts([query])
    return result[0]


if __name__ == "__main__":
    sample_texts = [
        "Kebijakan cuti tahunan karyawan adalah 12 hari per tahun.",
        "Annual leave policy for employees is 12 days per year.",
        "Resep nasi goreng membutuhkan bawang putih dan kecap manis.",
    ]

    print("Loading model and embedding sample texts...")
    vectors = embed_texts(sample_texts)
    print(f"Shape: {vectors.shape}")
    print(f"First vector (first 5 dims): {vectors[0][:5]}")

    from numpy import dot
    sim_related = dot(vectors[0], vectors[1])
    sim_unrelated = dot(vectors[0], vectors[2])
    print(f"\nSimilarity (leave policy ID vs EN): {sim_related:.4f}")
    print(f"Similarity (leave policy vs recipe): {sim_unrelated:.4f}")
    print("\n(Expect the first similarity score to be notably higher than the second.)")
