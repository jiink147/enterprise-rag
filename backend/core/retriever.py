"""
Retriever - Layer 3: Retrieval & Ranking

Combines FAISS vector search with cross-encoder re-ranking to find
the most relevant chunks for a given query.
"""

import logging
from backend.core.embedder import embed_query
from backend.core.faiss_store import FaissStore

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading re-ranker model '{RERANKER_MODEL}'...")
        _reranker = CrossEncoder(RERANKER_MODEL)
        logger.info("Re-ranker loaded.")
    return _reranker


def retrieve(query, store, top_k_faiss=10, top_k_final=3, use_reranker=True):
    """
    Retrieve the most relevant chunks for a query.

    Pipeline:
        1. Embed query → FAISS search (top_k_faiss candidates)
        2. (Optional) Cross-encoder re-rank → keep top_k_final
        3. Return ranked list of chunk dicts with scores

    Args:
        query: User's question string.
        store: FaissStore instance (already populated with document vectors).
        top_k_faiss: How many candidates to pull from FAISS before re-ranking.
        top_k_final: How many chunks to return after re-ranking.
        use_reranker: Set False to skip re-ranking (faster but less accurate).

    Returns:
        List of chunk dicts, each with a "score" key, sorted by relevance descending.
    """
    if store.total_vectors == 0:
        logger.warning("FAISS store is empty — no documents ingested yet.")
        return []

    # Step 1: FAISS vector search
    query_vec = embed_query(query)
    candidates = store.search(query_vec, top_k=top_k_faiss)

    if not candidates:
        return []

    if not use_reranker or len(candidates) == 0:
        return candidates[:top_k_final]

    # Step 2: Cross-encoder re-ranking
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k_final]


if __name__ == "__main__":
    from backend.core.document_loader import DocumentLoader
    from backend.core.chunker import chunk_document
    from backend.core.embedder import embed_texts

    print("Loading document and building FAISS store...")
    loader = DocumentLoader()
    doc = loader.load("data/documents/samples/Fachrurrozi_Rosyadi_Resume.pdf")
    chunks = chunk_document(doc, chunk_size=500, overlap=100)
    texts = [c["text"] for c in chunks]
    vectors = embed_texts(texts)

    store = FaissStore(dim=384)
    metadata = []
    for c in chunks:
        metadata.append({
            "text": c["text"],
            "source": doc["file_name"],
            "page_num": c.get("page_num"),
            "section_title": c.get("section_title"),
        })
    store.add(vectors, metadata)

    query = "Apa pengalaman kerja kandidat?"
    print(f"\nQuery: '{query}'")
    print("\n--- WITHOUT re-ranking ---")
    results_no_rerank = retrieve(query, store, use_reranker=False)
    for r in results_no_rerank:
        print(f"  FAISS score: {r['score']:.4f} | {r['text'][:80]}...")

    print("\n--- WITH re-ranking ---")
    results_reranked = retrieve(query, store, use_reranker=True)
    for r in results_reranked:
        print(f"  Rerank score: {r['rerank_score']:.4f} | {r['text'][:80]}...")
