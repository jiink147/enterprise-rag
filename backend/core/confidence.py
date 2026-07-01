"""
Confidence Scoring - Layer 3: Retrieval & Ranking

Evaluates how confident the system is that the retrieved chunks
can answer the user's question, and decides whether to answer,
answer with disclaimer, or refuse.
"""

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

THRESHOLD_HIGH = 0.7
THRESHOLD_MEDIUM = 0.4


def compute_confidence(chunks, use_reranker=True):
    """
    Compute confidence level based on the top chunk's score.

    Args:
        chunks: List of chunk dicts returned by retrieve().
                Each chunk has either "rerank_score" (if re-ranker was used)
                or "score" (raw FAISS cosine similarity).
        use_reranker: Whether re-ranker scores are available.

    Returns:
        dict:
            {
                "level": "HIGH" | "MEDIUM" | "LOW",
                "score": float,   # the top chunk's score
                "should_answer": bool,
                "disclaimer": str | None,
            }
    """
    if not chunks:
        return {
            "level": CONFIDENCE_LOW,
            "score": 0.0,
            "should_answer": False,
            "disclaimer": "Tidak ditemukan informasi relevan dalam dokumen yang tersedia.",
        }

    # Use rerank_score if available, otherwise fall back to FAISS cosine score
    top_chunk = chunks[0]
    if use_reranker and "rerank_score" in top_chunk:
        score = top_chunk["rerank_score"]
        # Normalize rerank score to 0-1 range (cross-encoder outputs can exceed 1.0)
        score = max(0.0, min(score / 10.0, 1.0))
    else:
        score = top_chunk.get("score", 0.0)

    if score >= THRESHOLD_HIGH:
        return {
            "level": CONFIDENCE_HIGH,
            "score": round(score, 4),
            "should_answer": True,
            "disclaimer": None,
        }
    elif score >= THRESHOLD_MEDIUM:
        return {
            "level": CONFIDENCE_MEDIUM,
            "score": round(score, 4),
            "should_answer": True,
            "disclaimer": "Jawaban berikut berdasarkan informasi yang terbatas dalam dokumen.",
        }
    else:
        return {
            "level": CONFIDENCE_LOW,
            "score": round(score, 4),
            "should_answer": False,
            "disclaimer": "Maaf, saya tidak menemukan informasi yang cukup relevan dalam dokumen yang tersedia.",
        }


def format_answer_with_confidence(answer_text, confidence, citations):
    """
    Format the final answer with confidence indicator and citations.

    Args:
        answer_text: Raw answer string from LLM.
        confidence: dict from compute_confidence().
        citations: list from build_citation_list().

    Returns:
        dict (the final structured response):
            {
                "answer": str,
                "confidence": str,
                "confidence_score": float,
                "disclaimer": str | None,
                "citations": list,
            }
    """
    return {
        "answer": answer_text if confidence["should_answer"] else confidence["disclaimer"],
        "confidence": confidence["level"],
        "confidence_score": confidence["score"],
        "disclaimer": confidence["disclaimer"],
        "citations": citations if confidence["should_answer"] else [],
    }


if __name__ == "__main__":
    # Simulate HIGH confidence
    chunks_high = [{"text": "cuti 12 hari", "score": 0.85, "rerank_score": 8.5}]
    print("HIGH confidence:")
    print(compute_confidence(chunks_high))

    # Simulate MEDIUM confidence
    chunks_medium = [{"text": "cuti tahunan", "score": 0.55, "rerank_score": 4.5}]
    print("\nMEDIUM confidence:")
    print(compute_confidence(chunks_medium))

    # Simulate LOW confidence
    chunks_low = [{"text": "resep masakan", "score": 0.15, "rerank_score": 1.2}]
    print("\nLOW confidence:")
    print(compute_confidence(chunks_low))

    # Simulate empty
    print("\nEMPTY chunks:")
    print(compute_confidence([]))
