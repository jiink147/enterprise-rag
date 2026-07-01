"""
Prompt Builder - Layer 4: Generation

Assembles the final prompt sent to the LLM from:
- A fixed system instruction
- Retrieved chunks (context)
- The user's question
"""


SYSTEM_PROMPT = """Kamu adalah asisten AI perusahaan yang menjawab pertanyaan HANYA berdasarkan dokumen yang diberikan dalam konteks di bawah ini.

Aturan penting:
1. Jawab HANYA berdasarkan informasi yang ada dalam konteks.
2. Jika informasi tidak ada dalam konteks, katakan dengan jelas: "Maaf, saya tidak menemukan informasi tersebut dalam dokumen yang tersedia."
3. Jangan mengarang atau menambahkan informasi dari luar konteks.
4. Selalu sebutkan sumber dokumen di akhir jawaban.
5. Jawab dalam Bahasa Indonesia yang jelas dan profesional."""


def build_prompt(query, chunks):
    """
    Build a full prompt string from a user query and retrieved chunks.

    Args:
        query: The user's question (string).
        chunks: List of chunk dicts from FaissStore.search(), each containing
                at minimum "text" and optionally "source", "page_num",
                "section_title", "sheet_name", "score".

    Returns:
        A single prompt string ready to be sent to the LLM.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "Tidak diketahui")
        page = chunk.get("page_num")
        section = chunk.get("section_title")
        sheet = chunk.get("sheet_name")
        score = chunk.get("score", 0)
        text = chunk.get("text", "").strip()

        # Build a human-readable location label for this chunk
        if page:
            location = f"Halaman {page}"
        elif section:
            location = f"Bagian: {section}"
        elif sheet:
            location = f"Sheet: {sheet}"
        else:
            location = "Lokasi tidak diketahui"

        context_parts.append(
            f"[Dokumen {i}] Sumber: {source} | {location} | Relevansi: {score:.2f}\n{text}"
        )

    context_block = "\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

---KONTEKS DOKUMEN---
{context_block}
---AKHIR KONTEKS---

Pertanyaan: {query}

Jawaban:"""

    return prompt


def build_citation_list(chunks):
    """
    Build a structured list of citations from retrieved chunks.

    Returns:
        List of dicts:
        [{"source": str, "page_num": int|None, "section_title": str|None,
          "sheet_name": str|None, "score": float}, ...]
    """
    citations = []
    for chunk in chunks:
        citations.append({
            "source": chunk.get("source", "Tidak diketahui"),
            "page_num": chunk.get("page_num"),
            "section_title": chunk.get("section_title"),
            "sheet_name": chunk.get("sheet_name"),
            "score": round(chunk.get("score", 0), 4),
        })
    return citations


if __name__ == "__main__":
    sample_chunks = [
        {
            "text": "Kebijakan cuti tahunan karyawan adalah 12 hari kerja per tahun kalender.",
            "source": "sop_hr.pdf",
            "page_num": 3,
            "score": 0.91,
        },
        {
            "text": "Pengajuan cuti harus dilakukan minimal 3 hari sebelum tanggal cuti dimulai.",
            "source": "sop_hr.pdf",
            "page_num": 4,
            "score": 0.78,
        },
    ]

    prompt = build_prompt("Berapa hari cuti tahunan yang didapat karyawan?", sample_chunks)
    print("=== GENERATED PROMPT ===")
    print(prompt)
    print("\n=== CITATIONS ===")
    citations = build_citation_list(sample_chunks)
    for c in citations:
        print(c)
