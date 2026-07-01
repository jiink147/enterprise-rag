"""
Chunker - Layer 2: Processing Pipeline

Splits extracted text into overlapping chunks suitable for embedding.
Uses a recursive character splitting strategy: try splitting on paragraph
breaks first, then sentence breaks, then word breaks, then raw characters --
falling back to a coarser separator only when a finer one can't produce
small-enough pieces.
"""

SEPARATORS = ["\n\n", ". ", "! ", "? ", " ", ""]


def _split_text(text, chunk_size):
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    for sep in SEPARATORS:
        if sep == "":
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        if sep in text:
            parts = text.split(sep)
            pieces = []
            buffer = ""
            for i, part in enumerate(parts):
                candidate = buffer + part + (sep if i < len(parts) - 1 else "")
                if len(candidate) <= chunk_size:
                    buffer = candidate
                else:
                    if buffer.strip():
                        pieces.append(buffer)
                    buffer = part + (sep if i < len(parts) - 1 else "")
            if buffer.strip():
                pieces.append(buffer)

            if len(pieces) > 1:
                final_pieces = []
                for p in pieces:
                    if len(p) > chunk_size:
                        final_pieces.extend(_split_text(p, chunk_size))
                    else:
                        final_pieces.append(p)
                return final_pieces

    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_text(text, chunk_size=500, overlap=100, min_chunk_size=50):
    if not text or not text.strip():
        return []

    raw_pieces = _split_text(text.strip(), chunk_size)

    chunks = []
    for i, piece in enumerate(raw_pieces):
        piece = piece.strip()
        if i > 0 and overlap > 0:
            prev_tail = raw_pieces[i - 1].strip()[-overlap:]
            piece = (prev_tail + " " + piece).strip()
        if len(piece) >= min_chunk_size:
            chunks.append(piece)

    return chunks


def chunk_document(parsed_doc, chunk_size=500, overlap=100):
    output_chunks = []
    chunk_index = 0

    for unit in parsed_doc.get("chunks_raw", []):
        unit_text = unit.get("text", "")
        if not unit_text.strip():
            continue

        pieces = chunk_text(unit_text, chunk_size=chunk_size, overlap=overlap)

        cursor = 0
        for piece in pieces:
            start = unit_text.find(piece, cursor)
            if start == -1:
                start = cursor
            end = start + len(piece)
            cursor = max(cursor, end - overlap if overlap > 0 else end)

            output_chunks.append({
                "chunk_index": chunk_index,
                "text": piece,
                "page_num": unit.get("page_num"),
                "section_title": unit.get("section_title"),
                "sheet_name": unit.get("sheet_name"),
                "char_start": start,
                "char_end": end,
            })
            chunk_index += 1

    return output_chunks


if __name__ == "__main__":
    sample = (
        "Ini adalah paragraf pertama. Ia berisi beberapa kalimat untuk diuji. "
        "Tujuannya adalah memastikan chunker bekerja dengan baik.\n\n"
        "Ini adalah paragraf kedua, yang membahas topik berbeda sama sekali. "
        "Sistem RAG ini akan menggunakan teks ini sebagai contoh chunking sederhana."
    )
    result = chunk_text(sample, chunk_size=100, overlap=20)
    print(f"Total chunks: {len(result)}")
    for i, c in enumerate(result):
        print(f"\n--- Chunk {i} ({len(c)} chars) ---")
        print(c)
