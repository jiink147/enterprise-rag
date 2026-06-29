"""
Document Loader - Layer 1: Document Ingestion
Parses various document formats (PDF, DOCX, XLSX, TXT) into structured text + metadata.

M2 scope: PDF parser only.
"""

import fitz  # PyMuPDF
import logging

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> list[dict]:
    """
    Parse a PDF file and extract text per page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A list of dicts, one per page, in the form:
        [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}, ...]

        Returns an empty list if the file cannot be parsed (and logs the error).
    """
    pages = []

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        logger.error(f"Failed to open PDF '{file_path}': {e}")
        return pages

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            text = page.get_text()

            # Clean up: strip leading/trailing whitespace
            text = text.strip()

            pages.append({
                "page_num": page_index + 1,  # human-friendly, 1-indexed
                "text": text,
            })
    except Exception as e:
        logger.error(f"Failed while extracting text from '{file_path}': {e}")
    finally:
        doc.close()

    return pages


if __name__ == "__main__":
    # Quick manual test when running this file directly:
    # python backend/core/document_loader.py path/to/sample.pdf
    import sys

    if len(sys.argv) < 2:
        print("Usage: python document_loader.py <path_to_pdf>")
        sys.exit(1)

    result = parse_pdf(sys.argv[1])
    print(f"Total pages parsed: {len(result)}")
    for p in result[:2]:  # preview first 2 pages only
        preview = p["text"][:200].replace("\n", " ")
        print(f"\n--- Page {p['page_num']} ---")
        print(f"{preview}...")
