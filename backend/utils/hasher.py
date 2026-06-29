"""
Hasher - Utility for document deduplication.

Computes an MD5 hash of a file's contents so the ingestion pipeline can
detect when a file has already been uploaded before (idempotent ingestion).
"""

import hashlib


def compute_file_hash(file_path, chunk_size=8192):
    """
    Compute the MD5 hash of a file's contents.

    Reads the file in chunks rather than all at once, so large files
    (e.g. big PDFs) don't get fully loaded into memory just to hash them.
    """
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5.update(chunk)

    return md5.hexdigest()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python hasher.py <path_to_file>")
        sys.exit(1)

    file_hash = compute_file_hash(sys.argv[1])
    print(f"MD5 hash: {file_hash}")
