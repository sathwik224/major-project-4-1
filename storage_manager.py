# storage_manager.py - saves chunks on disk, delta helpers
import os
import zlib

from database import BASE_DIR

STORAGE_DIR = os.path.join(BASE_DIR, "data", "chunks")


def save_chunk(hash_hex, payload):
    """Write a chunk to disk. The name is the SHA-256 hash, so a user can never choose the path."""
    folder = os.path.join(STORAGE_DIR, hash_hex[:2])
    os.makedirs(folder, exist_ok=True)
    rel_path = os.path.join(hash_hex[:2], hash_hex)
    with open(os.path.join(STORAGE_DIR, rel_path), "wb") as f:
        f.write(payload)
    return rel_path


def read_chunk_file(rel_path):
    full = os.path.abspath(os.path.join(STORAGE_DIR, rel_path))
    if not full.startswith(os.path.abspath(STORAGE_DIR)):
        raise ValueError("bad path")
    with open(full, "rb") as f:
        return f.read()


def delete_chunk_file(rel_path):
    full = os.path.abspath(os.path.join(STORAGE_DIR, rel_path))
    if full.startswith(os.path.abspath(STORAGE_DIR)) and os.path.exists(full):
        os.remove(full)


def make_delta(chunk, base):
    """Store only the difference to a similar chunk: zlib with the base chunk as dictionary."""
    c = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY, base[-32768:])
    return c.compress(chunk) + c.flush()


def apply_delta(delta, base):
    d = zlib.decompressobj(15, base[-32768:])
    return d.decompress(delta) + d.flush()
