# hashing.py - SHA-256 for exact duplicates, MinHash for approximate similarity
import hashlib
import numpy as np

PRIME = 4294967311          # a prime a bit bigger than 2**32
NUM_HASHES = 64
_rng = np.random.RandomState(1)
A = _rng.randint(1, 2 ** 31, NUM_HASHES).astype(np.uint64)
B = _rng.randint(0, 2 ** 31, NUM_HASHES).astype(np.uint64)


def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def minhash(data):
    """MinHash signature of the 4-byte shingles of a chunk."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    if len(arr) < 4:
        arr = np.concatenate([arr, np.zeros(4 - len(arr), dtype=np.uint64)])
    shingles = (arr[:-3] << np.uint64(24)) | (arr[1:-2] << np.uint64(16)) | (arr[2:-1] << np.uint64(8)) | arr[3:]
    shingles = np.unique(shingles)
    return np.array([((a * shingles + b) % PRIME).min() for a, b in zip(A, B)])


def minhash_similarity(sig1, sig2):
    # share of equal positions = estimated Jaccard similarity
    return float(np.mean(sig1 == sig2))
