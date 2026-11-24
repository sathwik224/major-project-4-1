# deep_learning.py - autoencoder features for chunks (uses the saved model from train_model.py)
import json
import os

import numpy as np

from database import BASE_DIR

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.npz")
INFO_PATH = os.path.join(BASE_DIR, "models", "model_info.json")
N_FEATURES = 256
_model = None


def features(data):
    """Count 3-byte groups of the chunk into 256 buckets (a kind of fingerprint of the content)."""
    arr = np.frombuffer(data, dtype=np.uint8).astype(np.uint64)
    if len(arr) < 3:
        arr = np.concatenate([arr, np.zeros(3 - len(arr), dtype=np.uint64)])
    groups = (arr[:-2] << np.uint64(16)) | (arr[1:-1] << np.uint64(8)) | arr[2:]
    buckets = ((groups * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(24)
    counts = np.bincount(buckets.astype(np.int64), minlength=N_FEATURES).astype(np.float32)
    f = np.sqrt(counts)
    return f / (np.linalg.norm(f) + 1e-9)


def encode(feats, m):
    """Encoder half of the autoencoder: 256 -> 128 -> 32 numbers."""
    x = (feats - m["mean"]) * m["scale"]
    h = np.tanh(x @ m["W0"] + m["b0"])
    z = np.tanh(h @ m["W1"] + m["b1"])
    return z - m["lat_mean"]       # centering makes the cosine more useful


def load_model():
    global _model
    if _model is None and os.path.exists(MODEL_PATH):
        d = np.load(MODEL_PATH)
        _model = {k: d[k] for k in d.files}
        _model["threshold"] = float(_model["threshold"])
    return _model


def embed(data):
    """Chunk bytes -> vector with 32 numbers and length 1 (so dot product = cosine similarity)."""
    m = load_model()
    z = encode(features(data), m)
    return (z / (np.linalg.norm(z) + 1e-9)).astype(np.float32)


def model_info():
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH) as f:
            return json.load(f)
    return None
