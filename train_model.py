# train_model.py - trains the autoencoder    ->   python train_model.py
import json
import random

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neural_network import MLPRegressor

import deep_learning as dl
import make_samples as ms


def make_data(n, size=4096):
    """n triples: original chunk, modified copy of it, and a different chunk."""
    original, modified, other = [], [], []
    for _ in range(n):
        c = ms.random_chunk(size)
        original.append(dl.features(c))
        modified.append(dl.features(ms.modify(c, random.uniform(0.5, 12))))
        other.append(dl.features(ms.random_chunk(size)))
    return np.array(original), np.array(modified), np.array(other)


def cosine_rows(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)


def best_threshold(scores, labels):
    best_t, best_f = 0.5, 0
    for t in np.arange(0.30, 0.9995, 0.0025):
        f = f1_score(labels, scores >= t, zero_division=0)
        if f > best_f:
            best_t, best_f = t, f
    return float(best_t)


def scores_for(model_encode, A, P, N):
    s = np.concatenate([cosine_rows(model_encode(A), model_encode(P)), cosine_rows(model_encode(A), model_encode(N))])
    y = np.concatenate([np.ones(len(A)), np.zeros(len(A))])
    return s, y


def report(scores, labels, t):
    pred = scores >= t
    return {"precision": round(precision_score(labels, pred, zero_division=0), 3),
            "recall": round(recall_score(labels, pred, zero_division=0), 3),
            "f1": round(f1_score(labels, pred, zero_division=0), 3), "threshold": round(float(t), 3)}


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    print("making training data...")
    A, P, N = make_data(2500)
    mean = np.concatenate([A, N]).mean(axis=0)

    # denoising autoencoder: input = original or modified chunk, target = the original chunk
    SCALE = 10.0   # features are small numbers, scaling them up makes training easier
    X = (np.vstack([A, P]) - mean) * SCALE
    Y = (np.vstack([A, A]) - mean) * SCALE
    print("training autoencoder 256-128-32-128-256 ...")
    net = MLPRegressor(hidden_layer_sizes=(128, 32, 128), activation="tanh", max_iter=300,
                       learning_rate_init=0.002, random_state=1)
    net.fit(X, Y)
    print("final loss:", round(net.loss_, 5))

    model = {"W0": net.coefs_[0], "b0": net.intercepts_[0], "W1": net.coefs_[1], "b1": net.intercepts_[1],
             "mean": mean, "scale": SCALE, "lat_mean": 0, "threshold": 0.8}
    # average latent vector of the training chunks (used for centering)
    model["lat_mean"] = np.concatenate([dl.encode(A, model), dl.encode(N, model)]).mean(axis=0)
    enc = lambda f: dl.encode(f, model)

    # choose the threshold on validation pairs, report on separate test pairs
    VA, VP, VN = make_data(500)
    s, y = scores_for(enc, VA, VP, VN)
    model["threshold"] = best_threshold(s, y)
    TA, TP, TN = make_data(500)
    s, y = scores_for(enc, TA, TP, TN)
    dl_result = report(s, y, model["threshold"])
    # for comparison: cosine on the plain 256 features without deep learning
    s2, y2 = scores_for(lambda f: f, TA, TP, TN)
    vs, vy = scores_for(lambda f: f, VA, VP, VN)
    raw_result = report(s2, y2, best_threshold(vs, vy))

    print("deep learning :", dl_result)
    print("plain features:", raw_result)

    np.savez("models/model.npz", **model)
    with open("models/model_info.json", "w") as f:
        json.dump({"layers": "256-128-32-128-256", "training_chunks": 5000, "test_pairs": 1000,
                   "deep_learning": dl_result, "plain_cosine": raw_result}, f, indent=2)
    print("model saved to models/model.npz")
