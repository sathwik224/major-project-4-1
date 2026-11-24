# deduplication.py - the main part: store files without storing the same data twice
import os
import time

import numpy as np
import psutil

import chunking
import deep_learning as dl
import hashing
import storage_manager as sm
from database import now


class IntegrityError(Exception):
    pass


def read_chunk(db, row):
    """Get the original bytes of a chunk (full chunk or delta) and check its hash."""
    payload = sm.read_chunk_file(row["path"])
    if row["kind"] == "delta":
        base = db.execute("SELECT * FROM chunks WHERE id=?", (row["base_id"],)).fetchone()
        data = sm.apply_delta(payload, read_chunk(db, base))
    else:
        data = payload
    if hashing.sha256_hex(data) != row["hash"]:
        raise IntegrityError("chunk %s is damaged" % row["hash"][:10])
    return data


def log(db, file_id, chunk_id, event, saved):
    db.execute("INSERT INTO deduplication_logs (file_id, chunk_id, event, bytes_saved, created_at) VALUES (?,?,?,?,?)",
               (file_id, chunk_id, event, saved, now()))


def process_file(db, filename, data, file_type, mode="cdc", chunk_size=4096, use_dl=True,
                 threshold=None, max_delta=0.7, min_minhash=0.1):
    """Chunk -> SHA-256 -> exact duplicates -> deep learning candidates -> verify -> store."""
    start = time.time()
    model = dl.load_model() if use_dl else None
    use_dl = model is not None
    if threshold is None and model is not None:
        threshold = model["threshold"]

    file_hash = hashing.sha256_hex(data)
    same = db.execute("SELECT id FROM files WHERE sha256=? AND size=? ORDER BY id LIMIT 1",
                      (file_hash, len(data))).fetchone()
    result = {"filename": filename, "size": len(data), "chunks": 0, "exact": 0, "similar": 0, "stored": 0,
              "duplicate_of": same["id"] if same else None}
    new_files = []   # chunk files written by this upload (deleted again if something fails)

    try:
        user_id = db.execute("SELECT id FROM users LIMIT 1").fetchone()["id"]
        cur = db.execute("INSERT INTO files (user_id, filename, size, file_type, sha256, upload_date, duplicate_of)"
                         " VALUES (?,?,?,?,?,?,?)",
                         (user_id, filename, len(data), file_type, file_hash, now(), result["duplicate_of"]))
        file_id = cur.lastrowid
        if same:
            log(db, file_id, None, "exact_file", len(data))

        # vectors of the chunks that are already stored (only full chunks can be a base for a delta)
        ids, matrix = [], np.zeros((0, 32), dtype=np.float32)
        if use_dl:
            rows = db.execute("SELECT id, vector FROM chunks WHERE kind='full' AND vector IS NOT NULL").fetchall()
            ids = [r["id"] for r in rows]
            if rows:
                matrix = np.array([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])

        chunks = chunking.split_file(data, mode, chunk_size)
        for position, chunk in enumerate(chunks):
            h = hashing.sha256_hex(chunk)
            row = db.execute("SELECT id FROM chunks WHERE hash=?", (h,)).fetchone()

            if row:                                   # exact duplicate chunk: only add a reference
                chunk_id = row["id"]
                db.execute("UPDATE chunks SET ref_count = ref_count + 1 WHERE id=?", (chunk_id,))
                log(db, file_id, chunk_id, "exact_chunk", len(chunk))
                result["exact"] += 1
            else:                                     # new chunk
                vec, base_id, payload, tries = None, None, chunk, []
                if use_dl and len(chunk) >= 64:
                    vec = dl.embed(chunk)
                    sims = matrix @ vec if len(ids) else []
                    best = np.argsort(-np.array(sims))[:3] if len(ids) else []
                    sig = None
                    for i in best:
                        if sims[i] < threshold:
                            break
                        # candidate found by the neural network - now CHECK it before using it
                        base_row = db.execute("SELECT * FROM chunks WHERE id=?", (ids[i],)).fetchone()
                        base_data = read_chunk(db, base_row)
                        if sig is None:
                            sig = hashing.minhash(chunk)
                        mh = hashing.minhash_similarity(sig, hashing.minhash(base_data))
                        delta = sm.make_delta(chunk, base_data)
                        verified = sm.apply_delta(delta, base_data) == chunk
                        if not verified:
                            outcome = "verification failed"
                        elif mh < min_minhash:
                            outcome = "not really similar (minhash)"
                        elif len(delta) > max_delta * len(chunk):
                            outcome = "too little saving"
                        else:
                            outcome = "merged"
                        tries.append((ids[i], float(sims[i]), mh, int(verified), outcome))
                        if outcome == "merged":
                            base_id, payload = ids[i], delta
                            break

                kind = "delta" if base_id else "full"
                path = sm.save_chunk(h, payload)
                new_files.append(path)
                cur = db.execute("INSERT INTO chunks (hash, size, stored_size, path, kind, base_id, ref_count, vector)"
                                 " VALUES (?,?,?,?,?,?,1,?)",
                                 (h, len(chunk), len(payload), path, kind, base_id,
                                  vec.tobytes() if vec is not None else None))
                chunk_id = cur.lastrowid
                result["stored"] += len(payload)
                if kind == "delta":
                    log(db, file_id, chunk_id, "similar_chunk_merged", len(chunk) - len(payload))
                    result["similar"] += 1
                elif vec is not None:
                    ids.append(chunk_id)
                    matrix = np.vstack([matrix, vec])
                for base, cos, mh, ok, outcome in tries:
                    db.execute("INSERT INTO model_results (file_id, chunk_id, match_id, cosine, minhash, verified,"
                               " result, created_at) VALUES (?,?,?,?,?,?,?,?)",
                               (file_id, chunk_id, base, cos, mh, ok, outcome, now()))

            db.execute("INSERT INTO file_chunks (file_id, chunk_id, position) VALUES (?,?,?)",
                       (file_id, chunk_id, position))
            result["chunks"] += 1

        seconds = time.time() - start
        memory = psutil.Process().memory_info().rss / 1024 / 1024
        db.execute("UPDATE files SET chunk_count=?, stored_bytes=?, process_time=?, memory_mb=? WHERE id=?",
                   (result["chunks"], result["stored"], seconds, memory, file_id))
        db.commit()
    except Exception:
        db.rollback()
        for p in new_files:
            sm.delete_chunk_file(p)
        raise

    result["file_id"] = file_id
    result["seconds"] = round(seconds, 3)
    result["saved_percent"] = round(100 * (len(data) - result["stored"]) / len(data), 1) if data else 0
    result["throughput"] = round(len(data) / 1024 / 1024 / max(seconds, 1e-6), 2)
    return result


def rebuild_file(db, file_id):
    """Put the file together again from its chunks and check the SHA-256 of the result."""
    f = db.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
    rows = db.execute("SELECT c.* FROM file_chunks fc JOIN chunks c ON c.id = fc.chunk_id "
                      "WHERE fc.file_id=? ORDER BY fc.position", (file_id,)).fetchall()
    data = b"".join(read_chunk(db, r) for r in rows)
    if hashing.sha256_hex(data) != f["sha256"]:
        raise IntegrityError("rebuilt file does not match the original")
    return data


def delete_file(db, file_id):
    """Delete a file. A chunk is only removed when no file uses it and no delta needs it as base."""
    for r in db.execute("SELECT chunk_id FROM file_chunks WHERE file_id=?", (file_id,)).fetchall():
        db.execute("UPDATE chunks SET ref_count = ref_count - 1 WHERE id=?", (r["chunk_id"],))
    db.execute("DELETE FROM files WHERE id=?", (file_id,))      # file_chunks and logs are deleted too
    removed = []
    while True:
        unused = db.execute("SELECT id, path FROM chunks WHERE ref_count <= 0 AND id NOT IN "
                            "(SELECT base_id FROM chunks WHERE base_id IS NOT NULL)").fetchall()
        if not unused:
            break
        for c in unused:
            db.execute("DELETE FROM chunks WHERE id=?", (c["id"],))
            removed.append(c["path"])
    db.commit()
    for p in removed:                  # remove from disk only after the database change is saved
        sm.delete_chunk_file(p)
    return len(removed)


def get_stats(db):
    f = db.execute("SELECT COUNT(*) n, COALESCE(SUM(size),0) s, COALESCE(AVG(process_time),0) t,"
                   " COALESCE(MAX(memory_mb),0) m FROM files").fetchone()
    c = db.execute("SELECT COALESCE(SUM(size),0) unique_raw, COALESCE(SUM(stored_size),0) stored,"
                   " COUNT(*) n FROM chunks").fetchone()
    original, unique_raw, stored = f["s"], c["unique_raw"], c["stored"]
    merged = db.execute("SELECT COUNT(*) FROM deduplication_logs WHERE event='similar_chunk_merged'").fetchone()[0]
    mb = original / 1024 / 1024
    total_time = db.execute("SELECT COALESCE(SUM(process_time),0) FROM files").fetchone()[0]
    return {
        "files": f["n"], "original": original,
        "unique": unique_raw,                       # what is left after normal hash deduplication
        "stored": stored,                           # what is really on disk (hash + deep learning)
        "duplicate": original - stored,
        "saved_percent": round(100 * (original - stored) / original, 2) if original else 0,
        "ratio": round(original / stored, 2) if stored else 1,
        "hash_saved_percent": round(100 * (original - unique_raw) / original, 2) if original else 0,
        "hash_ratio": round(original / unique_raw, 2) if unique_raw else 1,
        "chunks": c["n"], "merged": merged,
        "avg_time": round(f["t"], 3), "memory": round(f["m"], 1),
        "throughput": round(mb / total_time, 2) if total_time else 0,
    }
