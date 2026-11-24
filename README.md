# Intelligent Data Deduplication and Storage Optimization Using Deep Learning

A small Flask web app that stores uploaded files without keeping the same data twice.
There is no login - it opens directly on the **Upload Files** page.

## How it works

1. Upload files (type and content are checked, file names are cleaned).
2. Each file is cut into chunks - fixed size or content defined (gear hash, like FastCDC).
3. SHA-256 of every chunk. A chunk that is already stored is not stored again, only a reference is added (exact duplicate).
4. For every new chunk a small autoencoder (256-128-32-128-256) makes a 32 number vector. Cosine similarity with the
   stored chunks finds "almost the same" chunks. MinHash is used as a second check.
5. **Verification:** the new chunk is saved only as the difference to the similar chunk, and only if the difference
   rebuilds the exact original (SHA-256 is compared). A similarity score alone never deletes or merges anything.
6. Download rebuilds the file from its chunks and checks the SHA-256 again.

## Run (Windows / VS Code)

Open the folder in VS Code, open a terminal and run:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 . (If PowerShell blocks `activate`, use a Command Prompt terminal.)

A trained model is already in `models/`. Optional commands:

| Command | What it does |
|---|---|
| `python make_samples.py` | creates test files in `sample_data/` |
| `python train_model.py` | trains the autoencoder again (about 5 seconds) |
| `python test_dedup.py` | runs the tests |

Try it: upload everything from `sample_data/` and look at the Dashboard and "Duplicates & Similarity" pages.
Then upload with "Use deep learning" switched off to see the difference.

## Files

| File | Job |
|---|---|
| `app.py` | web pages (upload, dashboard, duplicates, files, download, delete) |
| `database.py` | SQLite tables: users, files, chunks, file_chunks, deduplication_logs, model_results |
| `chunking.py` | fixed-size and content defined chunking |
| `hashing.py` | SHA-256 and MinHash |
| `deep_learning.py` | features + autoencoder encoder, loads `models/model.npz` |
| `train_model.py` | trains the autoencoder on original / modified / different chunks |
| `deduplication.py` | the whole pipeline, rebuild, safe delete, statistics |
| `storage_manager.py` | writes chunks to `data/chunks/`, delta helpers |
| `make_samples.py` | sample files and training data |
| `test_dedup.py` | tests |
| `templates/`, `static/style.css` | Bootstrap pages |

## Results (measured on this machine)

**Model** (1000 test pairs, 500 modified copies and 500 different chunks):

| | Precision | Recall | F1 |
|---|---|---|---|
| Autoencoder + cosine | 0.998 | 1.000 | 0.999 |
| Plain cosine on the features (no deep learning) | 0.998 | 1.000 | 0.999 |
| SHA-256 only | finds identical chunks only |

The two scores are equal because the generated test data is easy. The autoencoder gives a short 32 number vector per
chunk instead of 256, which makes the search smaller. Real data may behave differently, so train again on your own files if needed.

**Sample files** (11 files, 1.34 MB): hash-only deduplication saves **35.2 %** (ratio 1.54 : 1),
with deep learning **56.8 %** (ratio 2.31 : 1). All files rebuild identical.

**Tests** (`python test_dedup.py`, all 9 pass):

| Test | Result |
|---|---|
| Identical files, different names | second file stores 0 bytes and is marked duplicate |
| Slightly modified file (1 %) | hash only stores 99,969 bytes, with deep learning 2,360 bytes |
| Completely different files | nothing merged, everything stored |
| Repeated chunks (same 4 KB block 30 times) | stored once, 29 references |
| Empty file | 0 chunks, rebuilds as empty file |
| Large file (4 MB) + text inserted in the middle | 802 of 803 chunks are exact duplicates, 7 MB/s |
| Rebuild and integrity | all identical; a damaged chunk is detected |
| Safe delete | shared chunks and delta bases are kept, nothing left at the end |
| Insert 17 bytes at the start | CDC keeps 98 % of the chunks, fixed size 0 % |

## Notes and limits

- TensorFlow / PyTorch were not available when this was built, so the autoencoder is a scikit-learn `MLPRegressor`
  (a normal neural network). Only the weights are saved, so no extra library is needed to use it.
- Files are read fully into memory. Compressed formats (png, jpg, docx, zip) change completely after an edit, so only
  exact duplicates are found for them.
- Single user, local only (127.0.0.1). If it is ever put on a network, add login and change `secret_key` in `app.py`.
- The pages load Bootstrap and Chart.js from the internet (CDN), so the browser needs internet access.

## Future work

Streaming for very large files, a faster nearest-neighbour index for millions of chunks, unpacking docx/zip before
chunking, a PyTorch model trained on real data, user accounts, compression and encryption of stored chunks.
