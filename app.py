# app.py - Flask web app    ->   python app.py   then open http://127.0.0.1:5000
import io
import os

from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

import deduplication as dd
import deep_learning as dl
from database import get_db, init_db

app = Flask(__name__)
app.secret_key = "change-this-secret-key"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB per upload

ALLOWED = {"pdf", "txt", "docx", "jpg", "jpeg", "png", "csv", "zip", "log", "json", "bin", "xlsx", "gif", "bmp"}
MAGIC = {"pdf": b"%PDF", "png": b"\x89PNG", "jpg": b"\xff\xd8\xff", "jpeg": b"\xff\xd8\xff",
         "zip": b"PK", "docx": b"PK", "xlsx": b"PK", "gif": b"GIF8", "bmp": b"BM"}


def check_file(filename, data):
    """Returns an error message, or None if the file is fine."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED:
        return "file type not allowed"
    if data and ext in MAGIC and not data.startswith(MAGIC[ext]):
        return "content does not match the .%s extension" % ext
    if ext in ("txt", "csv", "log", "json") and b"\x00" in data[:4096]:
        return "this does not look like a text file"
    return None


def human(n):
    n = float(n or 0)
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return "%d B" % n if unit == "B" else "%.2f %s" % (n, unit)
        n /= 1024


app.jinja_env.filters["size"] = human


@app.route("/", methods=["GET", "POST"])
def upload():
    results = []
    if request.method == "POST":
        mode = request.form.get("mode", "cdc")
        size = int(request.form.get("chunk_size", 4096))
        use_dl = request.form.get("use_dl") == "on"
        thr = request.form.get("threshold", "").strip()
        threshold = float(thr) if thr else None
        if mode not in ("cdc", "fixed") or size not in (1024, 2048, 4096, 8192, 16384):
            flash("Wrong chunking options", "danger")
            return redirect(url_for("upload"))
        db = get_db()
        for f in request.files.getlist("files"):
            if not f.filename:
                continue
            name = secure_filename(f.filename) or "file"      # removes ../ and strange characters
            data = f.read()
            error = check_file(name, data)
            if error:
                results.append({"filename": name, "error": error})
                continue
            try:
                ext = name.rsplit(".", 1)[-1].lower()
                results.append(dd.process_file(db, name, data, ext, mode, size, use_dl, threshold))
            except Exception as e:
                results.append({"filename": name, "error": "failed: %s" % e})
        db.close()
    return render_template("upload.html", results=results, page="upload",
                           model=dl.load_model() is not None, info=dl.model_info())


@app.route("/dashboard")
def dashboard():
    db = get_db()
    stats = dd.get_stats(db)
    recent = db.execute("SELECT * FROM files ORDER BY id DESC LIMIT 8").fetchall()
    per_file = db.execute("SELECT filename, size, stored_bytes FROM files ORDER BY id DESC LIMIT 12").fetchall()
    by_type = db.execute("SELECT file_type, SUM(size) s, SUM(stored_bytes) t FROM files GROUP BY file_type").fetchall()
    db.close()
    charts = {
        "names": [r["filename"] for r in per_file][::-1],
        "original": [r["size"] for r in per_file][::-1],
        "stored": [r["stored_bytes"] for r in per_file][::-1],
        "types": [r["file_type"] for r in by_type],
        "type_original": [r["s"] for r in by_type],
        "type_stored": [r["t"] for r in by_type],
    }
    return render_template("dashboard.html", page="dashboard", s=stats, recent=recent, charts=charts)


@app.route("/duplicates")
def duplicates():
    db = get_db()
    dup_files = db.execute("SELECT f.*, o.filename AS original_name FROM files f JOIN files o ON o.id = f.duplicate_of"
                           " ORDER BY f.id DESC").fetchall()
    logs = db.execute("SELECT l.*, f.filename FROM deduplication_logs l LEFT JOIN files f ON f.id = l.file_id"
                      " ORDER BY l.id DESC LIMIT 100").fetchall()
    results = db.execute("SELECT m.*, f.filename FROM model_results m LEFT JOIN files f ON f.id = m.file_id"
                         " ORDER BY m.id DESC LIMIT 100").fetchall()
    stats = dd.get_stats(db)
    db.close()
    return render_template("duplicates.html", page="duplicates", dup_files=dup_files, logs=logs,
                           results=results, s=stats, info=dl.model_info())


@app.route("/files")
def files():
    q = request.args.get("q", "").strip()
    db = get_db()
    if q:
        rows = db.execute("SELECT * FROM files WHERE filename LIKE ? OR file_type LIKE ? ORDER BY id DESC",
                          ("%" + q + "%", "%" + q + "%")).fetchall()
    else:
        rows = db.execute("SELECT * FROM files ORDER BY id DESC").fetchall()
    db.close()
    return render_template("files.html", page="files", files=rows, q=q)


@app.route("/download/<int:file_id>")
def download(file_id):
    db = get_db()
    row = db.execute("SELECT filename FROM files WHERE id=?", (file_id,)).fetchone()
    if row is None:
        db.close()
        return "File not found", 404
    try:
        data = dd.rebuild_file(db, file_id)    # rebuilt from chunks, hash is checked
    except dd.IntegrityError as e:
        db.close()
        return "Integrity check failed: %s" % e, 500
    db.close()
    return send_file(io.BytesIO(data), as_attachment=True, download_name=row["filename"])


@app.route("/delete/<int:file_id>", methods=["POST"])
def delete(file_id):
    db = get_db()
    removed = dd.delete_file(db, file_id)
    db.close()
    flash("File deleted (%d unused chunks removed from disk)." % removed, "success")
    return redirect(url_for("files"))


init_db()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
