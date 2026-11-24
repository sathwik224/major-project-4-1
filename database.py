# database.py - SQLite tables for the project
import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "dedup.db")


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    return db


def init_db():
    db = get_db()
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL,
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER REFERENCES users(id),
        filename TEXT NOT NULL,
        size INTEGER NOT NULL,
        file_type TEXT,
        sha256 TEXT NOT NULL,
        upload_date TEXT,
        chunk_count INTEGER DEFAULT 0,
        stored_bytes INTEGER DEFAULT 0,
        process_time REAL DEFAULT 0,
        memory_mb REAL DEFAULT 0,
        duplicate_of INTEGER
    );

    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        hash TEXT UNIQUE NOT NULL,
        size INTEGER NOT NULL,
        stored_size INTEGER NOT NULL,
        path TEXT NOT NULL,
        kind TEXT NOT NULL DEFAULT 'full',      -- 'full' or 'delta'
        base_id INTEGER REFERENCES chunks(id),  -- only for delta chunks
        ref_count INTEGER NOT NULL DEFAULT 0,
        vector BLOB
    );

    CREATE TABLE IF NOT EXISTS file_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
        chunk_id INTEGER NOT NULL REFERENCES chunks(id),
        position INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS deduplication_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
        chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
        event TEXT,
        bytes_saved INTEGER DEFAULT 0,
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS model_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
        chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
        match_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
        cosine REAL,
        minhash REAL,
        verified INTEGER,
        result TEXT,
        created_at TEXT
    );
    """)
    if db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        db.execute("INSERT INTO users (username, created_at) VALUES (?, ?)", ("default", now()))
    db.commit()
    db.close()
