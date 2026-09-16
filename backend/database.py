"""
database.py — Dual-mode persistence layer for Neural-Link.

MongoDB is OPTIONAL. If unavailable, module falls back to SQLite silently.
"""

from __future__ import annotations

import os
import sqlite3
import asyncio
from utils import setup_logger

logger = setup_logger("Database")

# ─── SQLite ───────────────────────────────────────────────────────────────────

def get_sqlite_conn(db_path: str = "telemetry.db"):
    """
    Opens (or creates) the SQLite database and ensures the schema exists.
    Returns (connection, cursor).  Thread-safe via check_same_thread=False.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id       TEXT,
            timestamp        REAL,
            dataset          TEXT,
            focus            REAL,
            attention        REAL,
            stress           REAL,
            fatigue          REAL,
            emotion          TEXT,
            valence          REAL,
            arousal          REAL,
            confidence       REAL,
            inference_latency REAL
        )
    """)
    conn.commit()
    logger.info(f"SQLite ready -> {os.path.abspath(db_path)}")
    return conn, cursor


# ─── MongoDB (optional) ───────────────────────────────────────────────────────

# Lazy import so that a missing 'motor' package only raises at call-time.
_mongo_available = False
users_collection = None   # set during init_mongodb if connection succeeds


async def init_mongodb():
    """
    Attempts to connect to MongoDB.  If the server is unreachable or the
    package is missing, logs a single WARNING and continues in SQLite-only
    mode — no exception is propagated.
    """
    global _mongo_available, users_collection

    mongo_url = os.environ.get("MONGO_URL", "mongodb://localhost:27017")

    try:
        from motor.motor_asyncio import AsyncIOMotorClient  # type: ignore

        client = AsyncIOMotorClient(
            mongo_url,
            serverSelectionTimeoutMS=2_000,   # fail-fast: 2 s
            connectTimeoutMS=2_000,
        )

        # Ping the deployment to verify the connection
        await asyncio.wait_for(
            client.admin.command("ping"),
            timeout=3.0,
        )

        db = client["bci_database"]
        users_collection = db["users"]

        # Seed a default user profile if the collection is empty
        count = await users_collection.count_documents({})
        if count == 0:
            await users_collection.insert_one({
                "username": "Dr. E. Tyrell",
                "role": "Lead Researcher",
                "settings": {"theme": "cyberpunk", "default_dataset": "PhysioNet"},
            })

        _mongo_available = True
        logger.info(f"MongoDB connected -> {mongo_url}")

    except Exception as exc:
        _mongo_available = False
        logger.info(
            f"MongoDB unavailable ({type(exc).__name__}: {exc}) - "
            "running in SQLite-only mode."
        )
