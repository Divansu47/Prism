from pathlib import Path
import sqlite3
import threading


ROOT = Path(__file__).resolve().parents[2]

DB_PATH = ROOT / "data" / "nutrition_db" / "food_lookup_usda.db"

_lock = threading.Lock()
_connection = None


def _create_connection():

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"USDA database not found at: {DB_PATH}"
        )

    conn = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
    )

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA query_only = TRUE")

    return conn


def get_connection():

    global _connection

    if _connection is not None:
        return _connection

    with _lock:

        if _connection is None:
            _connection = _create_connection()

    return _connection


def close_connection():

    global _connection

    with _lock:

        if _connection is not None:
            _connection.close()
            _connection = None


def get_food_count():

    conn = get_connection()

    row = conn.execute(
        "SELECT COUNT(*) AS count FROM foods"
    ).fetchone()

    return row["count"]


def get_by_fdc_id(fdc_id):

    conn = get_connection()

    row = conn.execute(
        "SELECT * FROM foods WHERE fdc_id = ?",
        (fdc_id,),
    ).fetchone()

    if row is None:
        return None

    return dict(row)