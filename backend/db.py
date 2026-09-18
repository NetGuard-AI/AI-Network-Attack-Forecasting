"""
db.py - plain sqlite3 storage for the MVP.

Deliberately NOT using an ORM or Postgres, per the MVP scope: this is a
single-file local database good enough for demo-scale data. Every function
opens its own short-lived connection (SQLite handles that fine at this
volume) so we don't have to think about connection pooling.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent / "attack_forecasting.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS traffic_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            flow_seq TEXT,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            traffic_id INTEGER,
            ts REAL NOT NULL,
            label TEXT NOT NULL,
            confidence REAL NOT NULL,
            is_attack INTEGER NOT NULL,
            injected INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (traffic_id) REFERENCES traffic_logs(id)
        );

        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            forecast_probability REAL NOT NULL,
            risk_level TEXT NOT NULL,
            trend TEXT NOT NULL,
            forecast_window_minutes INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            source TEXT NOT NULL,
            resolved INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_traffic_ts ON traffic_logs(ts);
        CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts);
        CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);
        """
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------- traffic --

def insert_traffic(record: dict, flow_seq: Optional[str] = None) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO traffic_logs (ts, flow_seq, raw_json) VALUES (?, ?, ?)",
        (time.time(), flow_seq, json.dumps(record)),
    )
    conn.commit()
    tid = cur.lastrowid
    conn.close()
    return tid


def recent_traffic(limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM traffic_logs ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def traffic_since(cutoff_ts: float) -> list[dict]:
    """Raw traffic rows (with parsed feature dict) newer than cutoff_ts."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM traffic_logs WHERE ts >= ? ORDER BY ts ASC", (cutoff_ts,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------- detections --

def insert_detection(
    traffic_id: Optional[int], label: str, confidence: float, is_attack: bool,
    injected: bool = False,
) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO detections (traffic_id, ts, label, confidence, is_attack, injected) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (traffic_id, time.time(), label, confidence, int(is_attack), int(injected)),
    )
    conn.commit()
    did = cur.lastrowid
    conn.close()
    return did


def recent_detections(limit: int = 50) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM detections ORDER BY ts DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def detections_since(cutoff_ts: float) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM detections WHERE ts >= ? ORDER BY ts ASC", (cutoff_ts,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# -------------------------------------------------------------- forecasts --

def insert_forecast(probability: float, risk_level: str, trend: str, window_minutes: int) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO forecasts (ts, forecast_probability, risk_level, trend, forecast_window_minutes) "
        "VALUES (?, ?, ?, ?, ?)",
        (time.time(), probability, risk_level, trend, window_minutes),
    )
    conn.commit()
    fid = cur.lastrowid
    conn.close()
    return fid


def latest_forecast() -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM forecasts ORDER BY ts DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ------------------------------------------------------------------ alerts --

def insert_alert(severity: str, message: str, source: str) -> int:
    conn = get_conn()
    cur = conn.execute(
        "INSERT INTO alerts (ts, severity, message, source) VALUES (?, ?, ?, ?)",
        (time.time(), severity, message, source),
    )
    conn.commit()
    aid = cur.lastrowid
    conn.close()
    return aid


def recent_alerts(limit: int = 50, active_only: bool = False) -> list[dict]:
    conn = get_conn()
    if active_only:
        rows = conn.execute(
            "SELECT * FROM alerts WHERE resolved = 0 ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------- stats --

def get_stats() -> dict[str, Any]:
    conn = get_conn()
    total_traffic = conn.execute("SELECT COUNT(*) c FROM traffic_logs").fetchone()["c"]
    total_detections = conn.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
    total_attacks = conn.execute(
        "SELECT COUNT(*) c FROM detections WHERE is_attack = 1"
    ).fetchone()["c"]
    active_alerts = conn.execute(
        "SELECT COUNT(*) c FROM alerts WHERE resolved = 0"
    ).fetchone()["c"]
    conn.close()

    attack_rate_pct = round(100 * total_attacks / total_detections, 2) if total_detections else 0.0

    return {
        "total_traffic": total_traffic,
        "total_detections": total_detections,
        "total_attacks": total_attacks,
        "attack_rate_pct": attack_rate_pct,
        "active_alerts": active_alerts,
        "updated_at": time.time(),
    }
