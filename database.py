import sqlite3
from datetime import datetime

DB_PATH = "rtads.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            temperature REAL,
            speed REAL,
            pressure REAL,
            status TEXT,
            confidence INTEGER,
            votes INTEGER,
            iso_result TEXT,
            lof_result TEXT,
            svm_result TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_reading(data: dict):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO readings (
            timestamp, temperature, speed, pressure,
            status, confidence, votes,
            iso_result, lof_result, svm_result
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["temperature"],
        data["speed"],
        data["pressure"],
        data["status"],
        data["confidence"],
        data["votes"],
        data["iso"],
        data["lof"],
        data["svm"]
    ))
    conn.commit()
    conn.close()

def get_stats():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM readings")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM readings WHERE status = 'ANOMALY'")
    anomalies = cursor.fetchone()[0]
    conn.close()
    return {
        "total": total,
        "anomalies": anomalies,
        "normal": total - anomalies,
        "anomaly_rate": round(anomalies / total * 100, 1) if total > 0 else 0
    }

def get_recent(limit=20):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, temperature, speed, pressure,
               status, confidence, votes
        FROM readings
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()
print("Database ready: rtads.db")