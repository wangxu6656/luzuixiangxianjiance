import csv
import os
import sqlite3
import threading
from datetime import datetime
from typing import List, Optional


class HistoryDB:
    def __init__(self, path: str = "history.db"):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._local = threading.local()
        self._schema()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path)
            self._local.conn = conn
        return conn

    def _schema(self) -> None:
        self._conn().execute(
            """CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                source TEXT,
                filter_found INTEGER,
                has_line INTEGER,
                line_count INTEGER,
                offset_ratio REAL,
                offset_mm REAL,
                qualified INTEGER,
                reject INTEGER,
                positions TEXT
            )"""
        )
        self._conn().commit()

    def add(self, source: str, result) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        positions = ";".join(f"({ln.x},{ln.y})" for ln in result.lines)
        self._conn().execute(
            "INSERT INTO detections VALUES (NULL,?,?,?,?,?,?,?,?,?,?)",
            (
                ts,
                source,
                int(result.filter_found),
                int(result.has_flavor_line),
                len(result.lines),
                result.offset_ratio,
                result.offset_mm,
                None if result.qualified is None else int(result.qualified),
                int(result.reject),
                positions,
            ),
        )
        self._conn().commit()

    def query(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        result_filter: str = "all",
    ) -> List[tuple]:
        sql = "SELECT * FROM detections WHERE 1=1"
        params: list = []
        if start:
            sql += " AND ts >= ?"
            params.append(start)
        if end:
            sql += " AND ts <= ?"
            params.append(end)
        if result_filter == "reject":
            sql += " AND reject = 1"
        elif result_filter == "ok":
            sql += " AND reject = 0"
        sql += " ORDER BY id DESC"
        return self._conn().execute(sql, params).fetchall()

    def export_csv(self, rows: List[tuple], path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(
                ["ID", "时间", "来源", "滤嘴", "有香线", "数量", "偏移比", "偏移mm", "合格", "剔除", "位置"]
            )
            for r in rows:
                w.writerow(list(r))

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None