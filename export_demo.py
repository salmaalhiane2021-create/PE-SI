"""
export_demo.py
==============
Exporte la base SQLite de demonstration vers ``demo_data.json``.

Le fichier JSON sert de **fixture** a l'application Streamlit
(``streamlit_app.py``) : il evite de re-entrainer / re-generer l'historique
a chaque demarrage sur le cloud (cold start instantane).

Utilisation :
    python export_demo.py                 # lit novapay_demo.db
    NOVAPAY_DB=autre.db python export_demo.py
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = os.getenv("NOVAPAY_DB", "novapay_demo.db")
OUT_PATH = Path(__file__).resolve().parent / "demo_data.json"


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    users = [dict(row) for row in conn.execute("SELECT * FROM users ORDER BY user_id")]

    transactions = []
    for row in conn.execute("SELECT * FROM transactions ORDER BY created_at DESC"):
        tx = dict(row)
        try:
            tx["factors"] = json.loads(tx.get("factors") or "[]")
        except (TypeError, json.JSONDecodeError):
            tx["factors"] = []
        try:
            tx["components"] = json.loads(tx.get("components") or "{}")
        except (TypeError, json.JSONDecodeError):
            tx["components"] = {}
        tx["is_anomaly"] = bool(tx.get("is_anomaly"))
        transactions.append(tx)

    conn.close()

    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "source_db": str(DB_PATH),
        "users": users,
        "transactions": transactions,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"[EXPORT] {len(users)} utilisateurs, {len(transactions)} transactions -> {OUT_PATH.name}")
    for user in users:
        count = sum(1 for t in transactions if t["user_id"] == user["user_id"])
        print(f"  - {user['user_id']} {user['name']:<15} solde {user['balance']:>9.2f} EUR  ({count} transactions)")


if __name__ == "__main__":
    main()
