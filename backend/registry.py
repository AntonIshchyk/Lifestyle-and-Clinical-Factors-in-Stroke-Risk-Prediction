import pandas as pd
import sqlite3
import json
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "healthcare.db"

def register_dataset(name: str, df: pd.DataFrame):
    """Call this from your notebook or pipeline script."""
    table_name = f"dataset__{name}"
    with sqlite3.connect(DB_PATH) as con:
        df.to_sql(table_name, con, if_exists="replace", index=False)
        con.execute("""
            CREATE TABLE IF NOT EXISTS _registry (
                name TEXT PRIMARY KEY,
                type TEXT,
                reference TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute(
            "INSERT OR REPLACE INTO _registry (name, type, reference) VALUES (?, ?, ?)",
            (name, "dataset", table_name)
        )

def load_dataset(name: str) -> pd.DataFrame:
    """Call this from Flask or Notebook."""
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT reference FROM _registry WHERE name=? AND type='dataset'", (name,)
        ).fetchone()
        if not row:
            raise KeyError(f"Dataset '{name}' not found")
        return pd.read_sql(f'SELECT * FROM "{row[0]}"', con)

def register_model(name: str, metadata: dict):
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS _registry (
                name TEXT PRIMARY KEY,
                type TEXT,
                reference TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.execute(
            "INSERT OR REPLACE INTO _registry (name, type, reference) VALUES (?, ?, ?)",
            (name, "model", json.dumps(metadata))
        )
