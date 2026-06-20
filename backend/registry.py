import json
import re
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "healthcare.db"
DATASETS_DIR = Path(__file__).resolve().parent / "datasets"
DATASETS_DIR.mkdir(exist_ok=True)

def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def _ensure_schema(con: sqlite3.Connection):
    con.executescript("""
        CREATE TABLE IF NOT EXISTS _registry (
            id          TEXT PRIMARY KEY,
            type        TEXT NOT NULL,
            label       TEXT NOT NULL,
            reference   TEXT NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS _model_results (
            model_id              TEXT PRIMARY KEY,
            algorithm             TEXT NOT NULL,
            feature_set           TEXT NOT NULL,
            uncertainty_variant   TEXT NOT NULL,
            balancing_method      TEXT NOT NULL DEFAULT 'random_oversampling',
            metrics               TEXT NOT NULL,
            classification_report TEXT NOT NULL,
            confusion_matrix      TEXT NOT NULL,
            feature_importances   TEXT NOT NULL,
            roc_curve             TEXT NOT NULL,
            feature_columns       TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS _automatic_training_runs (
            job_id        TEXT PRIMARY KEY,
            base_model_id TEXT NOT NULL,
            status        TEXT NOT NULL,
            message       TEXT NOT NULL,
            created_at    TEXT NOT NULL,
            started_at    TEXT,
            finished_at   TEXT,
            request_json  TEXT NOT NULL,
            result_json   TEXT,
            error         TEXT
        );
    """)

def register_dataset(name: str, df: pd.DataFrame, label: str | None = None):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError(
            f"Dataset name '{name}' contains invalid characters. Use only letters, digits, underscores, and hyphens."
        )

    parquet_path = DATASETS_DIR / f"{name}.parquet"
    df.to_parquet(parquet_path, index=False, compression="zstd")
    relative = parquet_path.relative_to(Path(__file__).resolve().parent)

    with _connect() as con:
        _ensure_schema(con)
        con.execute(
            "INSERT OR REPLACE INTO _registry (id, type, label, reference) VALUES (?, 'dataset', ?, ?)",
            (name, label or name, str(relative)),
        )

def load_dataset(name: str) -> pd.DataFrame:
    with _connect() as con:
        row = con.execute(
            "SELECT reference FROM _registry WHERE id = ? AND type = 'dataset'", (name,)
        ).fetchone()
        if not row:
            raise KeyError(f"Dataset '{name}' not found")
        path = Path(__file__).resolve().parent / row["reference"]
        return pd.read_parquet(path)

def clear_registered_models():
    with _connect() as con:
        _ensure_schema(con)
        con.execute("DELETE FROM _model_results")
        con.execute("DELETE FROM _registry WHERE type = 'model'")

def register_model(
    model_id: str,
    algorithm: str,
    feature_set: str,
    uncertainty_variant: str,
    model_path: str,
    metrics: dict,
    classification_report: dict,
    confusion_matrix: dict,
    feature_importances: list,
    roc_curve: dict,
    feature_columns: list,
    balancing_method: str = "random_oversampling",
):
    label = (
        f"{algorithm.replace('_', ' ').title()} / {feature_set.title()} / "
        f"{uncertainty_variant.replace('_', ' ').title()} / "
        f"{balancing_method.replace('_', ' ').title()}"
    )
    with _connect() as con:
        _ensure_schema(con)
        con.execute(
            "INSERT OR REPLACE INTO _registry (id, type, label, reference) VALUES (?, 'model', ?, ?)",
            (model_id, label, model_path),
        )
        con.execute(
            """
            INSERT OR REPLACE INTO _model_results
                (model_id, algorithm, feature_set, uncertainty_variant, balancing_method,
                 metrics, classification_report,
                 confusion_matrix, feature_importances, roc_curve, feature_columns)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                model_id,
                algorithm,
                feature_set,
                uncertainty_variant,
                balancing_method,
                json.dumps(metrics),
                json.dumps(classification_report),
                json.dumps(confusion_matrix),
                json.dumps(feature_importances),
                json.dumps(roc_curve),
                json.dumps(feature_columns),
            ),
        )

def get_model_path(model_id: str) -> str:
    with _connect() as con:
        row = con.execute(
            "SELECT reference FROM _registry WHERE id = ? AND type = 'model'", (model_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Model '{model_id}' not found")
        return row["reference"]
