from pathlib import Path
import json
import os
import sqlite3

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
NOTEBOOK_PATH = PROJECT_ROOT / "ai_module" / "main.ipynb"
DB_PATH = BASE_DIR / "healthcare.db"
PATIENTS_TABLE = "patients"
PATIENT_ID_COLUMN = "patient_id"


def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def table_exists(connection):
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (PATIENTS_TABLE,),
    ).fetchone()
    return row is not None


def get_columns(connection):
    rows = connection.execute(f'PRAGMA table_info("{PATIENTS_TABLE}")').fetchall()
    return [row["name"] for row in rows]


def load_clean_df_from_notebook():
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(f"Missing notebook: {NOTEBOOK_PATH}")

    namespace = {"__name__": "__notebook__", "__file__": str(NOTEBOOK_PATH)}
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    current_directory = os.getcwd()

    try:
        os.chdir(NOTEBOOK_PATH.parent)

        for cell in notebook["cells"]:
            if cell.get("cell_type") != "code":
                continue

            source = "".join(cell.get("source", []))
            if "find_high_correlation_pairs" in source:
                continue

            exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
            if "clean_df" in namespace:
                clean_df = namespace["clean_df"]
                if not isinstance(clean_df, pd.DataFrame):
                    raise TypeError("main.ipynb defines clean_df, but it is not a pandas DataFrame.")
                return clean_df.copy()
    finally:
        os.chdir(current_directory)

    raise RuntimeError("main.ipynb must define a pandas DataFrame named clean_df.")


def prepare_patients_dataframe():
    dataframe = load_clean_df_from_notebook().reset_index(drop=True)
    dataframe.insert(0, PATIENT_ID_COLUMN, range(1, len(dataframe) + 1))
    return dataframe


def rebuild_database():
    dataframe = prepare_patients_dataframe()

    with get_connection() as connection:
        connection.execute(f'DROP TABLE IF EXISTS "{PATIENTS_TABLE}"')
        dataframe.to_sql(PATIENTS_TABLE, connection, if_exists="replace", index=False)
        connection.execute(
            f'CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON "{PATIENTS_TABLE}" ("{PATIENT_ID_COLUMN}")'
        )
        connection.commit()


def ensure_database():
    with get_connection() as connection:
        if table_exists(connection):
            return
    rebuild_database()


def display_value(value):
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4g}"
    return str(value)


def get_patients(page, per_page):
    offset = (page - 1) * per_page

    with get_connection() as connection:
        columns = get_columns(connection)
        total = connection.execute(
            f'SELECT COUNT(*) FROM "{PATIENTS_TABLE}"'
        ).fetchone()[0]
        rows = connection.execute(
            f'''
            SELECT *
            FROM "{PATIENTS_TABLE}"
            ORDER BY "{PATIENT_ID_COLUMN}"
            LIMIT ? OFFSET ?
            ''',
            [per_page, offset],
        ).fetchall()

    patients = [
        {column: display_value(row[column]) for column in columns}
        for row in rows
    ]
    return columns, patients, total


@app.route("/api/patients")
def api_patients():
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=10, type=int)

    page = max(page, 1)
    per_page = min(max(per_page, 1), 100)

    columns, patients, total = get_patients(page, per_page)
    return jsonify(
        {
            "columns": columns,
            "patients": patients,
            "page": page,
            "per_page": per_page,
            "total": total,
        }
    )


@app.route("/assets/<path:filename>")
def frontend_assets(filename):
    if not FRONTEND_DIST.exists():
        abort(404)
    return send_from_directory(FRONTEND_DIST / "assets", filename)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def react_app(path):
    if path.startswith("api/"):
        abort(404)
    if not FRONTEND_DIST.exists():
        raise FileNotFoundError(
            "Missing frontend build output. Run 'cd frontend && npm run build' first."
        )
    return send_from_directory(FRONTEND_DIST, "index.html")


@app.cli.command("rebuild-db")
def rebuild_db_command():
    rebuild_database()
    print(f"Rebuilt {DB_PATH}")


if __name__ == "__main__":
    ensure_database()
    app.run(debug=True)
