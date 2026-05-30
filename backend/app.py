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

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection

def display_value(value):
    if value is None or pd.isna(value):
        return "-"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4g}"
    return str(value)

@app.route("/api/registry")
def api_registry():
    with get_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS _registry (
                name TEXT PRIMARY KEY, type TEXT, reference TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        rows = con.execute("SELECT name, type, reference, created_at FROM _registry ORDER BY created_at DESC").fetchall()
    
    result = []
    for r in rows:
        d = dict(r)
        if d["type"] == "model":
            d["metadata"] = json.loads(d["reference"])
        result.append(d)
    return jsonify(result)

@app.route("/api/data/<name>")
def api_data(name):
    page = request.args.get("page", default=1, type=int)
    per_page = request.args.get("per_page", default=10, type=int)

    page = max(page, 1)
    per_page = min(max(per_page, 1), 100)

    offset = (page - 1) * per_page

    with get_connection() as con:
        # Check if dataset exists
        row = con.execute("SELECT reference FROM _registry WHERE name=? AND type='dataset'", (name,)).fetchone()
        if not row:
            abort(404, description=f"Dataset '{name}' not found")
        
        table_name = row["reference"]
        columns_info = con.execute(f'PRAGMA table_info("{table_name}")').fetchall()
        columns = [c["name"] for c in columns_info]
        
        total = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        rows = con.execute(
            f'SELECT * FROM "{table_name}" LIMIT ? OFFSET ?',
            (per_page, offset)
        ).fetchall()

    patients = [
        {column: display_value(r[column]) for column in columns}
        for r in rows
    ]
    
    # We rename 'patients' to 'rows' to match generic needs, or keep 'patients' for frontend compat? 
    # The user asked to render in ui, so let's use 'rows' and change frontend.
    return jsonify({
        "name": name,
        "columns": columns,
        "rows": patients,
        "page": page,
        "per_page": per_page,
        "total": total,
    })

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

if __name__ == "__main__":
    app.run(debug=True)

