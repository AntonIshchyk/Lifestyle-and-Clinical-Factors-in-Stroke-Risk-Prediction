from pathlib import Path
from datetime import datetime, timezone
import threading
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

try:
    from .registry import _connect, get_model_path
except ImportError:
    from registry import _connect, get_model_path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
PREDICTION_LOG_PATH = BASE_DIR / "prediction_log.txt"

_model_cache: dict[str, object] = {}
_model_cache_lock = threading.Lock()
_prediction_log_lock = threading.Lock()

def _load_model(model_id: str):
    if model_id in _model_cache:
        return _model_cache[model_id]

    with _model_cache_lock:
        if model_id not in _model_cache:
            _model_cache[model_id] = joblib.load(get_model_path(model_id))
        return _model_cache[model_id]

def _display(value):
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
    except (TypeError, ValueError):
        pass
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else f"{value:.4g}"
    return str(value)


def _append_prediction_log(entry: dict):
    PREDICTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entry, indent=2, sort_keys=True, ensure_ascii=False)

    with _prediction_log_lock:
        with PREDICTION_LOG_PATH.open("a", encoding="utf-8") as handle:
            if handle.tell() > 0:
                handle.write("\n")
            handle.write(payload)
            handle.write("\n---\n")

@app.route("/api/registry")
def api_registry():
    with _connect() as con:
        rows = con.execute(
            "SELECT id, type, label, reference, created_at FROM _registry ORDER BY type, created_at"
        ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.route("/api/data/<name>")
def api_data(name: str):
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = min(max(request.args.get("per_page", 10, type=int), 1), 100)
    offset = (page - 1) * per_page

    with _connect() as con:
        row = con.execute(
            "SELECT reference FROM _registry WHERE id = ? AND type = 'dataset'",
            (name,),
        ).fetchone()
        if not row:
            abort(404, description=f"Dataset '{name}' not found")

        table_name = row["reference"]
        columns = [c["name"] for c in con.execute(f'PRAGMA table_info("{table_name}")').fetchall()]
        total = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
        rows = con.execute(
            f'SELECT * FROM "{table_name}" LIMIT ? OFFSET ?',
            (per_page, offset),
        ).fetchall()

    return jsonify({
        "name": name,
        "columns": columns,
        "rows": [{column: _display(row[column]) for column in columns} for row in rows],
        "page": page,
        "per_page": per_page,
        "total": total,
    })

@app.route("/api/models")
def api_models():
    with _connect() as con:
        rows = con.execute(
            "SELECT model_id, algorithm, feature_set, metrics FROM _model_results"
        ).fetchall()

    models = []
    for row in rows:
        metrics = json.loads(row["metrics"])
        models.append({
            "id": row["model_id"],
            "algorithm": row["algorithm"],
            "featureSet": row["feature_set"],
            **metrics,
        })
    return jsonify(models)

@app.route("/api/models/<model_id>")
def api_model_detail(model_id: str):
    with _connect() as con:
        row = con.execute(
            "SELECT * FROM _model_results WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    if not row:
        abort(404, description=f"Model '{model_id}' not found")

    metrics = json.loads(row["metrics"])
    return jsonify({
        "id": model_id,
        "algorithm": row["algorithm"],
        "featureSet": row["feature_set"],
        "auc": metrics["auc"],
        "classificationReport": json.loads(row["classification_report"]),
        "confusionMatrix": json.loads(row["confusion_matrix"]),
        "featureImportances": json.loads(row["feature_importances"]),
        "rocCurve": json.loads(row["roc_curve"]),
        "featureColumns": json.loads(row["feature_columns"]),
    })

@app.route("/api/models/<model_id>/predict", methods=["POST"])
def api_predict(model_id: str):
    body = request.get_json(force=True) or {}
    features = body.get("features", {})
    if not features:
        abort(400, description="'features' dict required in request body")

    with _connect() as con:
        row = con.execute(
            "SELECT feature_columns FROM _model_results WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    if not row:
        abort(404, description=f"Model '{model_id}' not found")

    feature_columns = json.loads(row["feature_columns"])

    try:
        X = pd.DataFrame(
            [[features.get(column, np.nan) for column in feature_columns]],
            columns=feature_columns,
        ).apply(pd.to_numeric, errors='coerce')
        classifier = _load_model(model_id)
        prediction = int(classifier.predict(X)[0])
        probability = float(classifier.predict_proba(X)[0][1])
    except Exception as exc:
        abort(500, description=str(exc))

    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "label": "Stroke" if prediction == 1 else "No Stroke",
    })


@app.route("/api/predictions/log", methods=["POST"])
def api_prediction_log():
    body = request.get_json(force=True) or {}
    model_id = body.get("modelId")
    if not model_id:
        abort(400, description="'modelId' required in request body")

    with _connect() as con:
        row = con.execute(
            "SELECT algorithm, feature_set, metrics, feature_columns FROM _model_results WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    if not row:
        abort(404, description=f"Model '{model_id}' not found")

    metrics = json.loads(row["metrics"])
    feature_columns = json.loads(row["feature_columns"])
    baseline_features = body.get("baselineFeatures") or {}
    scenario_features = body.get("scenarioFeatures") or body.get("features") or {}
    if not isinstance(baseline_features, dict):
        abort(400, description="'baselineFeatures' must be an object in request body")
    if not isinstance(scenario_features, dict):
        abort(400, description="'scenarioFeatures' must be an object in request body")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "model": {
            "id": model_id,
            "algorithm": row["algorithm"],
            "featureSet": row["feature_set"],
            "metrics": {
                "auc": metrics.get("auc"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1Score": metrics.get("f1-score") or metrics.get("f1_score"),
            },
        },
        "patient": body.get("patient") or {},
        "selectedFeatures": body.get("selectedFeatures") or [],
        "baseline": body.get("baseline") or {},
        "scenario": body.get("scenario") or {},
        "changedFeatures": body.get("changedFeatures") or [],
        "baselineFeatures": {column: baseline_features.get(column) for column in feature_columns},
        "scenarioFeatures": {column: scenario_features.get(column) for column in feature_columns},
    }

    _append_prediction_log(log_entry)
    return jsonify({"ok": True})

@app.route("/assets/<path:filename>")
def frontend_assets(filename: str):
    if not FRONTEND_DIST.exists():
        abort(404)
    return send_from_directory(FRONTEND_DIST / "assets", filename)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def react_app(path: str):
    if path.startswith("api/"):
        abort(404)
    if not FRONTEND_DIST.exists():
        raise FileNotFoundError("Run 'cd frontend && npm run build' first.")
    return send_from_directory(FRONTEND_DIST, "index.html")

if __name__ == "__main__":
    app.run(debug=True)
