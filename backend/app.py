from pathlib import Path
from datetime import datetime, timezone
import threading
import json
import uuid
import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

try:
    from .registry import _connect, _ensure_schema, get_model_path
    from .training import ALGORITHMS, BALANCING_STRATEGIES, parse_dataset_id, train_model
except ImportError:
    from registry import _connect, _ensure_schema, get_model_path
    from training import ALGORITHMS, BALANCING_STRATEGIES, parse_dataset_id, train_model

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
PREDICTION_LOG_PATH = BASE_DIR / "prediction_log.txt"
AI_MODULE_DIR = PROJECT_ROOT / "ai_module"

_model_cache: dict[str, object] = {}
_training_jobs: dict[str, dict[str, object]] = {}
_model_cache_lock = threading.Lock()
_prediction_log_lock = threading.Lock()
_training_jobs_lock = threading.Lock()

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


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _public_job(job: dict[str, object]):
    return {key: value for key, value in job.items() if key != "thread"}


def _set_job(job_id: str, **updates):
    with _training_jobs_lock:
        job = _training_jobs[job_id]
        job.update(updates)
        return _public_job(job)


def _request_list(body: dict[str, object], plural_key: str, singular_key: str):
    value = body.get(plural_key, body.get(singular_key, []))
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value:
        return [str(value)]
    return []


def _model_id(algorithm: str, dataset_id: str, balancing_method: str):
    return f"{algorithm}_{balancing_method}_{dataset_id}"


def _is_tuned_model(model_id: str):
    return "__tuned_" in model_id


def _delete_model_file(model_path: str | None):
    if not model_path:
        return False

    resolved_path = Path(model_path).resolve()
    try:
        resolved_path.relative_to(AI_MODULE_DIR.resolve())
    except ValueError:
        abort(500, description="Refusing to delete a model file outside ai_module")

    if not resolved_path.exists():
        return False
    if not resolved_path.is_file():
        abort(500, description="Model path is not a file")

    resolved_path.unlink()
    return True


def _fine_tune_suffix(payload: dict[str, object]):
    tuning_payload = {
        "removedFeatures": sorted(payload.get("removedFeatures", [])),
        "hyperparameters": payload.get("hyperparameters", {}),
        "targetRatio": payload.get("targetRatio", 1.0),
        "classificationThreshold": payload.get("classificationThreshold", 0.5),
    }
    serialized = json.dumps(tuning_payload, sort_keys=True, separators=(",", ":"))
    return f"tuned_{uuid.uuid5(uuid.NAMESPACE_URL, serialized).hex[:10]}"


def _dataset_options():
    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute(
            "SELECT id, label FROM _registry WHERE type = 'dataset' ORDER BY id"
        ).fetchall()

    datasets = []
    for row in rows:
        try:
            feature_set, uncertainty_variant = parse_dataset_id(row["id"])
        except ValueError:
            feature_set, uncertainty_variant = "", ""
        datasets.append({
            "id": row["id"],
            "label": row["label"],
            "featureSet": feature_set,
            "uncertaintyVariant": uncertainty_variant,
        })
    return datasets


def _expected_model_specs(datasets: list[dict[str, str]]):
    return [
        {
            "id": _model_id(algorithm, dataset["id"], balancing_method),
            "algorithm": algorithm,
            "datasetId": dataset["id"],
            "featureSet": dataset["featureSet"],
            "uncertaintyVariant": dataset["uncertaintyVariant"],
            "balancingMethod": balancing_method,
        }
        for dataset in datasets
        for balancing_method in BALANCING_STRATEGIES.keys()
        for algorithm in ALGORITHMS.keys()
    ]


def _run_training_job(job_id: str, payload: dict[str, object]):
    _set_job(job_id, status="running", startedAt=_now_iso(), message="Training started.")
    try:
        results = []
        combinations = payload.get("models") or [
            {
                "algorithm": algorithm,
                "datasetId": dataset_id,
                "balancingMethod": balancing_method,
            }
            for dataset_id in payload["datasetIds"]
            for balancing_method in payload["balancingMethods"]
            for algorithm in payload["algorithms"]
        ]
        total = len(combinations)
        for index, spec in enumerate(combinations, start=1):
            algorithm = spec["algorithm"]
            dataset_id = spec["datasetId"]
            balancing_method = spec["balancingMethod"]
            _set_job(
                job_id,
                message=(
                    f"Training {index}/{total}: "
                    f"{algorithm} / {dataset_id} / {balancing_method}."
                ),
            )
            result = train_model(
                algorithm=algorithm,
                dataset_id=dataset_id,
                balancing_method=balancing_method,
                target_ratio=payload["targetRatio"],
                classification_threshold=payload["classificationThreshold"],
                force_retrain=payload["forceRetrain"],
                use_gpu=payload["useGpu"],
                removed_features=payload.get("removedFeatures", []),
                hyperparameters=payload.get("hyperparameters", {}),
                model_id_suffix=payload.get("modelIdSuffix"),
            )
            with _model_cache_lock:
                _model_cache.pop(result["modelId"], None)
            results.append(result)

        reused = sum(1 for result in results if result["reusedExistingModel"])
        result_payload = {
            "models": results,
            "total": total,
            "trained": total - reused,
            "reused": reused,
        }
        _set_job(
            job_id,
            status="succeeded",
            finishedAt=_now_iso(),
            message=(
                "Existing models registered."
                if reused == total
                else "Training batch completed."
            ),
            result=result_payload,
        )
    except Exception as exc:
        _set_job(
            job_id,
            status="failed",
            finishedAt=_now_iso(),
            message=str(exc),
            error=str(exc),
        )

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


@app.route("/api/training/options")
def api_training_options():
    datasets = _dataset_options()

    return jsonify({
        "algorithms": [
            {"id": algorithm_id, "label": label}
            for algorithm_id, label in ALGORITHMS.items()
        ],
        "datasets": datasets,
        "balancingMethods": [
            {"id": balancing_id}
            for balancing_id in BALANCING_STRATEGIES.keys()
        ],
        "defaults": {
            "algorithms": ["xgboost"],
            "datasetIds": [datasets[0]["id"]] if datasets else [],
            "balancingMethods": ["random_oversampling"],
            "targetRatio": 1.0,
            "forceRetrain": False,
            "useGpu": True,
        },
    })


@app.route("/api/training/coverage")
def api_training_coverage():
    datasets = _dataset_options()
    expected = _expected_model_specs(datasets)
    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute(
            "SELECT model_id FROM _model_results"
        ).fetchall()
    registered_model_ids = {row["model_id"] for row in rows}
    available = [
        spec for spec in expected if spec["id"] in registered_model_ids
    ]
    missing = [
        spec for spec in expected if spec["id"] not in registered_model_ids
    ]

    missing_by_dataset = {}
    missing_by_algorithm = {}
    missing_by_balancing_method = {}
    for spec in missing:
        missing_by_dataset[spec["datasetId"]] = missing_by_dataset.get(spec["datasetId"], 0) + 1
        missing_by_algorithm[spec["algorithm"]] = missing_by_algorithm.get(spec["algorithm"], 0) + 1
        missing_by_balancing_method[spec["balancingMethod"]] = missing_by_balancing_method.get(spec["balancingMethod"], 0) + 1

    return jsonify({
        "totalExpected": len(expected),
        "availableCount": len(available),
        "missingCount": len(missing),
        "available": available,
        "missing": missing,
        "missingByDataset": missing_by_dataset,
        "missingByAlgorithm": missing_by_algorithm,
        "missingByBalancingMethod": missing_by_balancing_method,
    })


@app.route("/api/training/jobs", methods=["POST"])
def api_start_training_job():
    body = request.get_json(force=True) or {}
    algorithms = _request_list(body, "algorithms", "algorithm")
    dataset_ids = _request_list(body, "datasetIds", "datasetId")
    balancing_methods = _request_list(body, "balancingMethods", "balancingMethod")
    model_specs = body.get("models", [])
    target_ratio = body.get("targetRatio", 1.0)
    classification_threshold = body.get("classificationThreshold", 0.5)
    force_retrain = bool(body.get("forceRetrain", False))
    use_gpu = bool(body.get("useGpu", True))
    removed_features = _request_list(body, "removedFeatures", "removedFeature")
    hyperparameters = body.get("hyperparameters", {})
    model_id_suffix = body.get("modelIdSuffix")

    if hyperparameters is None:
        hyperparameters = {}
    if not isinstance(hyperparameters, dict):
        abort(400, description="'hyperparameters' must be an object")
    if model_id_suffix is not None:
        model_id_suffix = str(model_id_suffix).strip()
        if not model_id_suffix:
            model_id_suffix = None
        elif not all(char.isalnum() or char in {"_", "-"} for char in model_id_suffix):
            abort(400, description="'modelIdSuffix' may only contain letters, digits, underscores, or hyphens")

    if model_specs:
        if not isinstance(model_specs, list):
            abort(400, description="'models' must be an array of model specs")
        normalized_specs = []
        for spec in model_specs:
            if not isinstance(spec, dict):
                abort(400, description="'models' must contain objects")
            normalized_specs.append({
                "algorithm": str(spec.get("algorithm", "")),
                "datasetId": str(spec.get("datasetId", "")),
                "balancingMethod": str(spec.get("balancingMethod", "")),
            })
        algorithms = sorted({spec["algorithm"] for spec in normalized_specs})
        dataset_ids = sorted({spec["datasetId"] for spec in normalized_specs})
        balancing_methods = sorted({spec["balancingMethod"] for spec in normalized_specs})
    else:
        normalized_specs = []

    if not model_specs and not algorithms:
        abort(400, description="'algorithms' must include at least one supported algorithm id")
    if not model_specs and not dataset_ids:
        abort(400, description="'datasetIds' must include at least one registered dataset id")
    if not model_specs and not balancing_methods:
        abort(400, description="'balancingMethods' must include at least one supported balancing method id")
    invalid_algorithms = [algorithm for algorithm in algorithms if algorithm not in ALGORITHMS]
    if invalid_algorithms:
        abort(400, description=f"Unsupported algorithm id(s): {', '.join(invalid_algorithms)}")
    invalid_balancing_methods = [
        balancing_method
        for balancing_method in balancing_methods
        if balancing_method not in BALANCING_STRATEGIES
    ]
    if invalid_balancing_methods:
        abort(400, description=f"Unsupported balancing method id(s): {', '.join(invalid_balancing_methods)}")
    try:
        target_ratio = float(target_ratio)
    except (TypeError, ValueError):
        abort(400, description="'targetRatio' must be a number")
    try:
        classification_threshold = float(classification_threshold)
    except (TypeError, ValueError):
        abort(400, description="'classificationThreshold' must be a number")
    if "weighted" in balancing_methods:
        target_ratio = 1.0
    if not 0 < target_ratio <= 1:
        abort(400, description="'targetRatio' must be in the interval (0, 1]")
    if not 0 < classification_threshold < 1:
        abort(400, description="'classificationThreshold' must be in the interval (0, 1)")
    if removed_features or hyperparameters or classification_threshold != 0.5:
        force_retrain = True
        model_id_suffix = model_id_suffix or _fine_tune_suffix({
            "removedFeatures": removed_features,
            "hyperparameters": hyperparameters,
            "targetRatio": target_ratio,
            "classificationThreshold": classification_threshold,
        })

    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute(
            "SELECT id FROM _registry WHERE type = 'dataset'",
        ).fetchall()
    registered_dataset_ids = {row["id"] for row in rows}
    invalid_dataset_ids = [
        dataset_id for dataset_id in dataset_ids if dataset_id not in registered_dataset_ids
    ]
    if invalid_dataset_ids:
        abort(400, description=f"Unknown dataset id(s): {', '.join(invalid_dataset_ids)}")

    payload = {
        "algorithms": algorithms,
        "datasetIds": dataset_ids,
        "balancingMethods": balancing_methods,
        "targetRatio": target_ratio,
        "classificationThreshold": classification_threshold,
        "forceRetrain": force_retrain,
        "useGpu": use_gpu,
    }
    if removed_features:
        payload["removedFeatures"] = removed_features
    if hyperparameters:
        payload["hyperparameters"] = hyperparameters
    if model_id_suffix:
        payload["modelIdSuffix"] = model_id_suffix
    if model_specs:
        payload["models"] = normalized_specs
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "message": "Training queued.",
        "createdAt": _now_iso(),
        "startedAt": None,
        "finishedAt": None,
        "request": payload,
        "result": None,
        "error": None,
    }
    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, payload),
        daemon=True,
    )
    job["thread"] = thread
    with _training_jobs_lock:
        _training_jobs[job_id] = job
    thread.start()

    return jsonify(_public_job(job)), 202


@app.route("/api/training/jobs/<job_id>")
def api_training_job(job_id: str):
    with _training_jobs_lock:
        job = _training_jobs.get(job_id)
        if not job:
            abort(404, description=f"Training job '{job_id}' not found")
        return jsonify(_public_job(job))

@app.route("/api/models")
def api_models():
    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute(
            "SELECT model_id, algorithm, feature_set, uncertainty_variant, balancing_method, metrics FROM _model_results"
        ).fetchall()

    models = []
    for row in rows:
        metrics = json.loads(row["metrics"])
        metrics.setdefault("classificationThreshold", 0.5)
        models.append({
            "id": row["model_id"],
            "algorithm": row["algorithm"],
            "featureSet": row["feature_set"],
            "uncertaintyVariant": row["uncertainty_variant"],
            "balancingMethod": row["balancing_method"],
            "isTuned": _is_tuned_model(row["model_id"]),
            **metrics,
        })
    return jsonify(models)

@app.route("/api/models/<model_id>")
def api_model_detail(model_id: str):
    with _connect() as con:
        _ensure_schema(con)
        row = con.execute(
            "SELECT * FROM _model_results WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    if not row:
        abort(404, description=f"Model '{model_id}' not found")

    metrics = json.loads(row["metrics"])
    metrics.setdefault("classificationThreshold", 0.5)
    return jsonify({
        "id": model_id,
        "algorithm": row["algorithm"],
        "featureSet": row["feature_set"],
        "uncertaintyVariant": row["uncertainty_variant"],
        "balancingMethod": row["balancing_method"],
        "isTuned": _is_tuned_model(model_id),
        "auc": metrics["auc"],
        "classificationReport": json.loads(row["classification_report"]),
        "confusionMatrix": json.loads(row["confusion_matrix"]),
        "featureImportances": json.loads(row["feature_importances"]),
        "rocCurve": json.loads(row["roc_curve"]),
        "featureColumns": json.loads(row["feature_columns"]),
        "classificationThreshold": metrics["classificationThreshold"],
    })

@app.route("/api/models/<model_id>", methods=["DELETE"])
def api_delete_model(model_id: str):
    with _connect() as con:
        _ensure_schema(con)
        row = con.execute(
            """
            SELECT m.model_id, r.reference
            FROM _model_results m
            LEFT JOIN _registry r
                ON r.id = m.model_id AND r.type = 'model'
            WHERE m.model_id = ?
            """,
            (model_id,),
        ).fetchone()
        if not row:
            abort(404, description=f"Model '{model_id}' not found")
        if not _is_tuned_model(row["model_id"]):
            abort(400, description="Only fine-tuned models can be deleted")

        file_deleted = _delete_model_file(row["reference"])
        con.execute("DELETE FROM _model_results WHERE model_id = ?", (model_id,))
        con.execute(
            "DELETE FROM _registry WHERE id = ? AND type = 'model'",
            (model_id,),
        )

    with _model_cache_lock:
        _model_cache.pop(model_id, None)

    return jsonify({
        "ok": True,
        "modelId": model_id,
        "fileDeleted": file_deleted,
    })

@app.route("/api/models/<model_id>/predict", methods=["POST"])
def api_predict(model_id: str):
    body = request.get_json(force=True) or {}
    features = body.get("features", {})
    if not features:
        abort(400, description="'features' dict required in request body")

    with _connect() as con:
        _ensure_schema(con)
        row = con.execute(
            "SELECT feature_columns, metrics FROM _model_results WHERE model_id = ?",
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
        try:
            positive_class_index = list(classifier.classes_).index(1)
        except ValueError as exc:
            raise ValueError("Classifier does not expose class label 1.") from exc
        probability = float(classifier.predict_proba(X)[0][positive_class_index])
        metrics = json.loads(row["metrics"])
        classification_threshold = float(metrics.get("classificationThreshold", 0.5))
        prediction = int(probability >= classification_threshold)
    except Exception as exc:
        abort(500, description=str(exc))

    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "classificationThreshold": classification_threshold,
        "label": "Stroke" if prediction == 1 else "No Stroke",
    })


@app.route("/api/predictions/log", methods=["POST"])
def api_prediction_log():
    body = request.get_json(force=True) or {}
    model_id = body.get("modelId")
    if not model_id:
        abort(400, description="'modelId' required in request body")

    with _connect() as con:
        _ensure_schema(con)
        row = con.execute(
            "SELECT algorithm, feature_set, balancing_method, metrics, feature_columns FROM _model_results WHERE model_id = ?",
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
            "balancingMethod": row["balancing_method"],
            "metrics": {
                "auc": metrics.get("auc"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1Score": metrics.get("f1") or metrics.get("f1-score") or metrics.get("f1_score"),
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
