from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape
import threading
import json
import uuid
import joblib
import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, request, send_file, send_from_directory

try:
    from .registry import _connect, _ensure_schema, get_model_path, load_dataset
    from .training import ALGORITHMS, BALANCING_STRATEGIES, parse_dataset_id, train_model
except ImportError:
    from registry import _connect, _ensure_schema, get_model_path, load_dataset
    from training import ALGORITHMS, BALANCING_STRATEGIES, parse_dataset_id, train_model

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
PREDICTION_LOG_PATH = BASE_DIR / "prediction_log.txt"
AI_MODULE_DIR = PROJECT_ROOT / "ai_module"

_model_cache: dict[str, object] = {}
_training_jobs: dict[str, dict[str, object]] = {}
_shap_cache: dict[str, dict[str, object]] = {}
_model_cache_lock = threading.Lock()
_shap_cache_lock = threading.Lock()
_prediction_log_lock = threading.Lock()
_training_jobs_lock = threading.Lock()
_automatic_runs_lock = threading.Lock()
SHAP_BACKGROUND_ROWS = 64

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


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _json_number(value):
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _positive_class_index(classifier) -> int:
    classes = list(getattr(classifier, "classes_", []))
    if 1 in classes:
        return classes.index(1)
    return 1 if len(classes) > 1 else 0


def _load_shap_background(metadata: dict, feature_columns: list[str], fallback: pd.DataFrame) -> pd.DataFrame:
    dataset_id = f"{metadata['feature_set']}_{metadata['uncertainty_variant']}"

    try:
        df = load_dataset(dataset_id)
        selected_columns = [col for col in feature_columns if col in df.columns]
        if not selected_columns:
            return fallback
        background = df[selected_columns].head(SHAP_BACKGROUND_ROWS).copy()
    except Exception:
        return fallback

    for col in feature_columns:
        if col not in background.columns:
            background[col] = np.nan

    return background[feature_columns]


def _load_shap_explainer(model_id: str, classifier, metadata: dict, feature_columns: list[str], X: pd.DataFrame):
    if model_id in _shap_cache:
        return _shap_cache[model_id]

    with _shap_cache_lock:
        if model_id in _shap_cache:
            return _shap_cache[model_id]

        import shap

        if metadata["algorithm"] == "random_forest":
            explainer = shap.TreeExplainer(classifier)
            background_rows = None
        else:
            background = _load_shap_background(metadata, feature_columns, X)
            explainer = shap.TreeExplainer(
                classifier,
                data=background,
                model_output="probability",
                feature_perturbation="interventional",
            )
            background_rows = len(background)

        _shap_cache[model_id] = {
            "explainer": explainer,
            "backgroundRows": background_rows,
        }
        return _shap_cache[model_id]


def _extract_positive_shap_values(explainer, classifier, X: pd.DataFrame):
    class_index = _positive_class_index(classifier)
    shap_values = explainer.shap_values(X, check_additivity=False)
    expected_value = np.asarray(explainer.expected_value).reshape(-1)

    if isinstance(shap_values, list):
        values = np.asarray(shap_values[class_index])[0]
        base_value = expected_value[class_index] if len(expected_value) > class_index else expected_value[-1]
        return values, base_value

    values_array = np.asarray(shap_values)
    if values_array.ndim == 3:
        if values_array.shape[0] == len(X):
            values = values_array[0, :, class_index]
        else:
            values = values_array[class_index, 0, :]
        base_value = expected_value[class_index] if len(expected_value) > class_index else expected_value[-1]
        return values, base_value

    if values_array.ndim == 2:
        base_value = expected_value[class_index] if len(expected_value) > class_index else expected_value[-1]
        return values_array[0], base_value

    raise ValueError("Unsupported SHAP value shape")


def _build_shap_explanation(model_id: str, classifier, metadata: dict, feature_columns: list[str], X: pd.DataFrame, probability: float):
    cache_entry = _load_shap_explainer(model_id, classifier, metadata, feature_columns, X)
    values, base_value = _extract_positive_shap_values(cache_entry["explainer"], classifier, X)

    contributions = []
    row = X.iloc[0]
    for feature, contribution in zip(feature_columns, values):
        contribution_value = _json_number(contribution) or 0.0
        contributions.append({
            "feature": feature,
            "value": _json_number(row[feature]),
            "contribution": contribution_value,
            "absContribution": abs(contribution_value),
            "direction": "increases" if contribution_value >= 0 else "decreases",
        })

    contributions.sort(key=lambda item: item["absContribution"], reverse=True)
    base = _json_number(base_value) or 0.0

    return {
        "modelOutput": "probability",
        "classLabel": "Stroke",
        "baseValue": base,
        "outputValue": probability,
        "sumValue": base + sum(item["contribution"] for item in contributions),
        "backgroundRows": cache_entry["backgroundRows"],
        "features": contributions,
    }


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


def _json_payload(value):
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _load_json_payload(value, fallback):
    if value is None:
        return fallback
    return json.loads(value)


def _stroke_risk_score(metrics: dict[str, object]):
    return (
        float(metrics.get("auc", 0)) * 0.35
        + float(metrics.get("f1", 0)) * 0.3
        + float(metrics.get("recall", 0)) * 0.25
        + float(metrics.get("precision", 0)) * 0.1
    )


def _xlsx_column_name(index: int):
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _xlsx_cell(value, row_number: int, column_number: int):
    reference = f"{_xlsx_column_name(column_number)}{row_number}"
    if value is None:
        return f'<c r="{reference}"/>'
    if isinstance(value, bool):
        return f'<c r="{reference}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if pd.isna(value):
            return f'<c r="{reference}"/>'
        return f'<c r="{reference}"><v>{value}</v></c>'
    return f'<c r="{reference}" t="inlineStr"><is><t>{escape(str(value))}</t></is></c>'


def _xlsx_sheet(rows: list[list[object]]):
    sheet_rows = []
    for row_number, row in enumerate(rows, start=1):
        cells = "".join(
            _xlsx_cell(value, row_number, column_number)
            for column_number, value in enumerate(row, start=1)
        )
        sheet_rows.append(f'<row r="{row_number}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheetData>{"".join(sheet_rows)}</sheetData>'
        '</worksheet>'
    )


def _xlsx_workbook(sheets: list[tuple[str, list[list[object]]]]):
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            + "".join(
                f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
                'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                for index in range(1, len(sheets) + 1)
            )
            + "</Types>",
        )
        archive.writestr(
            "_rels/.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            "</Relationships>",
        )
        archive.writestr(
            "xl/workbook.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            "<sheets>"
            + "".join(
                f'<sheet name="{escape(name[:31])}" sheetId="{index}" r:id="rId{index}"/>'
                for index, (name, _) in enumerate(sheets, start=1)
            )
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{index}" '
                'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
                f'Target="worksheets/sheet{index}.xml"/>'
                for index in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(f"xl/worksheets/sheet{index}.xml", _xlsx_sheet(rows))
    output.seek(0)
    return output


def _safe_export_filename(value: str):
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _model_export_details(model_ids: list[str]):
    if not model_ids:
        return {}
    placeholders = ",".join("?" for _ in model_ids)
    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute(
            f"""
            SELECT model_id, confusion_matrix, classification_report
            FROM _model_results
            WHERE model_id IN ({placeholders})
            """,
            model_ids,
        ).fetchall()
    return {
        row["model_id"]: {
            "confusionMatrix": json.loads(row["confusion_matrix"]),
            "classificationReport": json.loads(row["classification_report"]),
        }
        for row in rows
    }


def _automatic_run_ranked_models(job: dict[str, object]):
    result = job.get("result") or {}
    models = result.get("models", []) if isinstance(result, dict) else []
    return sorted(
        models,
        key=lambda model: (
            _stroke_risk_score(model.get("metrics", {})),
            float(model.get("metrics", {}).get("auc", 0)),
            float(model.get("metrics", {}).get("recall", 0)),
            float(model.get("metrics", {}).get("f1", 0)),
        ),
        reverse=True,
    )


def _automatic_run_summary_rows(job: dict[str, object]):
    result = job.get("result") or {}
    return [
        ["Field", "Value"],
        ["Run ID", job["id"]],
        ["Status", job["status"]],
        ["Message", job["message"]],
        ["Created at", job["createdAt"]],
        ["Started at", job.get("startedAt")],
        ["Finished at", job.get("finishedAt")],
        ["Base model ID", (job.get("request") or {}).get("baseModelId")],
        ["Total combinations", result.get("total") if isinstance(result, dict) else None],
        ["Trained models", result.get("trained") if isinstance(result, dict) else None],
        ["Reused models", result.get("reused") if isinstance(result, dict) else None],
        ["Error", job.get("error")],
    ]


def _automatic_run_result_rows(
    job: dict[str, object],
    *,
    hyperparameter_keys: list[str] | None = None,
    details_by_model: dict[str, object] | None = None,
    include_run_columns: bool = False,
):
    ranked_models = _automatic_run_ranked_models(job)
    if hyperparameter_keys is None:
        hyperparameter_keys = sorted({
            key
            for model in ranked_models
            for key in (model.get("hyperparameters") or {}).keys()
        })
    if details_by_model is None:
        details_by_model = _model_export_details([
            str(model.get("modelId"))
            for model in ranked_models
            if model.get("modelId")
        ])

    run_header = []
    if include_run_columns:
        run_header = ["Run ID", "Base model ID", "Run created at"]

    result_header = [
        *run_header,
        "Rank",
        "Score",
        "Model ID",
        "Algorithm",
        "Dataset ID",
        "Feature set",
        "Uncertainty variant",
        "Balancing method",
        "Target ratio",
        "Classification threshold",
        "AUC-ROC",
        "Accuracy",
        "F1",
        "Precision",
        "Recall",
        "True negatives",
        "False positives",
        "False negatives",
        "True positives",
        *[f"Parameter: {key}" for key in hyperparameter_keys],
    ]
    result_rows = [result_header]
    for rank, model in enumerate(ranked_models, start=1):
        metrics = model.get("metrics") or {}
        details = details_by_model.get(model.get("modelId"), {})
        confusion_matrix = model.get("confusionMatrix") or details.get("confusionMatrix") or {}
        hyperparameters = model.get("hyperparameters") or {}
        run_values = []
        if include_run_columns:
            run_values = [
                job.get("id"),
                (job.get("request") or {}).get("baseModelId"),
                job.get("createdAt"),
            ]
        result_rows.append([
            *run_values,
            rank,
            _stroke_risk_score(metrics),
            model.get("modelId"),
            model.get("algorithm"),
            model.get("datasetId"),
            model.get("featureSet"),
            model.get("uncertaintyVariant"),
            model.get("balancingMethod"),
            model.get("targetRatio"),
            model.get("classificationThreshold"),
            metrics.get("auc"),
            metrics.get("accuracy"),
            metrics.get("f1"),
            metrics.get("precision"),
            metrics.get("recall"),
            confusion_matrix.get("tn"),
            confusion_matrix.get("fp"),
            confusion_matrix.get("fn"),
            confusion_matrix.get("tp"),
            *[hyperparameters.get(key) for key in hyperparameter_keys],
        ])
    return result_rows


def _automatic_run_export_workbook(job: dict[str, object]):
    return _xlsx_workbook([
        ("Run summary", _automatic_run_summary_rows(job)),
        ("Model results", _automatic_run_result_rows(job)),
    ])


def _automatic_runs_export_workbook(jobs: list[dict[str, object]]):
    all_ranked_models = [
        model
        for job in jobs
        for model in _automatic_run_ranked_models(job)
    ]
    hyperparameter_keys = sorted({
        key
        for model in all_ranked_models
        for key in (model.get("hyperparameters") or {}).keys()
    })
    details_by_model = _model_export_details([
        str(model.get("modelId"))
        for model in all_ranked_models
        if model.get("modelId")
    ])

    overview_rows = [[
        "Run ID",
        "Base model ID",
        "Status",
        "Created at",
        "Started at",
        "Finished at",
        "Total combinations",
        "Trained models",
        "Reused models",
        "Best model ID",
        "Best score",
        "Message",
        "Error",
    ]]
    combined_result_rows: list[list[object]] | None = None
    for job in jobs:
        result = job.get("result") or {}
        ranked_models = _automatic_run_ranked_models(job)
        best = ranked_models[0] if ranked_models else None
        overview_rows.append([
            job["id"],
            (job.get("request") or {}).get("baseModelId"),
            job["status"],
            job["createdAt"],
            job.get("startedAt"),
            job.get("finishedAt"),
            result.get("total") if isinstance(result, dict) else None,
            result.get("trained") if isinstance(result, dict) else None,
            result.get("reused") if isinstance(result, dict) else None,
            best.get("modelId") if best else None,
            _stroke_risk_score(best.get("metrics", {})) if best else None,
            job["message"],
            job.get("error"),
        ])

        result_rows = _automatic_run_result_rows(
            job,
            hyperparameter_keys=hyperparameter_keys,
            details_by_model=details_by_model,
            include_run_columns=True,
        )
        if combined_result_rows is None:
            combined_result_rows = result_rows
        else:
            combined_result_rows.extend(result_rows[1:])

    return _xlsx_workbook([
        ("Selected runs", overview_rows),
        ("Model results", combined_result_rows or _automatic_run_result_rows({"result": {"models": []}})),
    ])


def _load_automatic_training_run(job_id: str):
    with _training_jobs_lock:
        job = _training_jobs.get(job_id)
        if job and (job.get("request") or {}).get("automatic"):
            return _public_job(job)

    with _connect() as con:
        _ensure_schema(con)
        row = con.execute(
            "SELECT * FROM _automatic_training_runs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return None
    return _automatic_run_from_row(row, include_result=True)


def _automatic_run_summary(job: dict[str, object]):
    result = job.get("result") or {}
    models = result.get("models", []) if isinstance(result, dict) else []
    best = None
    if models:
        best = max(
            models,
            key=lambda model: (
                _stroke_risk_score(model.get("metrics", {})),
                float(model.get("metrics", {}).get("auc", 0)),
                float(model.get("metrics", {}).get("recall", 0)),
                float(model.get("metrics", {}).get("f1", 0)),
            ),
        )

    return {
        "id": job["id"],
        "status": job["status"],
        "message": job["message"],
        "createdAt": job["createdAt"],
        "startedAt": job.get("startedAt"),
        "finishedAt": job.get("finishedAt"),
        "baseModelId": (job.get("request") or {}).get("baseModelId"),
        "total": result.get("total") if isinstance(result, dict) else None,
        "trained": result.get("trained") if isinstance(result, dict) else None,
        "reused": result.get("reused") if isinstance(result, dict) else None,
        "bestModelId": best.get("modelId") if best else None,
        "bestScore": _stroke_risk_score(best.get("metrics", {})) if best else None,
        "error": job.get("error"),
    }


def _automatic_run_from_row(row, *, include_result: bool):
    request_payload = _load_json_payload(row["request_json"], {})
    result_payload = _load_json_payload(row["result_json"], None)
    job = {
        "id": row["job_id"],
        "status": row["status"],
        "message": row["message"],
        "createdAt": row["created_at"],
        "startedAt": row["started_at"],
        "finishedAt": row["finished_at"],
        "request": request_payload,
        "result": result_payload,
        "error": row["error"],
    }
    return job if include_result else _automatic_run_summary(job)


def _persist_automatic_run(job: dict[str, object]):
    request_payload = job.get("request") or {}
    if not isinstance(request_payload, dict) or not request_payload.get("automatic"):
        return

    with _automatic_runs_lock:
        with _connect() as con:
            _ensure_schema(con)
            con.execute(
                """
                INSERT INTO _automatic_training_runs
                    (job_id, base_model_id, status, message, created_at, started_at,
                     finished_at, request_json, result_json, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    base_model_id = excluded.base_model_id,
                    status = excluded.status,
                    message = excluded.message,
                    created_at = excluded.created_at,
                    started_at = excluded.started_at,
                    finished_at = excluded.finished_at,
                    request_json = excluded.request_json,
                    result_json = excluded.result_json,
                    error = excluded.error
                """,
                (
                    job["id"],
                    request_payload.get("baseModelId", ""),
                    job["status"],
                    job["message"],
                    job["createdAt"],
                    job.get("startedAt"),
                    job.get("finishedAt"),
                    _json_payload(request_payload),
                    _json_payload(job.get("result")),
                    job.get("error"),
                ),
            )


def _set_job(job_id: str, **updates):
    with _training_jobs_lock:
        job = _training_jobs[job_id]
        job.update(updates)
        public_job = _public_job(job)
    _persist_automatic_run(public_job)
    return public_job


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


def _fine_tune_suffix(payload: dict[str, object]):
    tuning_payload = {
        "removedFeatures": sorted(payload.get("removedFeatures", [])),
        "hyperparameters": payload.get("hyperparameters", {}),
        "targetRatio": payload.get("targetRatio", 1.0),
        "classificationThreshold": payload.get("classificationThreshold", 0.5),
    }
    serialized = json.dumps(tuning_payload, sort_keys=True, separators=(",", ":"))
    return f"tuned_{uuid.uuid5(uuid.NAMESPACE_URL, serialized).hex[:10]}"


def _clean_model_id_suffix(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if not all(char.isalnum() or char in {"_", "-"} for char in value):
        abort(400, description="'modelIdSuffix' may only contain letters, digits, underscores, or hyphens")
    return value


def _automatic_uses_gpu(algorithm: str) -> bool:
    return algorithm in {"xgboost", "lightgbm"}


def _automatic_fine_tune_grid(algorithm: str):
    if algorithm == "random_forest":
        return [
            {"n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf}
            for n_estimators in (100, 200, 300)
            for max_depth in (6, 10, 20)
            for min_samples_leaf in (1, 2, 4)
        ]
    if algorithm == "xgboost":
        return [
            {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate, "subsample": subsample}
            for n_estimators in (100, 200, 300)
            for max_depth in (3, 6, 9)
            for learning_rate in (0.05, 0.1, 0.3)
            for subsample in (0.8, 1.0)
        ]
    if algorithm == "lightgbm":
        return [
            {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate, "num_leaves": num_leaves}
            for n_estimators in (100, 200, 300)
            for max_depth in (-1, 10, 20)
            for learning_rate in (0.05, 0.1, 0.2)
            for num_leaves in (31, 63)
        ]
    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def _automatic_model_specs(base_model_id: str):
    with _connect() as con:
        _ensure_schema(con)
        row = con.execute(
            """
            SELECT model_id, algorithm, feature_set, uncertainty_variant, balancing_method
            FROM _model_results
            WHERE model_id = ?
            """,
            (base_model_id,),
        ).fetchone()
    if not row:
        abort(404, description=f"Model '{base_model_id}' not found")
    if _is_tuned_model(row["model_id"]):
        abort(400, description="Automatic fine-tuning can only start from a normal model")

    algorithm = row["algorithm"]
    dataset_id = f"{row['feature_set']}_{row['uncertainty_variant']}"
    balancing_method = row["balancing_method"]
    specs = []
    for hyperparameters in _automatic_fine_tune_grid(algorithm):
        for classification_threshold in (0.4, 0.5, 0.6):
            suffix_payload = {
                "removedFeatures": [],
                "hyperparameters": hyperparameters,
                "targetRatio": 1.0,
                "classificationThreshold": classification_threshold,
            }
            suffix = _fine_tune_suffix(suffix_payload).replace("tuned_", "tuned_auto_", 1)
            specs.append({
                "algorithm": algorithm,
                "datasetId": dataset_id,
                "balancingMethod": balancing_method,
                "targetRatio": 1.0,
                "classificationThreshold": classification_threshold,
                "forceRetrain": True,
                "useGpu": _automatic_uses_gpu(algorithm),
                "removedFeatures": [],
                "hyperparameters": hyperparameters,
                "modelIdSuffix": suffix,
            })
    return row, specs


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
            use_gpu = bool(spec.get("useGpu", payload["useGpu"])) and _automatic_uses_gpu(algorithm)
            device_label = "GPU" if use_gpu else "CPU"
            _set_job(
                job_id,
                message=(
                    f"Training {index}/{total}: "
                    f"{algorithm} / {dataset_id} / {balancing_method} ({device_label})."
                ),
            )
            result = train_model(
                algorithm=algorithm,
                dataset_id=dataset_id,
                balancing_method=balancing_method,
                target_ratio=spec.get("targetRatio", payload["targetRatio"]),
                classification_threshold=spec.get("classificationThreshold", payload["classificationThreshold"]),
                force_retrain=spec.get("forceRetrain", payload["forceRetrain"]),
                use_gpu=use_gpu,
                removed_features=spec.get("removedFeatures", payload.get("removedFeatures", [])),
                hyperparameters=spec.get("hyperparameters", payload.get("hyperparameters", {})),
                model_id_suffix=spec.get("modelIdSuffix", payload.get("modelIdSuffix")),
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


def _queue_training_job(payload: dict[str, object], message: str = "Training queued."):
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "message": message,
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
    _persist_automatic_run(_public_job(job))
    thread.start()
    return job

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

    try:
        df = load_dataset(name)
    except Exception as exc:
        abort(500, description=str(exc))

    columns = list(df.columns)
    total = len(df)
    page_df = df.iloc[offset:offset + per_page]

    return jsonify({
        "name": name,
        "columns": columns,
        "rows": [{col: _display(row[col]) for col in columns} for _, row in page_df.iterrows()],
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
    model_id_suffix = _clean_model_id_suffix(model_id_suffix)

    if model_specs:
        if not isinstance(model_specs, list):
            abort(400, description="'models' must be an array of model specs")
        normalized_specs = []
        for spec in model_specs:
            if not isinstance(spec, dict):
                abort(400, description="'models' must contain objects")
            normalized_spec = {
                "algorithm": str(spec.get("algorithm", "")),
                "datasetId": str(spec.get("datasetId", "")),
                "balancingMethod": str(spec.get("balancingMethod", "")),
            }
            for key in (
                "targetRatio",
                "classificationThreshold",
                "forceRetrain",
                "useGpu",
                "removedFeatures",
                "hyperparameters",
                "modelIdSuffix",
            ):
                if key in spec:
                    normalized_spec[key] = spec[key]
            if "modelIdSuffix" in normalized_spec:
                normalized_spec["modelIdSuffix"] = _clean_model_id_suffix(normalized_spec["modelIdSuffix"])
            normalized_specs.append(normalized_spec)
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

    for spec in normalized_specs:
        if spec["algorithm"] == "random_forest":
            spec["useGpu"] = False

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
    job = _queue_training_job(payload)
    return jsonify(_public_job(job)), 202


@app.route("/api/training/automatic-jobs", methods=["POST"])
def api_start_automatic_training_job():
    body = request.get_json(force=True) or {}
    model_id = str(body.get("modelId", "")).strip()
    if not model_id:
        abort(400, description="'modelId' required in request body")

    base_model, specs = _automatic_model_specs(model_id)
    payload = {
        "algorithms": [base_model["algorithm"]],
        "datasetIds": [f"{base_model['feature_set']}_{base_model['uncertainty_variant']}"],
        "balancingMethods": [base_model["balancing_method"]],
        "targetRatio": 1.0,
        "classificationThreshold": 0.5,
        "forceRetrain": True,
        "useGpu": _automatic_uses_gpu(base_model["algorithm"]),
        "automatic": True,
        "baseModelId": model_id,
        "models": specs,
    }
    job = _queue_training_job(
        payload,
        message=f"Automatic fine-tuning queued for {len(specs)} parameter combinations.",
    )
    return jsonify(_public_job(job)), 202


@app.route("/api/training/automatic-runs")
def api_automatic_training_runs():
    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute(
            """
            SELECT *
            FROM _automatic_training_runs
            ORDER BY created_at DESC
            """
        ).fetchall()
    return jsonify([
        _automatic_run_from_row(row, include_result=False)
        for row in rows
    ])


@app.route("/api/training/automatic-runs/<job_id>")
def api_automatic_training_run(job_id: str):
    job = _load_automatic_training_run(job_id)
    if job is None:
        abort(404, description=f"Automatic training run '{job_id}' not found")
    return jsonify(job)


@app.route("/api/training/automatic-runs-export")
def api_export_automatic_training_runs():
    job_ids = []
    for value in request.args.getlist("ids"):
        job_ids.extend(part.strip() for part in value.split(",") if part.strip())
    job_ids = list(dict.fromkeys(job_ids))

    if not job_ids:
        abort(400, description="At least one automatic training run ID is required")
    if len(job_ids) > 100:
        abort(400, description="Automatic training run export is limited to 100 runs")

    jobs = []
    missing_ids = []
    for job_id in job_ids:
        job = _load_automatic_training_run(job_id)
        if job is None:
            missing_ids.append(job_id)
        else:
            jobs.append(job)
    if missing_ids:
        abort(404, description=f"Automatic training run(s) not found: {', '.join(missing_ids)}")

    workbook = _automatic_runs_export_workbook(jobs)
    filename = f"automatic_fine_tuning_runs_{len(jobs)}.xlsx"
    return send_file(
        workbook,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.route("/api/training/automatic-runs/<job_id>/export")
def api_export_automatic_training_run(job_id: str):
    job = _load_automatic_training_run(job_id)
    if job is None:
        abort(404, description=f"Automatic training run '{job_id}' not found")

    workbook = _automatic_run_export_workbook(job)
    filename = f"automatic_fine_tuning_{_safe_export_filename(job_id)}.xlsx"
    return send_file(
        workbook,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


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

@app.route("/api/models/<model_id>/predict", methods=["POST"])
def api_predict(model_id: str):
    body = request.get_json(force=True) or {}
    features = body.get("features", {})
    if not features:
        abort(400, description="'features' dict required in request body")

    with _connect() as con:
        _ensure_schema(con)
        row = con.execute(
            "SELECT algorithm, feature_set, uncertainty_variant, feature_columns, metrics FROM _model_results WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    if not row:
        abort(404, description=f"Model '{model_id}' not found")

    feature_columns = json.loads(row["feature_columns"])
    metadata = {
        "algorithm": row["algorithm"],
        "feature_set": row["feature_set"],
        "uncertainty_variant": row["uncertainty_variant"],
    }

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

    response = {
        "prediction": prediction,
        "probability": probability,
        "classificationThreshold": classification_threshold,
        "label": "Stroke" if prediction == 1 else "No Stroke",
    }

    if body.get("explain"):
        try:
            response["explanation"] = _build_shap_explanation(
                model_id,
                classifier,
                metadata,
                feature_columns,
                X,
                probability,
            )
        except Exception as exc:
            response["explanation"] = None
            response["explanationError"] = f"SHAP explanation failed: {exc}"

    return jsonify(response)


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
