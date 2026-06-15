from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Callable

from .registry import _connect, _ensure_schema
from .training import (
    ALGORITHMS,
    BALANCING_STRATEGIES,
    DATA_BALANCED_MODELS_DIR,
    MODELS_DIR,
    parse_dataset_id,
    train_model,
)


StatusCallback = Callable[[str], None]

FINAL_MODEL_DATASET_IDS = [
    "clinical_with_uncertain",
    "clinical_without_uncertain",
    "lifestyle_with_uncertain",
    "lifestyle_without_uncertain",
    "combined_with_uncertain",
    "combined_without_uncertain",
]

FINAL_MODEL_CONFIGS = [
    {
        "profile": "best_balance",
        "selectionReason": "Best overall balance of AUC, F1, precision, recall, and false-negative control for Random Forest.",
        "algorithm": "random_forest",
        "classificationThreshold": 0.4,
        "hyperparameters": {"max_depth": 9, "min_samples_leaf": 2, "n_estimators": 300},
        "sourceModelId": "random_forest_weighted_combined_with_uncertain__tuned_auto_fb0ea071e1",
        "sourceMetrics": {"falsePositives": 34799},
    },
    {
        "profile": "low_false_positive",
        "selectionReason": "Lowest false-positive count for Random Forest; tied rows use the simplest leaf setting.",
        "algorithm": "random_forest",
        "classificationThreshold": 0.6,
        "hyperparameters": {"max_depth": 3, "min_samples_leaf": 1, "n_estimators": 100},
        "sourceModelId": "random_forest_weighted_combined_with_uncertain__tuned_auto_b336f63bb5",
        "sourceMetrics": {"falsePositives": 9799},
    },
    {
        "profile": "best_balance",
        "selectionReason": "Best overall balance of AUC, F1, precision, recall, and false-negative control for LightGBM.",
        "algorithm": "lightgbm",
        "classificationThreshold": 0.4,
        "hyperparameters": {"learning_rate": 0.05, "max_depth": 10, "n_estimators": 100, "num_leaves": 31},
        "sourceModelId": "lightgbm_weighted_combined_with_uncertain__tuned_auto_8582a30afa",
        "sourceMetrics": {"falsePositives": 32261},
    },
    {
        "profile": "low_false_positive",
        "selectionReason": "Lowest false-positive count for LightGBM.",
        "algorithm": "lightgbm",
        "classificationThreshold": 0.6,
        "hyperparameters": {"learning_rate": 0.2, "max_depth": 20, "n_estimators": 300, "num_leaves": 63},
        "sourceModelId": "lightgbm_weighted_combined_with_uncertain__tuned_auto_e845b362e6",
        "sourceMetrics": {"falsePositives": 9421},
    },
    {
        "profile": "best_balance",
        "selectionReason": "Best overall balance of AUC, F1, precision, recall, and false-negative control for XGBoost.",
        "algorithm": "xgboost",
        "classificationThreshold": 0.4,
        "hyperparameters": {"learning_rate": 0.1, "max_depth": 3, "n_estimators": 100, "subsample": 1.0},
        "sourceModelId": "xgboost_weighted_combined_with_uncertain__tuned_auto_3d07d349b5",
        "sourceMetrics": {"falsePositives": 32248},
    },
    {
        "profile": "low_false_positive",
        "selectionReason": "Lowest false-positive count for XGBoost.",
        "algorithm": "xgboost",
        "classificationThreshold": 0.6,
        "hyperparameters": {"learning_rate": 0.3, "max_depth": 9, "n_estimators": 300, "subsample": 0.8},
        "sourceModelId": "xgboost_weighted_combined_with_uncertain__tuned_auto_b5f9c60517",
        "sourceMetrics": {"falsePositives": 2564},
    },
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_payload(value):
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _load_json_payload(value, fallback):
    if value is None:
        return fallback
    return json.loads(value)


def _is_tuned_model(model_id: str) -> bool:
    return "__tuned_" in model_id


def _automatic_uses_gpu(algorithm: str) -> bool:
    return algorithm in {"xgboost", "lightgbm"}


def _fine_tune_suffix(payload: dict[str, object]) -> str:
    tuning_payload = {
        "removedFeatures": sorted(payload.get("removedFeatures", [])),
        "hyperparameters": payload.get("hyperparameters", {}),
        "targetRatio": payload.get("targetRatio", 1.0),
        "classificationThreshold": payload.get("classificationThreshold", 0.5),
    }
    serialized = json.dumps(tuning_payload, sort_keys=True, separators=(",", ":"))
    return f"tuned_{uuid.uuid5(uuid.NAMESPACE_URL, serialized).hex[:10]}"


def rank_models(models: list[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        models,
        key=lambda model: (
            float(model.get("metrics", {}).get("auc", 0)),
            float(model.get("metrics", {}).get("recall", 0)),
            float(model.get("metrics", {}).get("f1", 0)),
            float(model.get("metrics", {}).get("precision", 0)),
        ),
        reverse=True,
    )


def dataset_options() -> list[dict[str, str]]:
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


def expected_data_balanced_model_specs(
    datasets: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    datasets = datasets if datasets is not None else dataset_options()
    return [
        {
            "id": f"{algorithm}_{balancing_method}_{dataset['id']}",
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


def data_balanced_training_coverage() -> dict[str, object]:
    expected = expected_data_balanced_model_specs()
    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute("SELECT model_id FROM _model_results").fetchall()

    registered_model_ids = {row["model_id"] for row in rows}
    available = [spec for spec in expected if spec["id"] in registered_model_ids]
    missing = [spec for spec in expected if spec["id"] not in registered_model_ids]

    return {
        "totalExpected": len(expected),
        "availableCount": len(available),
        "missingCount": len(missing),
        "available": available,
        "missing": missing,
    }


def train_data_balanced_models(
    *,
    algorithms: list[str] | None = None,
    dataset_ids: list[str] | None = None,
    balancing_methods: list[str] | None = None,
    force_retrain: bool = False,
    use_gpu: bool = True,
    target_ratio: float = 1.0,
    classification_threshold: float = 0.5,
    status: StatusCallback = print,
) -> dict[str, object]:
    datasets = dataset_options()
    dataset_ids = dataset_ids or [dataset["id"] for dataset in datasets]
    algorithms = algorithms or list(ALGORITHMS.keys())
    balancing_methods = balancing_methods or list(BALANCING_STRATEGIES.keys())

    combinations = [
        {
            "algorithm": algorithm,
            "datasetId": dataset_id,
            "balancingMethod": balancing_method,
        }
        for dataset_id in dataset_ids
        for balancing_method in balancing_methods
        for algorithm in algorithms
    ]

    results = []
    total = len(combinations)
    for index, spec in enumerate(combinations, start=1):
        algorithm = spec["algorithm"]
        dataset_id = spec["datasetId"]
        balancing_method = spec["balancingMethod"]
        model_uses_gpu = bool(use_gpu) and _automatic_uses_gpu(algorithm)
        device_label = "GPU" if model_uses_gpu else "CPU"
        status(
            f"Training {index}/{total}: "
            f"{algorithm} / {dataset_id} / {balancing_method} ({device_label})"
        )
        result = train_model(
            algorithm=algorithm,
            dataset_id=dataset_id,
            balancing_method=balancing_method,
            target_ratio=target_ratio,
            classification_threshold=classification_threshold,
            force_retrain=force_retrain,
            use_gpu=model_uses_gpu,
            model_output_dir=DATA_BALANCED_MODELS_DIR,
        )
        results.append(result)

    reused = sum(1 for result in results if result["reusedExistingModel"])
    return {
        "models": results,
        "total": total,
        "trained": total - reused,
        "reused": reused,
    }


def train_missing_data_balanced_models(
    *,
    use_gpu: bool = True,
    status: StatusCallback = print,
) -> dict[str, object]:
    missing = data_balanced_training_coverage()["missing"]
    if not missing:
        status("No missing data-balanced models found.")
        return {"models": [], "total": 0, "trained": 0, "reused": 0}

    results = []
    total = len(missing)
    for index, spec in enumerate(missing, start=1):
        algorithm = spec["algorithm"]
        dataset_id = spec["datasetId"]
        balancing_method = spec["balancingMethod"]
        model_uses_gpu = bool(use_gpu) and _automatic_uses_gpu(algorithm)
        device_label = "GPU" if model_uses_gpu else "CPU"
        status(
            f"Training missing {index}/{total}: "
            f"{algorithm} / {dataset_id} / {balancing_method} ({device_label})"
        )
        results.append(train_model(
            algorithm=algorithm,
            dataset_id=dataset_id,
            balancing_method=balancing_method,
            force_retrain=False,
            use_gpu=model_uses_gpu,
            model_output_dir=DATA_BALANCED_MODELS_DIR,
        ))

    reused = sum(1 for result in results if result["reusedExistingModel"])
    return {
        "models": results,
        "total": total,
        "trained": total - reused,
        "reused": reused,
    }


def train_final_models(
    *,
    dataset_ids: list[str] | None = None,
    configs: list[dict[str, object]] | None = None,
    force_retrain: bool = False,
    use_gpu: bool = True,
    status: StatusCallback = print,
) -> dict[str, object]:
    dataset_ids = dataset_ids or FINAL_MODEL_DATASET_IDS
    configs = configs or FINAL_MODEL_CONFIGS

    combinations = [
        {**config, "datasetId": dataset_id}
        for dataset_id in dataset_ids
        for config in configs
    ]

    results = []
    total = len(combinations)
    for index, spec in enumerate(combinations, start=1):
        algorithm = str(spec["algorithm"])
        dataset_id = str(spec["datasetId"])
        profile = str(spec["profile"])
        model_uses_gpu = bool(use_gpu) and _automatic_uses_gpu(algorithm)
        device_label = "GPU" if model_uses_gpu else "CPU"
        status(
            f"Training final {index}/{total}: "
            f"{algorithm} / {dataset_id} / {profile} ({device_label})"
        )
        results.append(train_model(
            algorithm=algorithm,
            dataset_id=dataset_id,
            balancing_method="weighted",
            classification_threshold=float(spec["classificationThreshold"]),
            force_retrain=force_retrain,
            use_gpu=model_uses_gpu,
            hyperparameters=dict(spec["hyperparameters"]),
            model_id_suffix=f"final_{profile}",
            model_output_dir=MODELS_DIR,
        ))

    reused = sum(1 for result in results if result["reusedExistingModel"])
    return {
        "models": results,
        "total": total,
        "trained": total - reused,
        "reused": reused,
        "configs": configs,
        "datasetIds": dataset_ids,
        "modelOutputDir": str(MODELS_DIR),
    }


def automatic_fine_tune_grid(algorithm: str) -> list[dict[str, object]]:
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


def automatic_model_specs(base_model_id: str) -> tuple[dict[str, str], list[dict[str, object]]]:
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
        raise KeyError(f"Model '{base_model_id}' not found.")
    if _is_tuned_model(row["model_id"]):
        raise ValueError("Automatic fine-tuning can only start from a normal model.")

    base_model = dict(row)
    algorithm = base_model["algorithm"]
    dataset_id = f"{base_model['feature_set']}_{base_model['uncertainty_variant']}"
    balancing_method = base_model["balancing_method"]
    specs = []
    for hyperparameters in automatic_fine_tune_grid(algorithm):
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
    return base_model, specs


def persist_automatic_run(job: dict[str, object]) -> None:
    request_payload = job.get("request") or {}
    if not isinstance(request_payload, dict) or not request_payload.get("automatic"):
        return

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


def run_automatic_fine_tuning(
    base_model_id: str,
    *,
    persist: bool = True,
    use_gpu: bool | None = None,
    status: StatusCallback = print,
) -> dict[str, object]:
    base_model, specs = automatic_model_specs(base_model_id)
    if use_gpu is not None:
        for spec in specs:
            spec["useGpu"] = bool(use_gpu) and _automatic_uses_gpu(str(spec["algorithm"]))

    job = {
        "id": uuid.uuid4().hex,
        "status": "running",
        "message": f"Automatic fine-tuning started for {len(specs)} parameter combinations.",
        "createdAt": _now_iso(),
        "startedAt": _now_iso(),
        "finishedAt": None,
        "request": {
            "automatic": True,
            "baseModelId": base_model_id,
            "algorithms": [base_model["algorithm"]],
            "datasetIds": [f"{base_model['feature_set']}_{base_model['uncertainty_variant']}"],
            "balancingMethods": [base_model["balancing_method"]],
            "models": specs,
        },
        "result": None,
        "error": None,
    }
    if persist:
        persist_automatic_run(job)

    results = []
    total = len(specs)
    try:
        for index, spec in enumerate(specs, start=1):
            device_label = "GPU" if spec["useGpu"] else "CPU"
            message = (
                f"Training {index}/{total}: {spec['algorithm']} / "
                f"threshold={spec['classificationThreshold']} ({device_label})"
            )
            job["message"] = message
            status(message)
            if persist:
                persist_automatic_run(job)

            results.append(train_model(
                algorithm=str(spec["algorithm"]),
                dataset_id=str(spec["datasetId"]),
                balancing_method=str(spec["balancingMethod"]),
                target_ratio=float(spec["targetRatio"]),
                classification_threshold=float(spec["classificationThreshold"]),
                force_retrain=bool(spec["forceRetrain"]),
                use_gpu=bool(spec["useGpu"]),
                removed_features=list(spec["removedFeatures"]),
                hyperparameters=dict(spec["hyperparameters"]),
                model_id_suffix=str(spec["modelIdSuffix"]),
            ))

        reused = sum(1 for result in results if result["reusedExistingModel"])
        job.update({
            "status": "succeeded",
            "finishedAt": _now_iso(),
            "message": "Automatic fine-tuning completed.",
            "result": {
                "models": results,
                "total": total,
                "trained": total - reused,
                "reused": reused,
            },
        })
    except Exception as exc:
        job.update({
            "status": "failed",
            "finishedAt": _now_iso(),
            "message": str(exc),
            "error": str(exc),
        })
        raise
    finally:
        if persist:
            persist_automatic_run(job)

    return job


def automatic_run_summaries() -> list[dict[str, object]]:
    with _connect() as con:
        _ensure_schema(con)
        rows = con.execute(
            """
            SELECT *
            FROM _automatic_training_runs
            ORDER BY created_at DESC
            """
        ).fetchall()

    summaries = []
    for row in rows:
        result = _load_json_payload(row["result_json"], None) or {}
        ranked = rank_models(result.get("models", [])) if isinstance(result, dict) else []
        best = ranked[0] if ranked else None
        summaries.append({
            "id": row["job_id"],
            "status": row["status"],
            "message": row["message"],
            "createdAt": row["created_at"],
            "startedAt": row["started_at"],
            "finishedAt": row["finished_at"],
            "baseModelId": row["base_model_id"],
            "total": result.get("total") if isinstance(result, dict) else None,
            "trained": result.get("trained") if isinstance(result, dict) else None,
            "reused": result.get("reused") if isinstance(result, dict) else None,
            "bestModelId": best.get("modelId") if best else None,
            "error": row["error"],
        })
    return summaries
