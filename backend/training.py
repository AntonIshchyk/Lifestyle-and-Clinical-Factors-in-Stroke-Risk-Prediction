from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from .registry import _connect, load_dataset, register_model
except ImportError:
    from registry import _connect, load_dataset, register_model

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
AI_MODULE_DIR = PROJECT_ROOT / "ai_module"
MODELS_DIR = AI_MODULE_DIR / "models"
TUNED_MODELS_DIR = AI_MODULE_DIR / "tuned-models"
BALANCED_CACHE_DIR = AI_MODULE_DIR / "balanced_training_cache"

if str(AI_MODULE_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(AI_MODULE_DIR.resolve()))

from balancing import balance_training_data
from smote import balance_training_data_smote
from SMOTENC import balance_training_data_smotenc
from weighted import balance_training_data_weighted

TARGET_COL = "CVDSTRK3"

ALGORITHMS = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

BALANCING_STRATEGIES = {
    "random_oversampling": balance_training_data,
    "smote": balance_training_data_smote,
    "smotenc": balance_training_data_smotenc,
    "weighted": balance_training_data_weighted,
}

CACHEABLE_BALANCING_METHODS = {
    "random_oversampling",
    "smote",
    "smotenc",
}


def parse_dataset_id(dataset_id: str) -> tuple[str, str]:
    for uncertainty_variant in ("without_uncertain", "with_uncertain"):
        suffix = f"_{uncertainty_variant}"
        if dataset_id.endswith(suffix):
            return dataset_id[: -len(suffix)], uncertainty_variant
    raise ValueError(
        "Dataset id must end with '_with_uncertain' or '_without_uncertain'."
    )


def dataset_cache_metadata(dataset_id: str) -> dict[str, object]:
    with _connect() as con:
        row = con.execute(
            "SELECT reference, created_at FROM _registry WHERE id = ? AND type = 'dataset'",
            (dataset_id,),
        ).fetchone()
        if not row:
            raise KeyError(f"Dataset '{dataset_id}' not found")
        table_name = row["reference"]
        row_count = con.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]

    return {
        "dataset_id": dataset_id,
        "table_name": table_name,
        "created_at": row["created_at"],
        "row_count": int(row_count),
    }


def _cache_key(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]


def _balanced_cache_path(
    *,
    dataset_id: str,
    balancing_method: str,
    target_ratio: float,
    random_state: int,
    feature_columns: list[str],
    target_counts: dict[str, int],
) -> tuple[Path, dict[str, object]]:
    metadata = dataset_cache_metadata(dataset_id)
    payload = {
        "version": 1,
        **metadata,
        "target_col": TARGET_COL,
        "feature_columns": feature_columns,
        "target_counts": target_counts,
        "balancing_method": balancing_method,
        "target_ratio": target_ratio,
        "random_state": random_state,
        "test_size": 0.2,
        "split": "sklearn.train_test_split.stratify",
    }
    key = _cache_key(payload)
    return BALANCED_CACHE_DIR / balancing_method / dataset_id / f"{key}.joblib", payload


def _sanitized_balance_info(balance_info: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in balance_info.items()
        if key != "fit_params"
    }


def balance_training_fold(
    *,
    dataset_id: str,
    balancing_method: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_ratio: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object], bool]:
    balance_func = BALANCING_STRATEGIES[balancing_method]
    target_counts = {
        str(key): int(value)
        for key, value in y_train.value_counts().sort_index().items()
    }

    if balancing_method not in CACHEABLE_BALANCING_METHODS:
        X_balanced, y_balanced, balance_info = balance_func(
            X_train,
            y_train,
            target_ratio=target_ratio,
            random_state=random_state,
        )
        return X_balanced, y_balanced, balance_info, False

    cache_path, cache_payload = _balanced_cache_path(
        dataset_id=dataset_id,
        balancing_method=balancing_method,
        target_ratio=target_ratio,
        random_state=random_state,
        feature_columns=list(X_train.columns),
        target_counts=target_counts,
    )
    if cache_path.exists():
        cached = joblib.load(cache_path)
        balance_info = cached["balance_info"]
        balance_info["cache"] = {
            "hit": True,
            "path": str(cache_path),
        }
        return cached["X_train"], cached["y_train"], balance_info, True

    X_balanced, y_balanced, balance_info = balance_func(
        X_train,
        y_train,
        target_ratio=target_ratio,
        random_state=random_state,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    balance_info_for_cache = _sanitized_balance_info(balance_info)
    balance_info_for_cache["cache"] = {
        "hit": False,
        "path": str(cache_path),
        "key": cache_path.stem,
        "metadata": cache_payload,
    }
    joblib.dump(
        {
            "X_train": X_balanced,
            "y_train": y_balanced,
            "balance_info": balance_info_for_cache,
        },
        cache_path,
        compress=3,
    )
    balance_info = {
        **balance_info,
        "cache": {
            "hit": False,
            "path": str(cache_path),
        },
    }
    return X_balanced, y_balanced, balance_info, False


def _xgboost_gpu_params() -> dict[str, object]:
    import xgboost

    major_version = int(xgboost.__version__.split(".", 1)[0])
    if major_version >= 2:
        return {"tree_method": "hist", "device": "cuda"}
    return {"tree_method": "gpu_hist", "predictor": "gpu_predictor"}


def make_classifier(
    algorithm: str,
    *,
    use_gpu: bool = False,
    balancing_method: str | None = None,
    hyperparameters: dict[str, object] | None = None,
):
    hyperparameters = hyperparameters or {}

    if algorithm == "random_forest":
        params = {
            "n_estimators": int(hyperparameters.get("n_estimators", 200)),
            "max_depth": (
                None
                if hyperparameters.get("max_depth") in (None, "", 0, "0")
                else int(hyperparameters["max_depth"])
            ),
            "min_samples_leaf": int(hyperparameters.get("min_samples_leaf", 1)),
        }
        return RandomForestClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced" if balancing_method == "weighted" else None,
        )

    if algorithm == "xgboost":
        params = {
            "n_estimators": int(hyperparameters.get("n_estimators", 100)),
            "max_depth": int(hyperparameters.get("max_depth", 6)),
            "learning_rate": float(hyperparameters.get("learning_rate", 0.3)),
            "subsample": float(hyperparameters.get("subsample", 1.0)),
        }
        return XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            **(_xgboost_gpu_params() if use_gpu else {"tree_method": "hist"}),
        )

    if algorithm == "lightgbm":
        params = {
            "n_estimators": int(hyperparameters.get("n_estimators", 100)),
            "max_depth": int(hyperparameters.get("max_depth", -1)),
            "learning_rate": float(hyperparameters.get("learning_rate", 0.1)),
            "num_leaves": int(hyperparameters.get("num_leaves", 31)),
        }
        return LGBMClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **({"device_type": "gpu"} if use_gpu else {}),
        )

    raise ValueError(f"Unsupported algorithm '{algorithm}'.")


def _assert_gpu_backend_used(clf, algorithm: str) -> None:
    if algorithm == "xgboost":
        config = json.loads(clf.get_booster().save_config())
        device = config.get("learner", {}).get("generic_param", {}).get("device", "")
        if not str(device).startswith("cuda"):
            raise RuntimeError(f"XGBoost was requested to use GPU, but trained with device='{device}'.")
        return

    if algorithm == "lightgbm":
        device_type = getattr(clf.booster_, "params", {}).get("device_type", "")
        if str(device_type).lower() != "gpu":
            raise RuntimeError(
                f"LightGBM was requested to use GPU, but trained with device_type='{device_type}'."
            )


def train_model(
    *,
    algorithm: str,
    dataset_id: str,
    balancing_method: str,
    target_ratio: float = 1.0,
    classification_threshold: float = 0.5,
    force_retrain: bool = False,
    use_gpu: bool = False,
    removed_features: list[str] | None = None,
    hyperparameters: dict[str, object] | None = None,
    model_id_suffix: str | None = None,
) -> dict[str, object]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported algorithm '{algorithm}'.")
    if balancing_method not in BALANCING_STRATEGIES:
        raise ValueError(f"Unsupported balancing method '{balancing_method}'.")
    if balancing_method == "weighted":
        target_ratio = 1.0
    if not 0 < target_ratio <= 1:
        raise ValueError("targetRatio must be in the interval (0, 1].")
    if not 0 < classification_threshold < 1:
        raise ValueError("classificationThreshold must be in the interval (0, 1).")

    feature_set, uncertainty_variant = parse_dataset_id(dataset_id)
    ds = load_dataset(dataset_id)
    if TARGET_COL not in ds.columns:
        raise ValueError(f"Dataset '{dataset_id}' does not include target column {TARGET_COL}.")

    X = ds.drop(columns=[TARGET_COL]).apply(pd.to_numeric, errors="coerce")
    removed_features = sorted(set(removed_features or []))
    valid_removed_features = [column for column in removed_features if column in X.columns]
    if valid_removed_features:
        X = X.drop(columns=valid_removed_features)
    if X.empty:
        raise ValueError("At least one feature column must remain after feature removal.")
    y = ds[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_balanced, y_train_balanced, balance_info, balance_cache_hit = balance_training_fold(
        dataset_id=dataset_id,
        balancing_method=balancing_method,
        X_train=X_train,
        y_train=y_train,
        target_ratio=target_ratio,
        random_state=42,
    )

    base_model_id = f"{algorithm}_{balancing_method}_{dataset_id}"
    model_id = (
        f"{base_model_id}__{model_id_suffix}"
        if model_id_suffix
        else base_model_id
    )
    model_dir = TUNED_MODELS_DIR if model_id_suffix else MODELS_DIR
    model_dir.mkdir(exist_ok=True)
    pkl_path = (model_dir / f"{model_id}.pkl").resolve()

    reused = pkl_path.exists() and not force_retrain
    if reused:
        clf = joblib.load(pkl_path)
    else:
        clf = make_classifier(
            algorithm,
            use_gpu=use_gpu,
            balancing_method=balancing_method,
            hyperparameters=hyperparameters,
        )
        fit_params = dict(balance_info.get("fit_params", {}))
        if algorithm == "random_forest" and balancing_method == "weighted" and not use_gpu:
            fit_params.pop("sample_weight", None)
        try:
            clf.fit(X_train_balanced, y_train_balanced, **fit_params)
            if use_gpu and algorithm in {"xgboost", "lightgbm"}:
                _assert_gpu_backend_used(clf, algorithm)
        except Exception as exc:
            if use_gpu and algorithm in {"random_forest", "xgboost", "lightgbm"}:
                raise RuntimeError(
                    f"{ALGORITHMS[algorithm]} GPU training failed. Disable GPU training "
                    "or verify that the package GPU backend and drivers are installed. "
                    f"Details: {exc}"
                ) from exc
            raise
        joblib.dump(clf, pkl_path)

    try:
        positive_class_index = list(clf.classes_).index(1)
    except ValueError as exc:
        raise ValueError("Trained classifier does not expose class label 1.") from exc

    y_prob = clf.predict_proba(X_test)[:, positive_class_index]
    y_pred = (y_prob >= classification_threshold).astype(int)

    report_raw = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    importances = getattr(clf, "feature_importances_", np.zeros(len(X.columns)))
    metrics = {
        "auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "classificationThreshold": classification_threshold,
    }
    classification_report_payload = {
        "classes": {
            k: v
            for k, v in report_raw.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "macro_avg": report_raw["macro avg"],
        "weighted_avg": report_raw["weighted avg"],
        "accuracy": report_raw["accuracy"],
    }
    confusion_matrix_payload = {
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }

    register_model(
        model_id=model_id,
        algorithm=algorithm,
        feature_set=feature_set,
        uncertainty_variant=uncertainty_variant,
        balancing_method=balancing_method,
        model_path=str(pkl_path),
        metrics=metrics,
        classification_report=classification_report_payload,
        confusion_matrix=confusion_matrix_payload,
        feature_importances=[
            {"feature": col, "importance": float(imp)}
            for col, imp in zip(X.columns, importances)
        ],
        roc_curve={"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        feature_columns=list(X.columns),
    )

    return {
        "modelId": model_id,
        "algorithm": algorithm,
        "datasetId": dataset_id,
        "featureSet": feature_set,
        "uncertaintyVariant": uncertainty_variant,
        "balancingMethod": balancing_method,
        "targetRatio": target_ratio,
        "classificationThreshold": classification_threshold,
        "useGpu": use_gpu,
        "reusedExistingModel": reused,
        "reusedBalancedData": balance_cache_hit,
        "removedFeatures": valid_removed_features,
        "hyperparameters": hyperparameters or {},
        "metrics": metrics,
        "classificationReport": classification_report_payload,
        "confusionMatrix": confusion_matrix_payload,
        "balanceInfo": _sanitized_balance_info(balance_info),
    }
