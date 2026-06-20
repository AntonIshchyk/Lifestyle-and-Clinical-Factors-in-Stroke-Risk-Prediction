from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from sklearn.impute import SimpleImputer

CONTINUOUS_FEATURES = {
    "_AGE80",
    "_BMI5",
    "_DRNKWK3",
    "_LCSYQTS",
    "_LCSYSMK",
    "AVEDRNK3",
    "DRNK3GE5",
    "EXERHMM1",
    "FRUIT2",
    "FRUITJU2",
    "HEIGHT3",
    "LCSLAST_",
    "LCSNUMC_",
    "MARIJAN1",
    "MAXDRNKS",
    "MENTHLTH",
    "PHYSHLTH",
    "POORHLTH",
    "POTATOE1",
    "SSBFRUT3",
    "SSBSUGR2",
    "SLEPTIM1",
    "VEGETAB2",
    "WEIGHT2",
}


def _is_integer_like(values: pd.Series) -> bool:
    numeric_values = pd.to_numeric(values.dropna(), errors="coerce").dropna()
    if numeric_values.empty:
        return False
    return bool(np.isclose(numeric_values, np.round(numeric_values)).all())


def infer_categorical_features(
    X: pd.DataFrame,
    *,
    max_categories: int = 25,
) -> list[str]:
    """Infer BRFSS-style categorical code columns for SMOTENC."""
    categorical_features = []
    for column in X.columns:
        if column in CONTINUOUS_FEATURES:
            continue

        values = X[column].dropna()
        if values.empty:
            categorical_features.append(column)
            continue

        if values.nunique(dropna=True) <= max_categories and _is_integer_like(values):
            categorical_features.append(column)

    return categorical_features


def _impute_for_smotenc(
    X: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    X_imputed = X.copy()
    continuous_features = [column for column in X.columns if column not in categorical_features]

    if continuous_features:
        continuous_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_imputed[continuous_features] = continuous_imputer.fit_transform(X[continuous_features])

    if categorical_features:
        categorical_imputer = SimpleImputer(strategy="most_frequent", keep_empty_features=True)
        X_imputed[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

    return X_imputed


def balance_training_data_smotenc(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_ratio: float = 1.0,
    random_state: int = 42,
    k_neighbors: int = 5,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Balance the training fold with SMOTENC for mixed categorical/numeric data."""
    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be in the interval (0, 1].")

    X = X.reset_index(drop=True)
    y = pd.Series(y, name=y.name).reset_index(drop=True)
    class_counts = y.value_counts().sort_index()
    missing_values = int(X.isna().sum().sum())
    categorical_features = categorical_features or infer_categorical_features(X)
    categorical_indices = [X.columns.get_loc(column) for column in categorical_features]

    if len(class_counts) < 2:
        return X.copy(), y.copy(), {
            "strategy": "none",
            "before": {str(k): int(v) for k, v in class_counts.items()},
            "after": {str(k): int(v) for k, v in class_counts.items()},
            "imputed_missing_values": missing_values,
            "categorical_feature_count": len(categorical_features),
        }

    minority_count = int(class_counts.min())
    if minority_count < 2:
        return X.copy(), y.copy(), {
            "strategy": "none_minority_too_small_for_smotenc",
            "target_ratio": target_ratio,
            "before": {str(k): int(v) for k, v in class_counts.items()},
            "after": {str(k): int(v) for k, v in class_counts.items()},
            "imputed_missing_values": missing_values,
            "categorical_feature_count": len(categorical_features),
        }

    majority_count = int(class_counts.max())
    target_count = int(round(majority_count * target_ratio))
    sampling_strategy = {
        class_value: target_count
        for class_value, class_count in class_counts.items()
        if class_count < target_count
    }

    if not sampling_strategy:
        return X.copy(), y.copy(), {
            "strategy": "smotenc",
            "target_ratio": target_ratio,
            "before": {str(k): int(v) for k, v in class_counts.items()},
            "after": {str(k): int(v) for k, v in class_counts.items()},
            "imputed_missing_values": missing_values,
            "categorical_feature_count": len(categorical_features),
        }

    if not categorical_indices or len(categorical_indices) == len(X.columns):
        raise ValueError(
            "SMOTENC requires a mix of categorical and continuous features. "
            "Pass categorical_features explicitly if automatic inference is wrong."
        )

    X_imputed = _impute_for_smotenc(X, categorical_features)

    smotenc = SMOTENC(
        categorical_features=categorical_indices,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=min(k_neighbors, minority_count - 1),
    )
    X_balanced, y_balanced = smotenc.fit_resample(X_imputed, y)
    X_balanced = pd.DataFrame(X_balanced, columns=X.columns)
    y_balanced = pd.Series(y_balanced, name=y.name)
    after_counts = y_balanced.value_counts().sort_index()

    return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True), {
        "strategy": "smotenc",
        "target_ratio": target_ratio,
        "k_neighbors": min(k_neighbors, minority_count - 1),
        "imputed_missing_values": missing_values,
        "categorical_feature_count": len(categorical_features),
        "continuous_feature_count": len(X.columns) - len(categorical_features),
        "before": {str(k): int(v) for k, v in class_counts.items()},
        "after": {str(k): int(v) for k, v in after_counts.items()},
    }


__all__ = ["balance_training_data_smotenc", "infer_categorical_features"]
