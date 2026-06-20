from __future__ import annotations

import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer


def balance_training_data_smote_tomek(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_ratio: float = 1.0,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Balance the training fold with SMOTE followed by Tomek-link cleaning."""
    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be in the interval (0, 1].")

    X = X.reset_index(drop=True)
    y = pd.Series(y, name=y.name).reset_index(drop=True)
    class_counts = y.value_counts().sort_index()
    missing_values = int(X.isna().sum().sum())

    if len(class_counts) < 2:
        return X.copy(), y.copy(), {
            "strategy": "none",
            "before": {str(k): int(v) for k, v in class_counts.items()},
            "after": {str(k): int(v) for k, v in class_counts.items()},
            "imputed_missing_values": missing_values,
        }

    minority_count = int(class_counts.min())
    if minority_count < 2:
        return X.copy(), y.copy(), {
            "strategy": "none_minority_too_small_for_smote_tomek",
            "target_ratio": target_ratio,
            "before": {str(k): int(v) for k, v in class_counts.items()},
            "after": {str(k): int(v) for k, v in class_counts.items()},
            "imputed_missing_values": missing_values,
        }

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    smote = SMOTE(
        sampling_strategy=target_ratio,
        random_state=random_state,
        k_neighbors=min(k_neighbors, minority_count - 1),
    )
    sampler = SMOTETomek(
        sampling_strategy=target_ratio,
        random_state=random_state,
        smote=smote,
        n_jobs=-1,
    )
    X_balanced, y_balanced = sampler.fit_resample(X_imputed, y)
    y_balanced = pd.Series(y_balanced, name=y.name)
    after_counts = y_balanced.value_counts().sort_index()

    return X_balanced.reset_index(drop=True), y_balanced.reset_index(drop=True), {
        "strategy": "smote_tomek",
        "target_ratio": target_ratio,
        "k_neighbors": min(k_neighbors, minority_count - 1),
        "imputed_missing_values": missing_values,
        "before": {str(k): int(v) for k, v in class_counts.items()},
        "after": {str(k): int(v) for k, v in after_counts.items()},
    }


__all__ = ["balance_training_data_smote_tomek"]
