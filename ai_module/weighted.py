from __future__ import annotations

import pandas as pd
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


def balance_training_data_weighted(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_ratio: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Keep the training fold intact and return balanced sample weights."""
    del random_state
    if target_ratio != 1.0:
        raise ValueError("weighted balancing currently supports target_ratio=1.0 only.")

    X = X.reset_index(drop=True)
    y = pd.Series(y, name=y.name).reset_index(drop=True)
    class_counts = y.value_counts().sort_index()

    if len(class_counts) < 2:
        return X.copy(), y.copy(), {
            "strategy": "none",
            "before": {str(k): int(v) for k, v in class_counts.items()},
            "after": {str(k): int(v) for k, v in class_counts.items()},
            "class_weights": {str(k): 1.0 for k in class_counts.index},
            "fit_params": {},
        }

    classes = class_counts.index.to_numpy()
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y.to_numpy(),
    )
    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y.to_numpy(),
    )

    return X.copy(), y.copy(), {
        "strategy": "weighted",
        "target_ratio": target_ratio,
        "before": {str(k): int(v) for k, v in class_counts.items()},
        "after": {str(k): int(v) for k, v in class_counts.items()},
        "class_weights": {
            str(class_value): float(weight)
            for class_value, weight in zip(classes, class_weight_values)
        },
        "fit_params": {"sample_weight": sample_weight},
    }


__all__ = ["balance_training_data_weighted"]
