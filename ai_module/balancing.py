from __future__ import annotations

import numpy as np
import pandas as pd


def balance_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    target_ratio: float = 1.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Randomly oversample minority classes in the training fold only."""
    if not 0 < target_ratio <= 1:
        raise ValueError("target_ratio must be in the interval (0, 1].")

    X = X.reset_index(drop=True)
    y = pd.Series(y, name=y.name).reset_index(drop=True)
    class_counts = y.value_counts().sort_index()
    if len(class_counts) < 2:
        return X.copy(), y.copy(), {
            "strategy": "none",
            "before": {str(k): int(v) for k, v in class_counts.items()},
            "after": {str(k): int(v) for k, v in class_counts.items()},
        }

    rng = np.random.default_rng(random_state)
    majority_count = int(class_counts.max())
    target_count = int(round(majority_count * target_ratio))

    balanced_indices = [y.index.to_numpy()]
    for class_value, class_count in class_counts.items():
        if class_count >= target_count:
            continue

        class_indices = y.index[y == class_value].to_numpy()
        sampled_indices = rng.choice(
            class_indices,
            size=target_count - int(class_count),
            replace=True,
        )
        balanced_indices.append(sampled_indices)

    final_indices = np.concatenate(balanced_indices)
    rng.shuffle(final_indices)

    X_balanced = X.iloc[final_indices].reset_index(drop=True)
    y_balanced = y.iloc[final_indices].reset_index(drop=True)
    after_counts = y_balanced.value_counts().sort_index()

    return X_balanced, y_balanced, {
        "strategy": "random_oversampling",
        "target_ratio": target_ratio,
        "before": {str(k): int(v) for k, v in class_counts.items()},
        "after": {str(k): int(v) for k, v in after_counts.items()},
    }


__all__ = ["balance_training_data"]
