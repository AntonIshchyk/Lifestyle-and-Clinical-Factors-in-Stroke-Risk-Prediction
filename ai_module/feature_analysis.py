import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Bias-corrected Cramér's V between two categorical series
def cramers_v(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan
    confusion = pd.crosstab(x, y)
    if confusion.size == 0:
        return np.nan
    
    chi2 = chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum().sum()
    r, k = confusion.shape
    phi2 = max(0, (chi2 / n) - ((r - 1) * (k - 1)) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(r_corr - 1, k_corr - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2 / denom))

# Compute Cramér's V for all feature pairs and return those above threshold.
def find_high_correlation_pairs(
    df: pd.DataFrame,
    threshold: float = 0.85,
    sample_n: int = 50_000,
    random_state: int = 42,
) -> pd.DataFrame:
    df = df.copy()

    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state)

    feature_cols = list(df.columns)
    n = len(feature_cols)

    print(f"Computing Cramér's V for {n} features ({n * (n - 1) // 2:,} pairs)...")

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                v = cramers_v(
                    df[feature_cols[i]].astype("category"),
                    df[feature_cols[j]].astype("category"),
                )
            except Exception:
                v = np.nan
            if not np.isnan(v) and v >= threshold:
                pairs.append({
                    "feature_a": feature_cols[i],
                    "feature_b": feature_cols[j],
                    "cramers_v": round(v, 4),
                })

    if pairs:
        pairs_df = pd.DataFrame(pairs).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    else:
        pairs_df = pd.DataFrame(columns=["feature_a", "feature_b", "cramers_v"])
    print(f"High-correlation pairs (V >= {threshold}): {len(pairs_df)}")
    return pairs_df

__all__ = ["find_high_correlation_pairs"]