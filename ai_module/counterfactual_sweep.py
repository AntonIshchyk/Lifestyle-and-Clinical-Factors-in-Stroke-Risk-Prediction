"""
Counterfactual sweep: estimate per-patient change in stroke-risk probability
when lifestyle variables are set to a cessation or moderation reference state.

Run with:
    python ai_module/counterfactual_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

# ---------------------------------------------------------------------------
# SCENARIOS
# Each entry is a dict with:
#   name          – human-readable label
#   state         – "cessation" or "moderation"
#   apply         – callable(X_cf) that mutates X_cf in-place to apply the intervention
#   eligible_mask – callable(X) that returns boolean Series of eligible rows
# ---------------------------------------------------------------------------

def _cessation_totinda(X: pd.DataFrame) -> pd.Series:
    return X["_TOTINDA"] == 2  # currently inactive

def _cessation_bmi5(X: pd.DataFrame) -> pd.Series:
    return X["_BMI5"] > 25.0  # above healthy weight (stored as actual BMI)

def _cessation_drnkwk3(X: pd.DataFrame) -> pd.Series:
    return X["_DRNKWK3"] > 0  # any drinker

def _moderation_drnkwk3(X: pd.DataFrame) -> pd.Series:
    # heavy drinkers: women >7, men >14 drinks/week (stored as actual drinks/week)
    female_heavy = (X["_SEX"] == 2) & (X["_DRNKWK3"] > 7)
    male_heavy   = (X["_SEX"] == 1) & (X["_DRNKWK3"] > 14)
    return female_heavy | male_heavy

def _cessation_ecignow3(X: pd.DataFrame) -> pd.Series:
    return X["ECIGNOW3"].isin([2, 3])  # current vapers (every day or some days)

def _cessation_marijan1(X: pd.DataFrame) -> pd.Series:
    return X["MARIJAN1"] > 0  # current users

def _cessation_ssbsugr2(X: pd.DataFrame) -> pd.Series:
    return X["SSBSUGR2"] > 0  # any consumption (888 recoded to 0)

def _cessation_ssbfrut3(X: pd.DataFrame) -> pd.Series:
    return X["SSBFRUT3"] > 0

def _cessation_smoking_composite(X: pd.DataFrame) -> pd.Series:
    return X["_SMOKER3"].isin([1, 2])  # current daily or some-days smoker

SCENARIOS: list[dict] = [
    # ---- _TOTINDA ----
    {
        "name": "_TOTINDA",
        "state": "cessation",
        "eligible_mask": _cessation_totinda,
        "apply": lambda X: X.assign(**{"_TOTINDA": 1}),
    },

    # ---- _BMI5 ----
    {
        "name": "_BMI5",
        "state": "cessation",
        "eligible_mask": _cessation_bmi5,
        "apply": lambda X: X.assign(**{"_BMI5": 25.0}),
    },
    {
        "name": "_BMI5",
        "state": "moderation",
        "eligible_mask": _cessation_bmi5,
        "apply": lambda X: X.assign(**{"_BMI5": (X["_BMI5"] * 0.95).round()}),
    },

    # ---- _DRNKWK3 ----
    {
        "name": "_DRNKWK3",
        "state": "cessation",
        "eligible_mask": _cessation_drnkwk3,
        "apply": lambda X: X.assign(**{"_DRNKWK3": 0}),
    },
    {
        "name": "_DRNKWK3",
        "state": "moderation",
        "eligible_mask": _moderation_drnkwk3,
        "apply": lambda X: X.assign(**{
            "_DRNKWK3": np.where(
                X["_SEX"] == 2,
                X["_DRNKWK3"].clip(upper=7),
                X["_DRNKWK3"].clip(upper=14),
            )
        }),
    },

    # ---- ECIGNOW3 ----
    {
        "name": "ECIGNOW3",
        "state": "cessation",
        "eligible_mask": _cessation_ecignow3,
        "apply": lambda X: X.assign(**{"ECIGNOW3": 1}),  # 1 = never used
    },

    # ---- MARIJAN1 ----
    {
        "name": "MARIJAN1",
        "state": "cessation",
        "eligible_mask": _cessation_marijan1,
        "apply": lambda X: X.assign(**{"MARIJAN1": 0}),
    },
    {
        "name": "MARIJAN1",
        "state": "moderation",
        "eligible_mask": _cessation_marijan1,
        "apply": lambda X: X.assign(**{"MARIJAN1": (X["MARIJAN1"] / 2).round()}),
    },

    # ---- SSBSUGR2 ----
    {
        "name": "SSBSUGR2",
        "state": "cessation",
        "eligible_mask": _cessation_ssbsugr2,
        "apply": lambda X: X.assign(**{"SSBSUGR2": 0}),
    },

    # ---- SSBFRUT3 ----
    {
        "name": "SSBFRUT3",
        "state": "cessation",
        "eligible_mask": _cessation_ssbfrut3,
        "apply": lambda X: X.assign(**{"SSBFRUT3": 0}),
    },

    # ---- Smoking history composite ----
    # Cessation: "quits today" — set _SMOKER3 → 3 (former), LCSLAST_ → current age,
    #            _LCSYQTS → 0; LCSFIRST / LCSNUMC_ / _LCSYSMK preserved (history intact)
    {
        "name": "smoking_composite",
        "state": "cessation",
        "eligible_mask": _cessation_smoking_composite,
        "apply": lambda X: X.assign(**{
            "_SMOKER3":  3,
            "LCSLAST_":  X["_AGE80"],
            "_LCSYQTS":  0,
        }),
    },
    # Moderation: halve daily cigarettes and total years smoked
    {
        "name": "smoking_composite",
        "state": "moderation",
        "eligible_mask": _cessation_smoking_composite,
        "apply": lambda X: X.assign(**{
            "LCSNUMC_": (X["LCSNUMC_"] / 2).round(),
            "_LCSYSMK": (X["_LCSYSMK"] / 2).round(),
        }),
    },
]


# ---------------------------------------------------------------------------
# PHASE 2 — Data and models
# ---------------------------------------------------------------------------

import joblib
from sklearn.model_selection import train_test_split

MODELS_DIR  = ROOT / "ai_module" / "models"
DATASET_DIR = ROOT / "backend" / "datasets"
TARGET_COL  = "CVDSTRK3"

MODEL_KEYS = [
    ("random_forest", "lifestyle"),
    ("random_forest", "combined"),
    ("xgboost",       "lifestyle"),
    ("xgboost",       "combined"),
    ("lightgbm",      "lifestyle"),
    ("lightgbm",      "combined"),
]

def load_test_set(condition: str) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce the exact held-out test split used during training."""
    df = pd.read_parquet(DATASET_DIR / f"{condition}_with_uncertain.parquet")
    X = df.drop(columns=[TARGET_COL]).apply(pd.to_numeric, errors="coerce")
    y = df[TARGET_COL]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test, y_test

def load_models() -> dict:
    models = {}
    for algo, condition in MODEL_KEYS:
        name = f"{algo}_weighted_{condition}_with_uncertain__final_best_balance"
        models[(algo, condition)] = joblib.load(MODELS_DIR / f"{name}.pkl")
    return models


if __name__ == "__main__":
    print(f"Total scenarios defined: {len(SCENARIOS)}\n")
    for s in SCENARIOS:
        print(f"  [{s['state']:10s}]  {s['name']}")

    print("\n--- Phase 2: loading data and models ---")
    X_test_lifestyle, y_test_lifestyle = load_test_set("lifestyle")
    X_test_combined,  y_test_combined  = load_test_set("combined")
    print(f"Lifestyle test set : {X_test_lifestyle.shape}")
    print(f"Combined  test set : {X_test_combined.shape}")

    models = load_models()
    print(f"\nLoaded {len(models)} models:")
    for (algo, condition), clf in models.items():
        print(f"  {algo:15s}  {condition}")

    print("\n--- Sanity check: LightGBM lifestyle predict_proba[:5] ---")
    clf = models[("lightgbm", "lifestyle")]
    probs = clf.predict_proba(X_test_lifestyle)[:, 1]
    print(f"  Shape : {probs.shape}")
    print(f"  Values: {probs[:5].round(4)}")

    print("\n--- Phase 3: baselines ---")
    X_test = {"lifestyle": X_test_lifestyle, "combined": X_test_combined}
    baselines = {}
    for (algo, condition), clf in models.items():
        p0 = clf.predict_proba(X_test[condition])[:, 1]
        baselines[(algo, condition)] = p0
        print(f"  {algo:15s}  {condition:10s}  mean p0 = {p0.mean():.4f}")

    print("\n--- Phase 4: intervention sweep ---")
    def run_scenario(scenario: dict, clf, X: pd.DataFrame, p0: np.ndarray) -> dict:
        mask = scenario["eligible_mask"](X).fillna(False).to_numpy(dtype=bool)
        n_eligible = mask.sum()
        if n_eligible == 0:
            return {"n_eligible": 0, "delta_p": np.array([])}
        X_eligible = X[mask]
        X_cf = scenario["apply"](X_eligible)
        p1 = clf.predict_proba(X_cf)[:, 1]
        delta_p = p1 - p0[mask]
        return {"n_eligible": int(n_eligible), "delta_p": delta_p}

    print("\nSanity check — _TOTINDA cessation on LightGBM lifestyle:")
    test_result = run_scenario(
        SCENARIOS[0],
        models[("lightgbm", "lifestyle")],
        X_test_lifestyle,
        baselines[("lightgbm", "lifestyle")],
    )
    print(f"  n_eligible : {test_result['n_eligible']}")
    print(f"  mean dP    : {test_result['delta_p'].mean():.4f}")
    print(f"  min  dP    : {test_result['delta_p'].min():.4f}")
    print(f"  max  dP    : {test_result['delta_p'].max():.4f}")

    print("\nRunning full sweep (12 scenarios × 6 models)...")
    results = {}
    for scenario in SCENARIOS:
        for (algo, condition), clf in models.items():
            X = X_test[condition]
            p0 = baselines[(algo, condition)]
            key = (scenario["name"], scenario["state"], algo, condition)
            results[key] = run_scenario(scenario, clf, X, p0)

    print(f"Done. Total result entries: {len(results)}")

    print("\n--- Phase 5: aggregation + bootstrap CIs ---")
    RESULTS_DIR = ROOT / "ai_module" / "cf_results"
    RESULTS_DIR.mkdir(exist_ok=True)
    N_BOOTSTRAP = 1000
    rng = np.random.default_rng(42)

    rows = []
    for (name, state, algo, condition), res in results.items():
        dp = res["delta_p"]
        n  = res["n_eligible"]
        if n == 0:
            mean_dp = ci_lo = ci_hi = float("nan")
        else:
            mean_dp = dp.mean()
            boot_means = np.array([
                rng.choice(dp, size=len(dp), replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ])
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        rows.append({
            "scenario":  name,
            "state":     state,
            "algo":      algo,
            "condition": condition,
            "n_eligible": n,
            "mean_dp":   round(mean_dp, 6),
            "ci_lo":     round(ci_lo,   6),
            "ci_hi":     round(ci_hi,   6),
        })

    summary = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "summary.xlsx"
    summary.to_excel(out_path, index=False)
    print(f"Saved to {out_path}")
    print()
    print(summary.to_string(index=False))
