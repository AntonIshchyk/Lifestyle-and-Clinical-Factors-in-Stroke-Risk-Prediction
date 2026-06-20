"""
Visualize counterfactual sweep results as a grouped horizontal bar chart.

Run with:
    python ai_module/cf_visualize.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT       = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "ai_module" / "cf_results"

# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

df = pd.read_excel(RESULTS_DIR / "summary.xlsx")

# ---------------------------------------------------------------------------
# Display labels for scenarios (name + state → readable label)
# ---------------------------------------------------------------------------

LABELS = {
    ("_TOTINDA",           "cessation"):   "Become physically active\n(cessation)",
    ("_BMI5",              "cessation"):   "BMI → 25\n(cessation)",
    ("_BMI5",              "moderation"):  "BMI −5%\n(moderation)",
    ("_DRNKWK3",           "cessation"):   "Alcohol\n(cessation)",
    ("_DRNKWK3",           "moderation"):  "Alcohol\n(moderation)",
    ("ECIGNOW3",           "cessation"):   "E-cigarettes\n(cessation)",
    ("MARIJAN1",           "cessation"):   "Cannabis\n(cessation)",
    ("MARIJAN1",           "moderation"):  "Cannabis −50%\n(moderation)",
    ("SSBSUGR2",           "cessation"):   "Sugary drinks\n(cessation)",
    ("SSBFRUT3",           "cessation"):   "Fruit juice\n(cessation)",
    ("smoking_composite",  "cessation"):   "Smoking quit today\n(cessation)",
    ("smoking_composite",  "moderation"):  "Smoking −50%\n(moderation)",
}

ALGO_TITLES = {
    "random_forest": "Random Forest",
    "xgboost":       "XGBoost",
    "lightgbm":      "LightGBM",
}

ALGOS = ["random_forest", "xgboost", "lightgbm"]

# Scenario order (bottom → top on horizontal chart)
SCENARIO_ORDER = list(LABELS.keys())

COLOR_LIFESTYLE = "#2196F3"   # blue
COLOR_COMBINED  = "#FF9800"   # orange

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(18, 11), sharey=True, sharex=True)
fig.subplots_adjust(wspace=0.08)

bar_height  = 0.35
y_positions = np.arange(len(SCENARIO_ORDER))

for ax, algo in zip(axes, ALGOS):
    for i, (name, state) in enumerate(SCENARIO_ORDER):
        label = LABELS[(name, state)]
        for j, condition in enumerate(["lifestyle", "combined"]):
            row = df[
                (df["scenario"] == name) &
                (df["state"]    == state) &
                (df["algo"]     == algo) &
                (df["condition"]== condition)
            ]
            if row.empty:
                continue

            mean_dp = row["mean_dp"].values[0]
            ci_lo   = row["ci_lo"].values[0]
            ci_hi   = row["ci_hi"].values[0]
            err_lo  = mean_dp - ci_lo
            err_hi  = ci_hi  - mean_dp

            y = y_positions[i] + (0.5 - j) * bar_height * 0.9

            color = COLOR_LIFESTYLE if condition == "lifestyle" else COLOR_COMBINED

            ax.barh(
                y, mean_dp,
                height=bar_height,
                color=color, alpha=0.85,
                xerr=[[err_lo], [err_hi]],
                error_kw={"elinewidth": 1.8, "capsize": 3, "ecolor": "black"},
            )

    # Reference line at 0
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    ax.set_title(ALGO_TITLES[algo], fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Mean ΔP (change in stroke probability)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Y-axis labels only on leftmost panel
axes[0].set_yticks(y_positions)
axes[0].set_yticklabels(
    [LABELS[k] for k in SCENARIO_ORDER],
    fontsize=9,
)
axes[0].tick_params(axis="y", length=0)

# Legend
lifestyle_patch = mpatches.Patch(color=COLOR_LIFESTYLE, alpha=0.85, label="Lifestyle features only")
combined_patch  = mpatches.Patch(color=COLOR_COMBINED,  alpha=0.85, label="Combined features")
fig.legend(
    handles=[lifestyle_patch, combined_patch],
    loc="lower center",
    ncol=2,
    fontsize=10,
    frameon=False,
    bbox_to_anchor=(0.5, -0.02),
)

fig.suptitle(
    "Counterfactual Simulation: Mean Change in Predicted Stroke Probability\nper Lifestyle Intervention (held-out test set, n = 91,244)",
    fontsize=13, y=1.01,
)

out_path = RESULTS_DIR / "counterfactual_chart.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
