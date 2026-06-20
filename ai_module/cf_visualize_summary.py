"""
Forest-plot style summary chart: mean ΔP averaged across all 3 algorithms,
lifestyle vs combined side by side. Single panel, compact for paper.

Run with:
    python ai_module/cf_visualize_summary.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "ai_module" / "cf_results"

df = pd.read_excel(RESULTS_DIR / "summary.xlsx")

LABELS = {
    ("_TOTINDA",          "cessation"):  "Become physically active",
    ("_BMI5",             "cessation"):  "BMI → 25 (cessation)",
    ("_BMI5",             "moderation"): "BMI −5% (moderation)",
    ("_DRNKWK3",          "cessation"):  "Alcohol cessation",
    ("_DRNKWK3",          "moderation"): "Alcohol moderation",
    ("ECIGNOW3",          "cessation"):  "E-cigarettes cessation",
    ("MARIJAN1",          "cessation"):  "Cannabis cessation",
    ("MARIJAN1",          "moderation"): "Cannabis −50% (moderation)",
    ("SSBSUGR2",          "cessation"):  "Sugary drinks cessation",
    ("SSBFRUT3",          "cessation"):  "Fruit juice cessation",
    ("smoking_composite", "cessation"):  "Smoking quit today",
    ("smoking_composite", "moderation"): "Smoking −50% (moderation)",
}

COLOR_LIFESTYLE = "#2196F3"
COLOR_COMBINED  = "#FF9800"
CLIP_X          = 0.020   # x-axis upper clip; outlier annotated manually

# ---------------------------------------------------------------------------
# Average across algorithms
# ---------------------------------------------------------------------------

averaged = (
    df.groupby(["scenario", "state", "condition"])[["mean_dp", "ci_lo", "ci_hi"]]
    .mean()
    .reset_index()
)

# Sort scenarios by mean ΔP averaged across both conditions (most negative → top)
scenario_avg = (
    averaged.groupby(["scenario", "state"])["mean_dp"]
    .mean()
    .reset_index()
    .sort_values("mean_dp")
)
SCENARIO_ORDER = list(zip(scenario_avg["scenario"], scenario_avg["state"]))

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 8))

y_positions = np.arange(len(SCENARIO_ORDER))
offset = 0.18

for i, (name, state) in enumerate(SCENARIO_ORDER):
    for j, condition in enumerate(["lifestyle", "combined"]):
        row = averaged[
            (averaged["scenario"] == name) &
            (averaged["state"]    == state) &
            (averaged["condition"]== condition)
        ]
        if row.empty:
            continue

        mean_dp = row["mean_dp"].values[0]
        ci_lo   = row["ci_lo"].values[0]
        ci_hi   = row["ci_hi"].values[0]

        y     = y_positions[i] + (0.5 - j) * offset * 1.8
        color = COLOR_LIFESTYLE if condition == "lifestyle" else COLOR_COMBINED

        # Clip point to axis limit; annotate if clipped
        clipped = mean_dp > CLIP_X
        plot_x  = min(mean_dp, CLIP_X)
        err_lo  = mean_dp - ci_lo
        err_hi  = min(ci_hi, CLIP_X) - plot_x  # clip upper error bar too

        ax.errorbar(
            plot_x, y,
            xerr=[[err_lo], [err_hi]],
            fmt="o" if not clipped else ">",   # arrow marker when clipped
            color=color,
            markersize=6,
            linewidth=1.5,
            capsize=4,
            capthick=1.8,
            ecolor="black",
            elinewidth=1.8,
            alpha=0.9,
        )

        if clipped:
            ax.annotate(
                f"{mean_dp:.3f}",
                xy=(CLIP_X, y),
                xytext=(CLIP_X + 0.0005, y),
                fontsize=7.5,
                color=color,
                va="center",
            )

ax.axvline(0, color="black", linewidth=0.9, linestyle="--", alpha=0.7)
ax.set_xlim(left=None, right=CLIP_X + 0.004)

ax.set_yticks(y_positions)
ax.set_yticklabels([LABELS.get(k, str(k)) for k in SCENARIO_ORDER], fontsize=10)
ax.set_xlabel(
    "Mean ΔP (change in stroke probability)\naveraged across Random Forest, XGBoost, LightGBM",
    fontsize=10,
)
ax.set_title(
    "Counterfactual Simulation: Average Effect per Lifestyle Intervention\n"
    "(held-out test set, n = 91,244)",
    fontsize=11, pad=12,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", length=0)

# Legend with dot markers (not patches)
ax.plot([], [], "o", color=COLOR_LIFESTYLE, markersize=6, label="Lifestyle features only")
ax.plot([], [], "o", color=COLOR_COMBINED,  markersize=6, label="Combined features")
ax.legend(fontsize=10, frameon=False, loc="lower right")

fig.tight_layout()

out_path = RESULTS_DIR / "counterfactual_summary_chart.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.show()
