"""
EDA script — run as a plain Python script or convert to notebook with:
    jupytext --to notebook notebooks/01_eda.py

Produces figures in reports/figures/eda_*.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import load_raw
from preprocessing import build_features, NUMERIC_DIFF_PAIRS

FIGURES = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = load_raw()
print(f"Shape: {df.shape}")
print(f"\nColumns:\n{df.columns.tolist()}")
print(f"\nMissing (%):\n{(df.isnull().mean() * 100).sort_values(ascending=False).head(20)}")

# ── 2. Target distribution ────────────────────────────────────────────────────
win_counts = df["Winner"].value_counts()
fig, ax = plt.subplots(figsize=(6, 4))
win_counts.plot(kind="bar", ax=ax, color=["#e74c3c", "#3498db", "#95a5a6"])
ax.set_title("Fight Outcomes Distribution")
ax.set_xlabel("Winner")
ax.set_ylabel("Count")
ax.tick_params(axis="x", rotation=0)
for p in ax.patches:
    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom", fontsize=9)
fig.tight_layout()
fig.savefig(FIGURES / "eda_winner_dist.png", dpi=120)
plt.close(fig)
print("\nSaved: eda_winner_dist.png")

# ── 3. Fights by weight class ─────────────────────────────────────────────────
if "weight_class" in df.columns:
    wc = df["weight_class"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(9, 5))
    wc.plot(kind="barh", ax=ax)
    ax.set_title("Fights by Weight Class (Top 15)")
    ax.set_xlabel("Count")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_weight_class.png", dpi=120)
    plt.close(fig)
    print("Saved: eda_weight_class.png")

# ── 4. Win method distribution ────────────────────────────────────────────────
if "finish" in df.columns:
    fig, ax = plt.subplots(figsize=(8, 4))
    df["finish"].value_counts().head(10).plot(kind="bar", ax=ax)
    ax.set_title("Fight Finish Methods")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_finish_method.png", dpi=120)
    plt.close(fig)
    print("Saved: eda_finish_method.png")

# ── 5. Numeric feature distributions (diff features) ─────────────────────────
X, y = build_features(df)
print(f"\nFeature matrix: {X.shape}")

num_cols = X.select_dtypes(include="number").columns.tolist()
n = len(num_cols)
ncols = 4
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    ax = axes[i]
    ax.hist(X[col].dropna(), bins=40, color="#3498db", edgecolor="white", linewidth=0.3)
    ax.set_title(col, fontsize=8)
    ax.set_xlabel("")
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)
fig.suptitle("Distribution of Difference Features (Red − Blue)", fontsize=12)
fig.tight_layout()
fig.savefig(FIGURES / "eda_feature_distributions.png", dpi=100)
plt.close(fig)
print("Saved: eda_feature_distributions.png")

# ── 6. Correlation heatmap ────────────────────────────────────────────────────
corr = X[num_cols].corr()
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax, linewidths=0.3)
ax.set_title("Feature Correlation Matrix", fontsize=13)
fig.tight_layout()
fig.savefig(FIGURES / "eda_correlation.png", dpi=100)
plt.close(fig)
print("Saved: eda_correlation.png")

# ── 7. Target correlation ─────────────────────────────────────────────────────
target_corr = X[num_cols].corrwith(y.astype(float)).sort_values()
fig, ax = plt.subplots(figsize=(7, max(4, len(target_corr) * 0.35)))
colors = ["#e74c3c" if v > 0 else "#3498db" for v in target_corr]
target_corr.plot(kind="barh", ax=ax, color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Feature Correlation with Red-Win Label")
ax.set_xlabel("Pearson r")
fig.tight_layout()
fig.savefig(FIGURES / "eda_target_correlation.png", dpi=120)
plt.close(fig)
print("Saved: eda_target_correlation.png")

# ── 8. Win rate by stance matchup ─────────────────────────────────────────────
if "r_stance" in X.columns and "b_stance" in X.columns:
    stance_df = X[["r_stance", "b_stance"]].copy()
    stance_df["target"] = y.values
    matchup = stance_df.groupby(["r_stance", "b_stance"])["target"].agg(["mean", "count"])
    matchup = matchup[matchup["count"] >= 20].reset_index()
    matchup.columns = ["r_stance", "b_stance", "red_win_rate", "n"]
    matchup["label"] = matchup["r_stance"] + " vs " + matchup["b_stance"]
    matchup = matchup.sort_values("red_win_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, len(matchup) * 0.4)))
    bars = ax.barh(matchup["label"], matchup["red_win_rate"], color="#e74c3c", alpha=0.75)
    ax.axvline(0.5, color="black", lw=1, ls="--")
    ax.set_title("Red Win Rate by Stance Matchup (n ≥ 20)")
    ax.set_xlabel("Red Win Rate")
    ax.set_xlim(0, 1)
    for bar, n in zip(bars, matchup["n"]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"n={n}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "eda_stance_winrate.png", dpi=120)
    plt.close(fig)
    print("Saved: eda_stance_winrate.png")

print("\nEDA complete. All figures in reports/figures/")
