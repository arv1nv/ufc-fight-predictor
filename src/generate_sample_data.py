"""
Generate a synthetic UFC dataset that mirrors the schema of the Kaggle
ufc-master.csv. Used for CI / testing without the real data file.

Run:  python src/generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT = RAW_DIR / "ufc-master.csv"

RANDOM_STATE = 42
N = 5_000

rng = np.random.default_rng(RANDOM_STATE)

WEIGHT_CLASSES = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight",
    "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight",
    "Women's Strawweight", "Women's Flyweight",
]
STANCES = ["Orthodox", "Southpaw", "Switch", "Open Stance"]
FINISHES = ["KO/TKO", "Submission", "Decision - Unanimous", "Decision - Split", "TKO - Doctor's Stoppage"]


def _corner_stats(prefix, n, rng):
    wins = rng.integers(0, 30, n)
    losses = rng.integers(0, 15, n)
    return {
        f"{prefix}_age": rng.integers(22, 40, n),
        f"{prefix}_Height_cms": rng.normal(175, 8, n).round(1),
        f"{prefix}_Reach_cms": rng.normal(178, 8, n).round(1),
        f"{prefix}_Weight_lbs": rng.normal(170, 35, n).round(1),
        f"{prefix}_Stance": rng.choice(STANCES, n),
        f"{prefix}_wins": wins,
        f"{prefix}_losses": losses,
        f"{prefix}_current_win_streak": rng.integers(0, 8, n),
        f"{prefix}_current_lose_streak": rng.integers(0, 4, n),
        f"{prefix}_longest_win_streak": rng.integers(0, 12, n),
        f"{prefix}_win_by_KO_TKO": rng.integers(0, 10, n),
        f"{prefix}_win_by_Submission": rng.integers(0, 8, n),
        f"{prefix}_win_by_Decision_Unanimous": rng.integers(0, 12, n),
        f"{prefix}_total_rounds_fought": rng.integers(1, 50, n),
        f"{prefix}_total_title_bouts": rng.integers(0, 5, n),
        f"{prefix}_avg_SIG_STR_pct": rng.uniform(0.3, 0.7, n).round(3),
        f"{prefix}_avg_TD_pct": rng.uniform(0.2, 0.8, n).round(3),
        f"{prefix}_avg_SUB_ATT": rng.uniform(0, 2, n).round(2),
        f"{prefix}_avg_KD": rng.uniform(0, 0.5, n).round(2),
        f"{prefix}_avg_opp_SIG_STR_pct": rng.uniform(0.3, 0.6, n).round(3),
    }


rows = {}
rows.update(_corner_stats("R", N, rng))
rows.update(_corner_stats("B", N, rng))

rows["weight_class"] = rng.choice(WEIGHT_CLASSES, N)
rows["gender"] = np.where(
    np.isin(rows["weight_class"], ["Women's Strawweight", "Women's Flyweight"]),
    "FEMALE", "MALE"
)
rows["title_bout"] = rng.choice([True, False], N, p=[0.05, 0.95])
rows["finish"] = rng.choice(FINISHES, N)
rows["date"] = pd.date_range("2000-01-01", periods=N, freq="D").astype(str)

# Winner: slightly favour Red corner (home-octagon advantage proxy)
red_wins_base = rng.random(N) < 0.54
rows["Winner"] = np.where(red_wins_base, "Red", "Blue")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print(f"Synthetic dataset written → {OUT}  ({len(df):,} rows)")
