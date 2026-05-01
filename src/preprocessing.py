"""
Data cleaning and feature engineering for the UFC dataset.

The Kaggle ufc-master.csv has one row per fight with prefixed columns:
  R_ / B_  for Red / Blue corner fighter stats.

Target: 'Winner' column ('Red' or 'Blue') → binary label 1 = Red wins.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# ── Column groups ─────────────────────────────────────────────────────────────

NUMERIC_DIFF_PAIRS = [
    # (red_col, blue_col, diff_feature_name)
    ("R_age", "B_age", "age_diff"),
    ("R_Height_cms", "B_Height_cms", "height_diff"),
    ("R_Reach_cms", "B_Reach_cms", "reach_diff"),
    ("R_Weight_lbs", "B_Weight_lbs", "weight_diff"),
    ("R_avg_SIG_STR_pct", "B_avg_SIG_STR_pct", "sig_str_pct_diff"),
    ("R_avg_TD_pct", "B_avg_TD_pct", "td_pct_diff"),
    ("R_avg_SUB_ATT", "B_avg_SUB_ATT", "sub_att_diff"),
    ("R_avg_KD", "B_avg_KD", "kd_diff"),
    ("R_avg_opp_SIG_STR_pct", "B_avg_opp_SIG_STR_pct", "opp_sig_str_pct_diff"),
    ("R_wins", "B_wins", "wins_diff"),
    ("R_losses", "B_losses", "losses_diff"),
    ("R_win_by_KO_TKO", "B_win_by_KO_TKO", "ko_wins_diff"),
    ("R_win_by_Submission", "B_win_by_Submission", "sub_wins_diff"),
    ("R_win_by_Decision_Unanimous", "B_win_by_Decision_Unanimous", "dec_wins_diff"),
    ("R_total_rounds_fought", "B_total_rounds_fought", "rounds_fought_diff"),
    ("R_total_title_bouts", "B_total_title_bouts", "title_bouts_diff"),
    ("R_current_win_streak", "B_current_win_streak", "win_streak_diff"),
    ("R_current_lose_streak", "B_current_lose_streak", "lose_streak_diff"),
    ("R_longest_win_streak", "B_longest_win_streak", "longest_streak_diff"),
]

CATEGORICAL_FEATURES = ["weight_class", "gender"]


def _safe_diff(df, r_col, b_col):
    """Return R - B difference, coercing to numeric and ignoring missing pairs."""
    r = pd.to_numeric(df.get(r_col, np.nan), errors="coerce")
    b = pd.to_numeric(df.get(b_col, np.nan), errors="coerce")
    return r - b


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Engineer features and return (X, y).

    y = 1 if Red wins, 0 if Blue wins. Draws are dropped.
    """
    df = df.copy()

    # Drop draws / no-contests
    if "Winner" not in df.columns:
        raise KeyError("'Winner' column not found. Check dataset.")
    df = df[df["Winner"].isin(["Red", "Blue"])].copy()

    y = (df["Winner"] == "Red").astype(int)

    feature_frames = []

    # Difference features (Red − Blue) capture relative advantage
    diff_data = {}
    for r_col, b_col, name in NUMERIC_DIFF_PAIRS:
        diff_data[name] = _safe_diff(df, r_col, b_col)
    feature_frames.append(pd.DataFrame(diff_data, index=df.index))

    # Categorical features
    cat_data = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            cat_data[col] = df[col].astype(str).str.strip().str.lower()
    if cat_data:
        feature_frames.append(pd.DataFrame(cat_data, index=df.index))

    # Red fighter stance advantage (ordinal encoding of stance matchup)
    if "R_Stance" in df.columns and "B_Stance" in df.columns:
        cat_data2 = {
            "r_stance": df["R_Stance"].astype(str).str.lower(),
            "b_stance": df["B_Stance"].astype(str).str.lower(),
        }
        feature_frames.append(pd.DataFrame(cat_data2, index=df.index))

    # Title bout flag
    if "title_bout" in df.columns:
        feature_frames.append(
            pd.DataFrame(
                {"title_bout": df["title_bout"].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0)},
                index=df.index,
            )
        )

    X = pd.concat(feature_frames, axis=1)
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Return a fitted-ready ColumnTransformer for the feature matrix."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    transformers = [
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            num_cols,
        ),
    ]
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_cols,
        ))

    return ColumnTransformer(transformers, remainder="drop")
