"""
Predict the winner of a single fight using the saved best model.

Usage:
    python src/predict.py --model models/xgboost.pkl --red-wins 15 --blue-wins 10 ...

For a quick demo with synthetic values, just run:
    python src/predict.py
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def predict_fight(model_path: Path, features: dict) -> dict:
    pipe = joblib.load(model_path)
    X = pd.DataFrame([features])
    prob = pipe.predict_proba(X)[0]
    pred = pipe.predict(X)[0]
    return {
        "prediction": "Red" if pred == 1 else "Blue",
        "red_win_prob": round(float(prob[1]), 4),
        "blue_win_prob": round(float(prob[0]), 4),
    }


def demo():
    """Run a quick demo with made-up feature values."""
    model_path = MODELS_DIR / "xgboost.pkl"
    if not model_path.exists():
        print("No saved model found. Run src/train.py first.")
        return

    sample_features = {
        "age_diff": 2,
        "height_diff": 3.0,
        "reach_diff": 5.0,
        "weight_diff": 0.0,
        "sig_str_pct_diff": 0.05,
        "td_pct_diff": 0.10,
        "sub_att_diff": 0.2,
        "kd_diff": 0.1,
        "opp_sig_str_pct_diff": -0.02,
        "wins_diff": 3,
        "losses_diff": -1,
        "ko_wins_diff": 1,
        "sub_wins_diff": 0,
        "dec_wins_diff": 2,
        "rounds_fought_diff": 5,
        "title_bouts_diff": 0,
        "win_streak_diff": 1,
        "lose_streak_diff": 0,
        "longest_streak_diff": 2,
        "weight_class": "lightweight",
        "gender": "male",
        "r_stance": "orthodox",
        "b_stance": "southpaw",
        "title_bout": 0,
    }

    result = predict_fight(model_path, sample_features)
    print("Demo prediction (synthetic fighter stats):")
    print(f"  Predicted winner : {result['prediction']}")
    print(f"  Red win prob     : {result['red_win_prob']:.1%}")
    print(f"  Blue win prob    : {result['blue_win_prob']:.1%}")


if __name__ == "__main__":
    demo()
