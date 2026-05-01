"""
Load and validate the raw UFC dataset.

Expected source: Kaggle dataset "mdabbert/ultimate-ufc-dataset"
  - Download manually: kaggle datasets download mdabbert/ultimate-ufc-dataset
  - Or place ufc-master.csv in data/raw/
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_CSV = RAW_DIR / "ufc-master.csv"


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            "Download from Kaggle: mdabbert/ultimate-ufc-dataset\n"
            "and place ufc-master.csv in data/raw/"
        )
    df = pd.read_csv(path, low_memory=False)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {path.name}")
    return df
