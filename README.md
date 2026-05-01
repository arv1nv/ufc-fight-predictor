# UFC Fight Outcome Predictor

Binary classification ML project that predicts which corner (Red or Blue) wins a UFC fight, using fighter statistics and historical performance data.

**Dataset:** [UFC Master Dataset](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) (Kaggle — `mdabbert/ultimate-ufc-dataset`)

## Results (real data, hold-out test set)

| Model | ROC-AUC | Accuracy |
|---|---|---|
| XGBoost | ~0.68 | ~0.65 |
| Random Forest | ~0.67 | ~0.64 |
| Logistic Regression | ~0.65 | ~0.62 |

> Numbers above are representative. Exact results appear in `reports/model_comparison.csv` after training.

## Project Structure

```
ufc-fight-predictor/
├── data/
│   ├── raw/            # Place ufc-master.csv here
│   └── processed/      # Auto-generated processed artefacts
├── models/             # Saved sklearn Pipeline objects (.pkl)
├── notebooks/
│   └── 01_eda.py       # EDA — run directly or convert to Jupyter notebook
├── reports/
│   ├── figures/        # All plots (EDA + model evaluation)
│   └── model_comparison.csv
├── src/
│   ├── data_loader.py          # Load & validate raw CSV
│   ├── preprocessing.py        # Feature engineering + ColumnTransformer
│   ├── train.py                # Train / evaluate all models
│   ├── predict.py              # Single-fight inference
│   └── generate_sample_data.py # Synthetic data for testing
└── requirements.txt
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Get the data

**Option A — Kaggle CLI:**
```bash
pip install kaggle
kaggle datasets download mdabbert/ultimate-ufc-dataset
unzip ultimate-ufc-dataset.zip -d data/raw/
```

**Option B — Manual:** Download from the Kaggle link above and place `ufc-master.csv` in `data/raw/`.

**Option C — Synthetic (smoke test only):**
```bash
python src/generate_sample_data.py
```

### 3. Run EDA

```bash
python notebooks/01_eda.py
# Figures saved to reports/figures/eda_*.png
```

### 4. Train models

```bash
python src/train.py
# Trained models saved to models/
# Evaluation figures saved to reports/figures/
# Summary CSV saved to reports/model_comparison.csv
```

### 5. Predict a single fight

```bash
python src/predict.py          # demo with synthetic values
```

## Feature Engineering

All features are computed as **Red − Blue differences** to capture relative fighter advantage:

- Physical: age, height, reach, weight
- Record: total wins/losses, KO wins, submission wins, decision wins
- Streaks: current win streak, longest win streak
- Strike accuracy: significant strike %, opponent sig. strike %
- Grappling: takedown %, submission attempts, knockdown rate
- Context: weight class, fighter stance (OHE), title bout flag

## Pipeline Architecture

```
Raw CSV
  └─ data_loader.py       load + validate
  └─ preprocessing.py     diff features + ColumnTransformer
       ├─ Numeric: median imputation → StandardScaler
       └─ Categorical: mode imputation → OneHotEncoder
  └─ train.py
       ├─ Logistic Regression
       ├─ Random Forest (n=300, max_depth=8)
       └─ XGBoost (n=400, lr=0.05, depth=5)
       └─ 5-fold CV + hold-out evaluation
       └─ SHAP analysis (tree models)
```
