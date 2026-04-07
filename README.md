# Dynamic Win Probability — IPL

A machine learning pipeline that predicts **ball-by-ball win probabilities** for IPL matches using five model architectures, evaluated on the 2023 season holdout.

## Models

| Model | Type | Output |
|-------|------|--------|
| Logistic Regression | Sklearn | `models/logistic_regression.pkl` |
| Random Forest | Sklearn | `models/random_forest.pkl` |
| XGBoost | Gradient Boosting | `models/xgboost.pkl` |
| BiLSTM | PyTorch (sequence) | `models/bilstm.pth` + `bilstm_scaler.pkl` |
| GRU | PyTorch (sequence) | `models/gru.pth` + `gru_scaler.pkl` |

## Project Structure

```
├── dataset/
│   ├── ipl_ball_by_ball.csv      # Raw source data
│   ├── train_data.csv            # Seasons 2008–2022 (generated)
│   └── test_data.csv             # Season 2023 (generated)
├── configs/
│   ├── logistic_regression.yaml
│   ├── random_forest.yaml
│   ├── xgboost.yaml
│   ├── bilstm.yaml
│   └── gru.yaml
├── src/
│   ├── feature_extraction.py     # Build train/test CSVs from raw data
│   ├── train_<model>.py          # One training script per model
│   ├── test_<model>.py           # One test script per model
│   ├── test_all_models.py        # Run all tests + print comparison table
│   └── simulate_match.py         # Live ball-by-ball simulation
├── notebooks/
│   └── train_all_models.ipynb    # Colab notebook (CUDA, all 5 models)
└── models/                       # Trained model files (git-ignored)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn xgboost pyyaml joblib rich
# PyTorch (BiLSTM/GRU) — requires Python ≤ 3.12
pip install torch
```

> **Note:** PyTorch does not yet support Python 3.13. BiLSTM and GRU training should be run via the Colab notebook.

## Usage

### 1. Extract Features

```bash
python3 src/feature_extraction.py
```

Generates `dataset/train_data.csv` and `dataset/test_data.csv`.

### 2. Train Models

```bash
python3 src/train_logistic_regression.py
python3 src/train_random_forest.py
python3 src/train_xgboost.py
python3 src/train_bilstm.py   # Requires PyTorch
python3 src/train_gru.py      # Requires PyTorch
```

Each script runs a **3-fold expanding-window temporal CV** over the configs in `configs/`, then retrains the best config on all data ≤ 2022.

| Fold | Train | Validate |
|------|-------|----------|
| 1 | ≤ 2019 | 2020 |
| 2 | ≤ 2020 | 2021 |
| 3 | ≤ 2021 | 2022 |

### 3. Test Models

```bash
python3 src/test_all_models.py   # All models + comparison table
python3 src/test_xgboost.py      # Individual model
```

### 4. Simulate a Match

```bash
python3 src/simulate_match.py
```

Prompts for a 2023 match ID and replays it ball-by-ball for each trained model. Final probability panels persist for side-by-side comparison.

### 5. Colab Training

Upload `dataset/train_data.csv` and `dataset/test_data.csv` to Colab, then run `notebooks/train_all_models.ipynb`. Trained models can be downloaded as a zip file.

## Features (21)

`innings`, `Over`, `Balls Remaining`, `Present Score`, `Wickets Left`, `Target`, `Striker Current Runs`, `Non-Striker Current Runs`, `Partnership Runs`, `Bowler Economy`, `Runs_Last_18_Balls`, `Wickets_Last_18_Balls`, `CRR`, `RRR`, `RRR/CRR Ratio`, `Powerplay`, `Middle`, `Death`, `Resource Ratio`, `Venue Average 1st Innings Score`, `Relative Target Difficulty`

**Target:** `is_batting_team_winner` (binary)  
**Test set:** 2023 season (17,744 ball-level rows across 73 matches)
