"""
Random Forest Training Script for IPL Win Probability
- Loads configs from configs/random_forest.yaml
- Expanding-window temporal CV (3 folds)
- Saves best model to models/random_forest.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ─── Constants ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_COLS = ['match_id', 'ball_idx', 'season', 'is_batting_team_winner']
TARGET_COL = 'is_batting_team_winner'

TEMPORAL_CV_FOLDS = [
    (2019, 2020),
    (2020, 2021),
    (2021, 2022),
]

# ─── Data Loading ─────────────────────────────────────────────────────
def load_data(subset='train'):
    path = os.path.join(PROJECT_ROOT, 'dataset', f'{subset}_data.csv')
    df = pd.read_csv(path)
    df = df.fillna(0)
    return df

def get_features(df):
    return [col for col in df.columns if col not in EXCLUDE_COLS]

# ─── Temporal CV Evaluation ──────────────────────────────────────────
def evaluate_config(df_train, features, config):
    """Evaluate a single hyperparameter config across all temporal CV folds."""
    fold_scores = []

    for train_end, val_season in TEMPORAL_CV_FOLDS:
        train_mask = df_train['season'] <= train_end
        val_mask = df_train['season'] == val_season

        X_tr, y_tr = df_train.loc[train_mask, features], df_train.loc[train_mask, TARGET_COL]
        X_val, y_val = df_train.loc[val_mask, features], df_train.loc[val_mask, TARGET_COL]

        if len(X_val) == 0:
            continue

        model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **config
        )
        model.fit(X_tr, y_tr)

        preds = model.predict(X_val)
        f1 = f1_score(y_val, preds)
        fold_scores.append(f1)
        print(f"    Fold (≤{train_end} / {val_season}): F1 = {f1:.4f}")

    mean_f1 = np.mean(fold_scores) if fold_scores else 0.0
    return mean_f1

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'random_forest.yaml')
    with open(config_path, 'r') as f:
        all_configs = yaml.safe_load(f)['configs']

    print(f"Loaded {len(all_configs)} hyperparameter configurations.\n")

    df_train = load_data('train')
    features = get_features(df_train)
    print(f"Features ({len(features)}): {features}\n")

    # ── Hyperparameter Search ──
    best_f1 = -1
    best_config = None

    for i, config in enumerate(all_configs):
        print(f"[Config {i+1}/{len(all_configs)}] {config}")
        mean_f1 = evaluate_config(df_train, features, config)
        print(f"  → Mean F1 = {mean_f1:.4f}\n")

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_config = config

    print(f"{'='*60}")
    print(f"Best Config: {best_config}")
    print(f"Best Mean CV F1: {best_f1:.4f}")
    print(f"{'='*60}\n")

    # ── Retrain on all data ≤ 2022 ──
    print("Retraining best model on all data ≤ 2022...")
    X_full = df_train[features]
    y_full = df_train[TARGET_COL]

    final_model = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        **best_config
    )
    final_model.fit(X_full, y_full)

    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
    model_path = os.path.join(PROJECT_ROOT, 'models', 'random_forest.pkl')
    joblib.dump(final_model, model_path)
    print(f"Model saved to {model_path}\n")

    # ── Evaluate on 2023 holdout ──
    print("Evaluating on 2023 holdout test set...")
    df_test = load_data('test')
    X_test = df_test[features]
    y_test = df_test[TARGET_COL]

    preds = final_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print(f"\n{'─'*40}")
    print(f"  RANDOM FOREST — Test Metrics")
    print(f"{'─'*40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"{'─'*40}")

if __name__ == '__main__':
    main()
