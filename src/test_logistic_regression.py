"""
Test Logistic Regression model on 2023 holdout set.
Loads: models/logistic_regression.pkl
"""

import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_COLS = ['match_id', 'ball_idx', 'season', 'is_batting_team_winner']
TARGET_COL = 'is_batting_team_winner'


def load_test_data():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'dataset', 'test_data.csv'))
    df = df.fillna(0)
    return df


def test():
    model_path = os.path.join(PROJECT_ROOT, 'models', 'logistic_regression.pkl')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Train the model first.")
        return None

    model = joblib.load(model_path)
    df_test = load_test_data()
    features = [c for c in df_test.columns if c not in EXCLUDE_COLS]

    X_test = df_test[features]
    y_test = df_test[TARGET_COL]
    preds = model.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds),
        'Recall': recall_score(y_test, preds),
        'F1': f1_score(y_test, preds),
    }

    print(f"\n{'─'*40}")
    print(f"  LOGISTIC REGRESSION — Test Metrics")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")
    print(f"{'─'*40}")

    return metrics


if __name__ == '__main__':
    test()
