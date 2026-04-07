"""
Test BiLSTM model on 2023 holdout set.
Loads: models/bilstm.pth + models/bilstm_scaler.pkl
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_COLS = ['match_id', 'ball_idx', 'season', 'is_batting_team_winner']
TARGET_COL = 'is_batting_team_winner'


# ─── Device ───────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ─── Model ────────────────────────────────────────────────────────────
class IPLBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


# ─── Dataset ──────────────────────────────────────────────────────────
class MatchSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.labels[idx]])


# ─── Helpers ──────────────────────────────────────────────────────────
def load_test_data():
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'dataset', 'test_data.csv'))
    df = df.fillna(0)
    return df


def build_sequences(df, features, seq_length, scaler):
    df_scaled = pd.DataFrame(scaler.transform(df[features]), columns=features, index=df.index)
    sequences, labels = [], []
    for (mid, inn), group in df.groupby(['match_id', 'innings']):
        seq = df_scaled.loc[group.index].values
        label = group[TARGET_COL].iloc[-1]
        if len(seq) >= seq_length:
            seq = seq[-seq_length:]
        else:
            pad = np.zeros((seq_length - len(seq), len(features)))
            seq = np.vstack([pad, seq])
        sequences.append(seq)
        labels.append(int(label))
    return sequences, labels


def test(hidden_dim=128, num_layers=2, dropout=0.3, seq_length=120, batch_size=64):
    model_path = os.path.join(PROJECT_ROOT, 'models', 'bilstm.pth')
    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'bilstm_scaler.pkl')

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found at {model_path}. Train the model first.")
        return None
    if not os.path.exists(scaler_path):
        print(f"[ERROR] Scaler not found at {scaler_path}. Train the model first.")
        return None

    device = get_device()
    df_test = load_test_data()
    features = [c for c in df_test.columns if c not in EXCLUDE_COLS]

    scaler = joblib.load(scaler_path)
    model = IPLBiLSTM(
        input_dim=len(features), hidden_dim=hidden_dim,
        num_layers=num_layers, dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    test_seqs, test_labels = build_sequences(df_test, features, seq_length, scaler)
    test_dl = DataLoader(MatchSequenceDataset(test_seqs, test_labels), batch_size=batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            logits = model(X_batch.to(device))
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_labels.extend(y_batch.squeeze().numpy().astype(int))

    metrics = {
        'Accuracy': accuracy_score(all_labels, all_preds),
        'Precision': precision_score(all_labels, all_preds),
        'Recall': recall_score(all_labels, all_preds),
        'F1': f1_score(all_labels, all_preds),
    }

    print(f"\n{'─'*40}")
    print(f"  BiLSTM — Test Metrics")
    print(f"{'─'*40}")
    for k, v in metrics.items():
        print(f"  {k:>10}: {v:.4f}")
    print(f"{'─'*40}")

    return metrics


if __name__ == '__main__':
    test()
