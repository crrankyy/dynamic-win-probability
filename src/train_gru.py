"""
GRU Training Script for IPL Win Probability
- Loads configs from configs/gru.yaml
- Sequences grouped by (match_id, innings)
- Expanding-window temporal CV (3 folds)
- Saves best model to models/gru.pth
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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

# ─── Device Selection ─────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# ─── Model ────────────────────────────────────────────────────────────
class IPLGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        last_out = gru_out[:, -1, :]
        out = self.dropout(last_out)
        return self.fc(out).squeeze(-1)

# ─── Dataset ──────────────────────────────────────────────────────────
class MatchSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor([self.labels[idx]])
        )

# ─── Data Loading & Sequencing ────────────────────────────────────────
def load_data(subset='train'):
    path = os.path.join(PROJECT_ROOT, 'dataset', f'{subset}_data.csv')
    df = pd.read_csv(path)
    df = df.fillna(0)
    return df

def get_features(df):
    return [col for col in df.columns if col not in EXCLUDE_COLS]

def build_sequences(df, features, seq_length, scaler=None):
    """Group by (match_id, innings), pad/truncate to seq_length, scale features."""
    if scaler is None:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features, index=df.index)
    else:
        df_scaled = pd.DataFrame(scaler.transform(df[features]), columns=features, index=df.index)

    sequences = []
    labels = []

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

    return sequences, labels, scaler

# ─── Training Loop ────────────────────────────────────────────────────
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device).squeeze()
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_labels.extend(y_batch.squeeze().numpy().astype(int))
    return f1_score(all_labels, all_preds)

# ─── Temporal CV Evaluation ──────────────────────────────────────────
def evaluate_config(df_train, features, config, device):
    fold_scores = []
    seq_length = config.get('sequence_length', 120)

    for train_end, val_season in TEMPORAL_CV_FOLDS:
        train_mask = df_train['season'] <= train_end
        val_mask = df_train['season'] == val_season

        df_tr = df_train[train_mask]
        df_val = df_train[val_mask]

        if len(df_val) == 0:
            continue

        train_seqs, train_labels, scaler = build_sequences(df_tr, features, seq_length)
        val_seqs, val_labels, _ = build_sequences(df_val, features, seq_length, scaler)

        train_ds = MatchSequenceDataset(train_seqs, train_labels)
        val_ds = MatchSequenceDataset(val_seqs, val_labels)

        train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

        model = IPLGRU(
            input_dim=len(features),
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()

        best_val_f1 = 0
        patience_counter = 0
        patience = 5

        for epoch in range(config['epochs']):
            loss = train_one_epoch(model, train_dl, optimizer, criterion, device)
            val_f1 = evaluate(model, val_dl, device)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                print(f"      Epoch {epoch+1}/{config['epochs']}: loss={loss:.4f}, val_f1={val_f1:.4f}")

            if patience_counter >= patience:
                print(f"      Early stopping at epoch {epoch+1}")
                break

        fold_scores.append(best_val_f1)
        print(f"    Fold (≤{train_end} / {val_season}): Best F1 = {best_val_f1:.4f}")

    return np.mean(fold_scores) if fold_scores else 0.0

# ─── Main ─────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Using device: {device}\n")

    config_path = os.path.join(PROJECT_ROOT, 'configs', 'gru.yaml')
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
        mean_f1 = evaluate_config(df_train, features, config, device)
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
    seq_length = best_config.get('sequence_length', 120)
    train_seqs, train_labels, scaler = build_sequences(df_train, features, seq_length)

    train_ds = MatchSequenceDataset(train_seqs, train_labels)
    train_dl = DataLoader(train_ds, batch_size=best_config['batch_size'], shuffle=True)

    final_model = IPLGRU(
        input_dim=len(features),
        hidden_dim=best_config['hidden_dim'],
        num_layers=best_config['num_layers'],
        dropout=best_config['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(best_config['epochs']):
        loss = train_one_epoch(final_model, train_dl, optimizer, criterion, device)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{best_config['epochs']}: loss={loss:.4f}")

    # Save model + scaler
    os.makedirs(os.path.join(PROJECT_ROOT, 'models'), exist_ok=True)
    model_path = os.path.join(PROJECT_ROOT, 'models', 'gru.pth')
    torch.save(final_model.state_dict(), model_path)

    scaler_path = os.path.join(PROJECT_ROOT, 'models', 'gru_scaler.pkl')
    import joblib
    joblib.dump(scaler, scaler_path)
    print(f"Model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}\n")

    # ── Evaluate on 2023 holdout ──
    print("Evaluating on 2023 holdout test set...")
    df_test = load_data('test')
    test_seqs, test_labels, _ = build_sequences(df_test, features, seq_length, scaler)
    test_ds = MatchSequenceDataset(test_seqs, test_labels)
    test_dl = DataLoader(test_ds, batch_size=best_config['batch_size'], shuffle=False)

    final_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_dl:
            X_batch = X_batch.to(device)
            logits = final_model(X_batch)
            preds = (torch.sigmoid(logits) >= 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds)
            all_labels.extend(y_batch.squeeze().numpy().astype(int))

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print(f"\n{'─'*40}")
    print(f"  GRU — Test Metrics")
    print(f"{'─'*40}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"{'─'*40}")

if __name__ == '__main__':
    main()
