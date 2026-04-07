"""
Match Simulation Script — IPL Win Probability
Simulates a match ball-by-ball for each trained model, displaying
an inline probability dashboard using Rich panels.

Usage:
    python src/simulate_match.py
    (will prompt for match_id)
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import sys
import time
import numpy as np
import pandas as pd
import joblib
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

console = Console()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXCLUDE_COLS = ['match_id', 'ball_idx', 'season', 'is_batting_team_winner']
TARGET_COL = 'is_batting_team_winner'

# ─── Model Registry ──────────────────────────────────────────────────

def load_sklearn_model(name):
    """Load a sklearn/xgboost model from models/<name>.pkl"""
    path = os.path.join(PROJECT_ROOT, 'models', f'{name}.pkl')
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_pytorch_model(name, model_class, features_count):
    """Load a PyTorch model + scaler from models/<name>.pth"""
    import torch
    model_path = os.path.join(PROJECT_ROOT, 'models', f'{name}.pth')
    scaler_path = os.path.join(PROJECT_ROOT, 'models', f'{name}_scaler.pkl')

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None

    device = get_device()
    scaler = joblib.load(scaler_path)

    # Infer architecture from saved state_dict
    state = torch.load(model_path, map_location=device, weights_only=True)
    hidden_dim = state[list(state.keys())[0]].shape[0]

    # Count layers from state dict keys
    layer_keys = [k for k in state.keys() if 'weight_ih_l' in k and 'reverse' not in k]
    num_layers = len(layer_keys)

    model = model_class(
        input_dim=features_count,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.0  # Dropout disabled at inference
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, scaler


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ─── PyTorch Model Definitions (needed for loading) ──────────────────

def get_bilstm_class():
    import torch.nn as nn

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

    return IPLBiLSTM


def get_gru_class():
    import torch.nn as nn

    class IPLGRU(nn.Module):
        def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
            super().__init__()
            self.gru = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_dim, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)

    return IPLGRU


# ─── Inline UI Rendering ─────────────────────────────────────────────

def count_rendered_lines(renderable):
    """Count how many terminal lines a Rich renderable occupies."""
    temp = Console(file=open(os.devnull, 'w'), width=console.width)
    with temp.capture() as capture:
        temp.print(renderable)
    return capture.get().count('\n')


def render_inline(renderable, prev_lines):
    """Overwrite previous output in-place, then print new renderable."""
    if prev_lines > 0:
        # Move cursor up and clear those lines
        sys.stdout.write(f"\033[{prev_lines}A\033[J")
        sys.stdout.flush()
    console.print(renderable)
    return count_rendered_lines(renderable)


def build_ui(row, prob_batting, team1, team2, model_name, batting_team, bowling_team):
    """Build Rich panels for match state + win probability."""
    prob_bowling = 1.0 - prob_batting

    # Determine Top Panel stats
    inn = int(row['innings'])
    ovr = int(row['Over'])
    ball_in_over = int(row['ball_idx']) % 6 + 1

    score = int(row['Present Score'])
    target = int(row['Target'])
    wickets = 10 - int(row['Wickets Left'])

    # Build Text components
    header = Text(f"Innings {inn} | Over {ovr}.{ball_in_over}", style="bold white", justify="center")
    score_text = Text(f"Score: {score}/{wickets}", style="bold cyan", justify="center")
    if target > 0:
        score_text.append(f" | Target: {target}", style="bold red")

    teams_text = Text(justify="center")
    teams_text.append(f"Batting: {batting_team}", style="bold yellow")
    teams_text.append(f"  |  ", style="dim")
    teams_text.append(f"Bowling: {bowling_team}", style="bold blue")

    top_panel = Panel(
        Group(header, score_text, teams_text),
        border_style="blue", title="Match State"
    )

    # Construct Probabilistic weight explicitly tracking Team 1 uniquely
    if inn == 1:
        prob_team1 = prob_batting
    else:
        prob_team1 = 1.0 - prob_batting

    team1_pct = int(prob_team1 * 100)
    team2_pct = 100 - team1_pct

    # Bar construction
    bar_width = 40
    t1_chars = int(bar_width * prob_team1)
    t2_chars = bar_width - t1_chars

    bar_string = ("█" * t1_chars) + ("░" * t2_chars)
    prob_text = Text()
    prob_text.append(f"{team1}: {team1_pct}% ", style="bold green")
    prob_text.append(bar_string, style="magenta")
    prob_text.append(f" {team2_pct}% :{team2}", style="bold red")
    prob_text.stylize("center")

    bottom_panel = Panel(
        Align.center(prob_text),
        border_style="magenta",
        title=f"Dynamic Win Probability — {model_name}"
    )

    return Group(top_panel, bottom_panel)


# ─── Simulation Runners ──────────────────────────────────────────────

def run_simulation_sklearn(model, df_match, features, team1, team2, model_name):
    """Run simulation for sklearn/xgboost models using predict_proba."""
    print(f"\nLaunching Simulation — {model_name}...\n")
    prev_lines = 0

    for i in range(len(df_match)):
        row = df_match.iloc[i]
        row_df = df_match.iloc[[i]]
        X_current = row_df[features]
        prob_batting = model.predict_proba(X_current)[0][1]

        inn = int(row['innings'])
        batting_team = team1 if inn == 1 else team2
        bowling_team = team2 if inn == 1 else team1

        ui = build_ui(row, prob_batting, team1, team2, model_name, batting_team, bowling_team)

        # Overwrite previous frame, keep final frame
        if i < len(df_match) - 1:
            prev_lines = render_inline(ui, prev_lines)
        else:
            # Final ball — render permanently (overwrite animation frame, then print fresh)
            render_inline(ui, prev_lines)

        # The pause logic natively simulating actual Over dynamics
        time.sleep(1 / 6.0)

    print("\nSimulation Complete!\n")


def run_simulation_pytorch(model, scaler, df_match, features, team1, team2, model_name, seq_length=120):
    """Run simulation for PyTorch sequence models (BiLSTM/GRU)."""
    import torch

    device = get_device()

    # Pre-scale all features
    X_scaled = pd.DataFrame(
        scaler.transform(df_match[features]),
        columns=features, index=df_match.index
    )

    print(f"\nLaunching Simulation — {model_name}...\n")
    prev_lines = 0

    for i in range(len(df_match)):
        row = df_match.iloc[i]

        # Build sequence up to current ball
        seq = X_scaled.iloc[:i + 1].values

        # Pad or truncate to seq_length
        if len(seq) >= seq_length:
            seq = seq[-seq_length:]
        else:
            pad = np.zeros((seq_length - len(seq), len(features)))
            seq = np.vstack([pad, seq])

        X_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)  # (1, seq_len, features)

        with torch.no_grad():
            logits = model(X_tensor)
            prob_batting = torch.sigmoid(logits).cpu().item()

        inn = int(row['innings'])
        batting_team = team1 if inn == 1 else team2
        bowling_team = team2 if inn == 1 else team1

        ui = build_ui(row, prob_batting, team1, team2, model_name, batting_team, bowling_team)

        if i < len(df_match) - 1:
            prev_lines = render_inline(ui, prev_lines)
        else:
            render_inline(ui, prev_lines)

        time.sleep(1 / 6.0)

    print("\nSimulation Complete!\n")


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    # Load test data
    test_path = os.path.join(PROJECT_ROOT, 'dataset', 'test_data.csv')
    df_test = pd.read_csv(test_path)
    df_test = df_test.fillna(0)

    features = [c for c in df_test.columns if c not in EXCLUDE_COLS]

    # Load raw data for team names
    raw_path = os.path.join(PROJECT_ROOT, 'dataset', 'ipl_ball_by_ball.csv')
    df_raw = pd.read_csv(raw_path)

    # Show available match IDs
    match_ids = sorted(df_test['match_id'].unique())
    print(f"\nAvailable Match IDs (2023 season): {len(match_ids)} matches")
    print(f"Range: {match_ids[0]} — {match_ids[-1]}")
    print(f"All IDs: {match_ids}\n")

    # Prompt for match ID
    try:
        match_id = int(input("Enter Match ID to simulate: "))
    except (ValueError, EOFError):
        print("Invalid input. Exiting.")
        return

    # Filter match
    df_match = df_test[df_test['match_id'] == match_id].sort_values(by=['innings', 'ball_idx'])
    if df_match.empty:
        print(f"Error: Match ID {match_id} not found in 2023 test data.")
        return

    # Get team names from raw data
    raw_match = df_raw[df_raw['match_id'] == match_id]
    inn1 = raw_match[raw_match['innings'] == 1]
    if not inn1.empty:
        team1 = str(inn1.iloc[0]['batting_team'])
        team2 = str(inn1.iloc[0]['bowling_team'])
    else:
        team1, team2 = "Team A", "Team B"

    print(f"\n{'='*60}")
    print(f"  Match {match_id}: {team1} vs {team2}")
    print(f"  Total balls: {len(df_match)}")
    print(f"{'='*60}")

    # ── Discover and run available models ──
    models_run = 0

    # 1. Logistic Regression
    lr = load_sklearn_model('logistic_regression')
    if lr:
        run_simulation_sklearn(lr, df_match, features, team1, team2, "Logistic Regression")
        models_run += 1
    else:
        print("\n⚠ Logistic Regression model not found — skipping")

    # 2. Random Forest
    rf = load_sklearn_model('random_forest')
    if rf:
        run_simulation_sklearn(rf, df_match, features, team1, team2, "Random Forest")
        models_run += 1
    else:
        print("\n⚠ Random Forest model not found — skipping")

    # 3. XGBoost
    xgb = load_sklearn_model('xgboost')
    if xgb:
        run_simulation_sklearn(xgb, df_match, features, team1, team2, "XGBoost")
        models_run += 1
    else:
        print("\n⚠ XGBoost model not found — skipping")

    # 4. BiLSTM
    try:
        bilstm_cls = get_bilstm_class()
        bilstm, bilstm_scaler = load_pytorch_model('bilstm', bilstm_cls, len(features))
        if bilstm:
            run_simulation_pytorch(bilstm, bilstm_scaler, df_match, features, team1, team2, "BiLSTM")
            models_run += 1
        else:
            print("\n⚠ BiLSTM model not found — skipping")
    except ImportError:
        print("\n⚠ PyTorch not available — skipping BiLSTM")

    # 5. GRU
    try:
        gru_cls = get_gru_class()
        gru, gru_scaler = load_pytorch_model('gru', gru_cls, len(features))
        if gru:
            run_simulation_pytorch(gru, gru_scaler, df_match, features, team1, team2, "GRU")
            models_run += 1
        else:
            print("\n⚠ GRU model not found — skipping")
    except ImportError:
        print("\n⚠ PyTorch not available — skipping GRU")

    print(f"\n{'='*60}")
    print(f"  Simulation complete for {models_run} model(s)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
