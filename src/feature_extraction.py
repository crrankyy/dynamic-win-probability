import os
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def normalize_columns(df):
    """Normalize common Cricsheet/Kaggle column names to standard internal names."""
    col_map = {
        'id': 'match_id', 'ID': 'match_id',
        'Season': 'season',
        'Venue': 'venue',
        'BattingTeam': 'batting_team', 'Team1': 'batting_team',
        'BowlingTeam': 'bowling_team', 'Team2': 'bowling_team',
        'WinningTeam': 'winning_team',
        'Innings': 'innings',
        'over': 'overs',
        'ball': 'ballnumber',
        'striker': 'batter', 'batsman': 'batter',
        'non_striker': 'non_striker', 'non-striker': 'non_striker', 'nonstriker': 'non_striker',
        'runs_off_bat': 'batsman_run', 'batsman_runs': 'batsman_run',
        'total_runs': 'total_run',
        'isWicketDelivery': 'is_wicket', 'is_wicket': 'is_wicket',
        'extras_type': 'extra_type'
    }
    df = df.rename(columns=lambda x: col_map.get(x, x))
    
    # Handle ball decimal (e.g. 0.1, 0.2) if overs is missing
    if 'overs' not in df.columns and 'ballnumber' in df.columns:
        df['overs'] = df['ballnumber'].astype(int)
        df['ballnumber'] = ((df['ballnumber'] - df['overs']) * 10).round().astype(int)

    # Calculate total_run if missing
    if 'total_run' not in df.columns:
        df['total_run'] = df['batsman_run'] + df['extras']

    # Calculate winning_team if missing (by comparing totals)
    if 'winning_team' not in df.columns:
        print("Winning Team not found. Calculating winner from innings totals...")
        scores = df.groupby(['match_id', 'innings'])['total_run'].sum().reset_index()
        pivot_scores = scores.pivot(index='match_id', columns='innings', values='total_run')
        teams = df.drop_duplicates(['match_id', 'innings'])[['match_id', 'innings', 'batting_team']]
        pivot_teams = teams.pivot(index='match_id', columns='innings', values='batting_team')
        
        def determine_winner(match_id):
            s1 = pivot_scores.loc[match_id, 1] if 1 in pivot_scores.columns else 0
            s2 = pivot_scores.loc[match_id, 2] if 2 in pivot_scores.columns else 0
            t1 = pivot_teams.loc[match_id, 1] if 1 in pivot_teams.columns else None
            t2 = pivot_teams.loc[match_id, 2] if 2 in pivot_teams.columns else None
            if s2 > s1: return t2
            if s1 > s2: return t1
            return 'Tie'

        match_winners = pd.Series(pivot_scores.index).apply(determine_winner)
        winner_map = pd.DataFrame({'match_id': pivot_scores.index, 'winning_team': match_winners})
        df = df.merge(winner_map, on='match_id', how='left')

    # Create is_wicket if only player_dismissed is available
    if 'is_wicket' not in df.columns and 'player_dismissed' in df.columns:
        df['is_wicket'] = df['player_dismissed'].notna().astype(int)
    elif 'is_wicket' in df.columns:
        df['is_wicket'] = df['is_wicket'].fillna(0).astype(int)
        
    # Ensure standard extra types if available
    if 'extra_type' not in df.columns:
        df['extra_type'] = pd.Series([np.nan] * len(df), dtype=object)
        if 'wides' in df.columns:
            df.loc[df['wides'] > 0, 'extra_type'] = 'wides'
        if 'noballs' in df.columns:
            df.loc[df['noballs'] > 0, 'extra_type'] = 'noballs'

    return df

def extract_features(df):
    print("Initializing Feature Extraction Pipeline...")
    
    # Pre-sort to maintain chronological sequence
    df = df.sort_values(by=['match_id', 'innings', 'overs', 'ballnumber']).reset_index(drop=True)
    df['ball_idx'] = np.arange(len(df)) # Unique sequence ID
    
    # ---------------------------------------------------------
    # Stage 1: Structural Context
    # ---------------------------------------------------------
    print("Stage 1: Structural Context...")
    df['Over'] = df['overs']
    df['is_wide'] = df['extra_type'].isin(['wides', 'wide'])
    df['is_noball'] = df['extra_type'].isin(['noballs', 'noball'])
    df['valid_ball'] = ~(df['is_wide'] | df['is_noball']).astype(bool)

    # Valid balls and balls remaining
    df['valid_balls_bowled'] = df.groupby(['match_id', 'innings'])['valid_ball'].cumsum()
    df['Balls Remaining'] = 120 - df['valid_balls_bowled']
    df['Balls Remaining'] = df['Balls Remaining'].clip(lower=0)
    
    # ---------------------------------------------------------
    # Stage 2: Match State Aggregations
    # ---------------------------------------------------------
    print("Stage 2: Match State Aggregations...")
    df['Present Score'] = df.groupby(['match_id', 'innings'])['total_run'].cumsum()
    
    df['cumulative_wickets_lost'] = df.groupby(['match_id', 'innings'])['is_wicket'].cumsum()
    df['Wickets Left'] = 10 - df['cumulative_wickets_lost']

    # Target Logic (0 for 1st Innings, Innings 1 Score + 1 for 2nd Innings)
    inn1_scores = df[df['innings'] == 1].groupby('match_id')['total_run'].sum().reset_index()
    inn1_scores.rename(columns={'total_run': 'inn1_total'}, inplace=True)
    df = df.merge(inn1_scores, on='match_id', how='left')
    df['Target'] = np.where(df['innings'] == 2, df['inn1_total'] + 1, 0)
    df['Target'] = df['Target'].fillna(0)

    # ---------------------------------------------------------
    # Stage 3: Dense Player & Partnership Context
    # ---------------------------------------------------------
    print("Stage 3: Dense Player Context...")
    # Striker current runs (Dynamically via cumsum)
    df['Striker Current Runs'] = df.groupby(['match_id', 'batter'])['batsman_run'].cumsum()

    # Non-Striker current runs (using vectorized merge_asof for speed)
    # 1. Create a log of all runs scored by every player mapped against the sequence ID
    m_b_r = df[['match_id', 'ball_idx', 'batter', 'batsman_run']].copy()
    m_b_r['cum_runs'] = m_b_r.groupby(['match_id', 'batter'])['batsman_run'].cumsum()
    m_b_r = m_b_r.sort_values('ball_idx')

    # 2. Merge backward to find what the non_striker's runs were just before/at this ball
    df_ns = df[['match_id', 'ball_idx', 'non_striker']].copy()
    df_ns = df_ns.rename(columns={'non_striker': 'batter'}).sort_values('ball_idx')
    
    ns_runs = pd.merge_asof(
        df_ns, m_b_r[['match_id', 'ball_idx', 'batter', 'cum_runs']],
        on='ball_idx', by=['match_id', 'batter'], direction='backward'
    )
    df['Non-Striker Current Runs'] = ns_runs['cum_runs'].fillna(0).sort_index()

    # Partnership Runs
    df['partnership_id'] = df.groupby(['match_id', 'innings'])['is_wicket'].shift().fillna(0).cumsum()
    df['Partnership Runs'] = df.groupby(['match_id', 'innings', 'partnership_id'])['total_run'].cumsum()

    # Bowler Economy
    df['bowler_cumulative_runs'] = df.groupby(['match_id', 'bowler'])['total_run'].cumsum()
    df['bowler_valid_balls'] = df.groupby(['match_id', 'bowler'])['valid_ball'].cumsum()
    df['bowler_legal_overs'] = df['bowler_valid_balls'] / 6
    df['Bowler Economy'] = np.where(df['bowler_legal_overs'] > 0,
                                    df['bowler_cumulative_runs'] / df['bowler_legal_overs'], 0)

    # ---------------------------------------------------------
    # Stage 4: Advanced Match Heuristics
    # ---------------------------------------------------------
    print("Stage 4: Advanced Match Heuristics...")
    # 18-ball Rolling Windows
    df['Runs_Last_18_Balls'] = df.groupby(['match_id', 'innings'])['total_run'].transform(
        lambda x: x.rolling(18, min_periods=1).sum())
    df['Wickets_Last_18_Balls'] = df.groupby(['match_id', 'innings'])['is_wicket'].transform(
        lambda x: x.rolling(18, min_periods=1).sum())

    # Rate Dynamics
    df['overs_completed'] = df['valid_balls_bowled'] / 6
    df['Current Run Rate (CRR)'] = np.where(df['overs_completed'] > 0, df['Present Score'] / df['overs_completed'], 0)
    
    df['overs_remaining'] = df['Balls Remaining'] / 6
    df['Required Run Rate (RRR)'] = np.where(
        (df['innings'] == 2) & (df['overs_remaining'] > 0),
        (df['Target'] - df['Present Score']) / df['overs_remaining'], 0
    )
    df['Required Run Rate (RRR)'] = df['Required Run Rate (RRR)'].clip(lower=0) # Prevent negative RRR if target passed

    df['RRR/CRR Ratio'] = np.where(df['Current Run Rate (CRR)'] > 0, 
                                   df['Required Run Rate (RRR)'] / df['Current Run Rate (CRR)'], 0)

    # Match Phases
    df['Powerplay'] = ((df['Over'] >= 0) & (df['Over'] <= 5)).astype(int)
    df['Middle'] = ((df['Over'] >= 6) & (df['Over'] <= 14)).astype(int)
    df['Death'] = ((df['Over'] >= 15) & (df['Over'] <= 19)).astype(int)

    # Resource Ratio
    df['Resource Ratio'] = np.where(df['Balls Remaining'] > 0, 
                                    df['Wickets Left'] / df['Balls Remaining'], 0)

    # ---------------------------------------------------------
    # Stage 5: Venue DNA & Difficulty
    # ---------------------------------------------------------
    print("Stage 5: Venue DNA & Difficulty...")
    venue_history = df[['match_id', 'venue', 'season']].drop_duplicates().sort_values(['season', 'match_id'])
    venue_history = pd.merge(venue_history, inn1_scores, on='match_id', how='inner')
    
    # Expanding mean EXCLUDING the current match
    venue_history['Venue Average 1st Innings Score'] = venue_history.groupby('venue')['inn1_total'].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    
    global_avg = venue_history['inn1_total'].mean()
    venue_history['Venue Average 1st Innings Score'] = venue_history['Venue Average 1st Innings Score'].fillna(global_avg)
    
    df = pd.merge(df, venue_history[['match_id', 'Venue Average 1st Innings Score']], on='match_id', how='left')
    df['Venue Average 1st Innings Score'] = df['Venue Average 1st Innings Score'].fillna(global_avg)
    
    df['Relative Target Difficulty'] = np.where(df['innings'] == 2, 
                                                df['Target'] - df['Venue Average 1st Innings Score'], 0)

    # ---------------------------------------------------------
    # Target Variable Preparation
    # ---------------------------------------------------------
    df['is_batting_team_winner'] = (df['batting_team'] == df['winning_team']).astype(int)
    
    return df

def split_and_save(df, base_dir):
    # Select desired core columns for output sequences
    core_features = [
        'match_id', 'season', 'innings', 'ball_idx',
        'Over', 'Balls Remaining', 
        'Present Score', 'Wickets Left', 'Target',
        'Striker Current Runs', 'Non-Striker Current Runs', 'Partnership Runs', 'Bowler Economy',
        'Runs_Last_18_Balls', 'Wickets_Last_18_Balls', 
        'Current Run Rate (CRR)', 'Required Run Rate (RRR)', 'RRR/CRR Ratio',
        'Powerplay', 'Middle', 'Death', 
        'Resource Ratio', 'Venue Average 1st Innings Score', 'Relative Target Difficulty',
        'is_batting_team_winner'
    ]
    
    # Filter out missing columns defensively
    missing_cols = [c for c in core_features if c not in df.columns]
    if missing_cols:
        print(f"Warning: Columns missing from dataset schema: {missing_cols}")
        core_features = [c for c in core_features if c in df.columns]

    final_df = df[core_features]

    print("Splitting Data (Train: 2008-2022 | Test: 2023)...")
    train_data = final_df[final_df['season'] <= 2022]
    test_data = final_df[final_df['season'] == 2023]

    print(f"Train matches sequence rows: {len(train_data)}")
    print(f"Test matches sequence rows: {len(test_data)}")

    train_path = os.path.join(base_dir, 'train_data.csv')
    test_path = os.path.join(base_dir, 'test_data.csv')

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)
    print(f"Pipeline Complete! Saved accurately to:\n- {train_path}\n- {test_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(BASE_DIR)
    RAW_CSV_PATH = os.path.join(PROJECT_ROOT, 'dataset', 'ipl_ball_by_ball.csv')
    
    if not os.path.exists(RAW_CSV_PATH):
        raise FileNotFoundError(f"Could not find primary dataset at {RAW_CSV_PATH}. Please ensure it is present in the dataset folder.")
    
    print(f"Loading raw dataset from {RAW_CSV_PATH}...")
    df_raw = pd.read_csv(RAW_CSV_PATH)
    
    # Normalize input schema
    df_clean = normalize_columns(df_raw)
    
    # Parse seasons intelligently (handles '2007/08' -> 2007)
    df_clean['season'] = df_clean['season'].astype(str).str.extract(r'(\d{4})').astype(int)
    
    # Filter out ties & no-results BEFORE extraction to prevent bad metrics
    valid_matches = df_clean['winning_team'].notna() & (~df_clean['winning_team'].isin(['Tie', 'No Result']))
    df_clean = df_clean[valid_matches]
    
    # Run the processing pipeline
    df_featurized = extract_features(df_clean)
    
    # Save train/test partitions
    split_and_save(df_featurized, os.path.join(PROJECT_ROOT, 'dataset'))