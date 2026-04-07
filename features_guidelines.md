# IPL Win Probability Feature Guidelines

This document outlines the feature engineering pipeline used to transform the raw `ipl_ball_by_ball.csv` dataset into predictive features for the IPL Win Probability models (Logistic Regression, Random Forest, XGBoost, and BiLSTM).

## 1. Raw Data Source
The primary dataset is `dataset/ipl_ball_by_ball.csv`, which contains ball-by-ball records for every IPL match.

## 2. Feature Extraction Pipeline

The extraction logic is implemented in `src/ipl_predictor/features.py` and follows a multi-stage process:

### Stage 1: Structural Context
Basic match structure features are derived from the raw ball counts.
- **Over**: The current over number (extracted from the `ball` decimal).
- **Balls Remaining**: Calculated as `120 - valid_balls_bowled`. This explicitly excludes wides and no-balls from the countdown.

### Stage 2: Match State Aggregations
Cumulative metrics that track the progress of the innings.
- **Present Score**: Cumulative sum of runs (off bat + extras) scored in the current innings.
- **Wickets Left**: Remaining wickets for the batting team (`10 - cumulative_wickets_lost`).
- **Target**: The score to beat + 1. This is only applicable for the 2nd innings; it is set to 0 for the 1st innings.

### Stage 3: Dense Player & Partnership Context
Dynamic tracking of players currently on the field.
- **Striker Current Runs**: Total runs scored by the batsman currently facing the ball.
- **Non-Striker Current Runs**: Total runs scored by the batsman at the other end.
- **Partnership Runs**: Cumulative runs scored since the last wicket fell.
- **Bowler Economy**: The current economy rate of the bowler in the match (`runs_conceded / legal_overs_bowled`).

### Stage 4: Advanced Match Heuristics
Derived features that capture momentum and pressure.
- **Momentum Window (Last 18 Balls)**:
  - `Runs_Last_18_Balls`: Sum of runs scored in the previous 3 overs.
  - `Wickets_Last_18_Balls`: Number of wickets lost in the previous 3 overs.
- **Run Rate Dynamics**:
  - `Current Run Rate (CRR)`: `Present Score / (overs_completed)`.
  - `Required Run Rate (RRR)`: `(Target - Present Score) / (overs_remaining)`. (2nd innings only).
  - `RRR/CRR Ratio`: A pressure index comparing current pace vs. required pace.
- **Match Phase (One-Hot Encoded)**:
  - `Powerplay`: Overs 0-5.
  - `Middle`: Overs 6-14.
  - `Death`: Overs 15-20.
- **Resource Ratio**: `Wickets Left / Balls Remaining` – scales the value of a wicket based on how much time is left.

### Stage 5: Venue DNA & Difficulty
Historical context to normalize scores across different stadiums.
- **Venue Average 1st Innings Score**: The expanding mean of 1st innings totals at the specific venue, **excluding** the current match to prevent data leakage.
- **Relative Target Difficulty**: `Target - Venue_Avg_1st_Inn_Score`. This measures if the target is above or below par for that specific ground.

## 3. Predicted Label
- **is_batting_team_winner**: A binary target variable.
  - `1`: The team currently batting won the match.
  - `0`: The team currently batting lost the match.
  - *Tied matches are excluded from the training and test sets.*

## 4. Sequence Handling (BiLSTM)
For the BiLSTM model (implemented in `src/ipl_predictor/bilstm.py`), the features are grouped by `match_id` and `innings` to maintain the temporal flow.
- Models ingest matches as sequences of balls.
- Padding is applied to ensure uniform sequence lengths for batch hardware acceleration (MPS/GPU).
- The model currently utilizes a subset of the **top 20 core features** for inference.

## 5. Data Splitting Strategy
- **Training Set**: All matches from the **2008 to 2022** seasons.
- **Test Set**: All matches from the **2023** season (Holdout set for evaluation).