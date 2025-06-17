import numpy as np
import pandas as pd
from io import StringIO
from pathlib import Path
from datetime import datetime

# useful paths
BASE_PATH = '../../data'
PROCESSED_DIR = f'{BASE_PATH}/preprocessed_data/'
curr_year = datetime.now().year

def build_new_features() -> None:

    def update_elo(home_elo, away_elo, home_score, away_score, k_factor=30) -> tuple[float, float]:
        """
        Updates Elo ratings for both the home and away teams after a game.
        home_elo: current Elo rating of the home team.
        away_elo: current Elo rating of the away team.
        home_score: score of the home team.
        away_score: score of the away team.
        k_factor: the K-factor, which controls the sensitivity of rating changes.
        """

        # Calculate the expected scores for each team
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 / (1 + 10 ** ((home_elo - away_elo) / 400))

        # Calculate the actual scores based on the result of the game
        if home_score > away_score:
            actual_home = 1  # Home team wins
            actual_away = 0  # Away team loses
        elif away_score > home_score:
            actual_home = 0  # Home team loses
            actual_away = 1  # Away team wins
        else:
            actual_home = 0.5  # Draw
            actual_away = 0.5  # Draw

        # Update Elo ratings for both teams
        new_home_elo = home_elo + k_factor * (actual_home - expected_home)
        new_away_elo = away_elo + k_factor * (actual_away - expected_away)

        return new_home_elo, new_away_elo

    fp = Path(f'{PROCESSED_DIR}/preprocessed_2002_{curr_year}.json')
    nfl_df = pd.read_json(StringIO(fp.read_text()))

    # Incorporate Elo Ratings
    initial_elo = 1500
    team_elos = {}  # Dictionary to store Elo ratings for each team

    # Add columns for Elo ratings
    nfl_df['elo_home'] = initial_elo
    nfl_df['elo_away'] = initial_elo
    nfl_df = nfl_df.astype({'elo_home': 'float64', 'elo_away': 'float64'})

    # Process each game to assign and update Elo ratings
    for idx, row in nfl_df.iterrows():
        home_team = row['Home_Team']
        away_team = row['Away_Team']

        # Get current Elo ratings (or initialize if first game)
        home_elo = team_elos.get(home_team, initial_elo)
        away_elo = team_elos.get(away_team, initial_elo)

        # Assign Elo ratings to the current game (before the game is played)
        nfl_df.at[idx, 'elo_home'] = home_elo
        nfl_df.at[idx, 'elo_away'] = away_elo

        # Update Elo ratings based on game outcome
        new_home_elo, new_away_elo = update_elo(
            home_elo, away_elo, row['home_score'], row['away_score']
        )

        # Store updated Elo ratings for the teams' next games
        team_elos[home_team] = new_home_elo
        team_elos[away_team] = new_away_elo

    def average_terms(df, term1, term2) -> pd.DataFrame:
        return (df[term1] + df[term2]) / 2

    # Home team performance features
    home_features = [
        "Home_Score", "Offense_Total_Yrds", "Offense_Pass_Yrds", "Offense_Rush_Yrds",
        "Turnovers_Lost", "Total_Yrds_Allowed", "Pass_Yrds_Allowed", "Rush_Yrds_Allowed",
        "Turnovers_Gained", "Offense_Expected_Points", "Defense_Expected_Points", "Spteams_Expected_Points",
        "turnover_differential"
    ]

    # Away team rolling average features
    away_features = [
        "Home_Score_away", "Offense_Total_Yrds_away", "Offense_Pass_Yrds_away", "Offense_Rush_Yrds_away",
        "Turnovers_Lost_away", "Total_Yrds_Allowed_away", "Pass_Yrds_Allowed_away", "Rush_Yrds_Allowed_away",
        "Turnovers_Gained_away", "Offense_Expected_Points_away", "Defense_Expected_Points_away",
        "Spteams_Expected_Points_away"
    ]

    # Create interaction terms (multiply home and away features)
    offense_terms = ["Pass", "Rush", "Total"]
    home_away_terms = ['', '_away_stats']

    for term in offense_terms:
        for status in home_away_terms:
            other_status = '_away_stats' if status == '' else ''

            nfl_df[f'Expected_{term}_Yrds{status}'] = average_terms(nfl_df, f'Offense_{term}_Yrds{status}',
                                                                    f'{term}_Yrds_Allowed{other_status}')

    # Create Elo diff
    nfl_df['elo_diff'] = nfl_df['elo_home'] - nfl_df['elo_away']
    nfl_df.to_json(f'{PROCESSED_DIR}/engineered_2002_{curr_year}.json', orient='records')

    print("Feature Engineering completed!")