from src.preprocessing.Team import Team
import pandas as pd
from datetime import datetime
from pathlib import Path

BASE_PATH = '../../data'
TEAM_DIR = f'{BASE_PATH}/test_jsons'
GAMES_DIR = f'{BASE_PATH}/nfl_games'

def preprocess_data():
    curr_year = datetime.now().year

    def preprocess_team_stats(df):
        def get_rolling_df(df):
            numeric_cols = ['Home_Score', 'Away_Score', 'Offense_Total_Yrds', 'Offense_Pass_Yrds', 'Offense_Rush_Yrds',
                            'Turnovers_Lost', 'Total_Yrds_Allowed', 'Pass_Yrds_Allowed',
                            'Rush_Yrds_Allowed', 'Turnovers_Gained', 'Offense_Expected_Points',
                            'Defense_Expected_Points', 'Spteams_Expected_Points', 'Offensive_1sts']
        
            team_stats = df.copy()
        
            team_stats[numeric_cols] = team_stats[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
            team_stats = team_stats.sort_values(['Home_Team', 'Year'])
        
            rolling_stats = team_stats.copy()
        
            for col in numeric_cols:
                rolling_stats[col] = (
                    team_stats
                    .groupby(['Home_Team', 'Year'])[col]
                    .transform(lambda x: x.rolling(window=4, min_periods=1).mean().shift(1))
                )
        
            first_week_mask = team_stats.duplicated(subset=['Home_Team', 'Year'], keep='first') == False
            rolling_stats.loc[first_week_mask, numeric_cols] = pd.NA
        
            team_stats[numeric_cols] = rolling_stats[numeric_cols]
        
            return team_stats
            
        team_list = df['Home_Team'].unique().tolist()
        teams_dict = {}
        
        for team in team_list:
            try:
                try_year = curr_year

                while True:
                    json_path = Path(f'{TEAM_DIR}/{team}_2002_{try_year}.json')
                    
                    if json_path.exists():
                        team_df = pd.read_json(json_path)
                        break  # file found, break out of loop
                    else:
                        try_year -= 1
                        if try_year < 2002:
                            raise FileNotFoundError(f"JSON for {team} does not exist or can't be found")
        
                team_obj = Team(team_df, team)
                cleaned_df = team_obj.preprocess_team().get_df()
                teams_dict[team] = cleaned_df
                
            except Exception as e:
                print(f"Error processing {team} during JSON reading process: {e}")
                continue

        all_team_stats = pd.DataFrame()

        for team in team_list:
            df = teams_dict[team]
            df = get_rolling_df(df)
            all_team_stats = pd.concat([all_team_stats, df], axis=0, ignore_index=True)

        all_team_stats.to_json(f'{BASE_PATH}/check.json')
        return all_team_stats

    def preprocess_nfl_games(df):
        def change_old_teams(df, team):
            exclude = {"AFC", "NFC", "RIC", "CTR", "IRV"}
            df = df[~df[team].isin(exclude)].copy()
        
            replacements = {
                'STL': 'LAR', 'OAK': 'LV', 'SD': 'LAC', 'CLT': 'IND', 'OTI': 'TEN',
                'NOR': 'NO', 'WAS': 'WSH', 'RAM': 'LAR', 'TAM': 'TB', 'RAV': 'BAL',
                'GNB': 'GB', 'NWE': 'NE', 'SFO': 'SF', 'KAN': 'KC', 'CRD': 'ARI',
                'HTX': 'HOU', 'SDG': 'LAC', 'RAI': 'LV'
            }
            df[team] = df[team].replace(replacements)

            return df

        def adjust_record(record, team_type, winner):
            overall_summary = next((item['summary'] for item in record if item['name'] == 'overall'), None)
            home_summary = next((item['summary'] for item in record if item['name'] == 'Home'), None)
            road_summary = next((item['summary'] for item in record if item['name'] == 'Road'), None)
        
            def parse_summary(summary):
                if summary and '-' in summary:
                    try:
                        wins, losses = map(int, summary.split('-'))
                        return wins, losses
                    except Exception as e:
                        return None, None
                else:
                    return None, None
        
            overall_wins, overall_losses = parse_summary(overall_summary)
            home_wins, home_losses = parse_summary(home_summary)
            road_wins, road_losses = parse_summary(road_summary)
        
            if team_type == "home":
                if winner:  # Home team won
                    if home_wins is not None:
                        home_wins -= 1  # Don't count this win for home team
                    if overall_wins is not None:
                        overall_wins -= 1  # Don't count this win for overall record
                else:  # Away team won
                    if home_losses is not None:
                        home_losses -= 1  # Home team lost, so adjust their loss count
                    if overall_losses is not None:
                        overall_losses -= 1  # Don't count this loss for overall record
            else:  # away team
                if winner:  # Away team won
                    if road_wins is not None:
                        road_wins -= 1  # Don't count this win for away team
                    if overall_wins is not None:
                        overall_wins -= 1  # Don't count this win for overall record
                else:  # Home team won
                    if road_losses is not None:
                        road_losses -= 1  # Away team lost, so adjust their loss count
                    if overall_losses is not None:
                        overall_losses -= 1  # Don't count this loss for overall record
        
            return overall_wins, overall_losses, home_wins, home_losses, road_wins, road_losses

        included_cols = [
            "game_id", "season", "season_type", "week", "venue_id", "venue_indoor", "neutral_site",
            "home_id", "home_name", "home_abbreviation", "home_score", "home_winner",
            "home_records", "home_linescores",
            "away_id", "away_name", "away_abbreviation", "away_score", "away_winner",
            "away_records", "away_linescores",
            "broadcast_market", "broadcast_name", "status_type_description"
        ]
        
        nfl_df = df[included_cols]
        nfl_df = nfl_df.copy()
        nfl_df["broadcast_name"] = nfl_df["broadcast_name"].replace("", "local")
        nfl_df["broadcast_market"] = nfl_df["broadcast_market"].replace("", "local")
        nfl_df[['overall_wins(home)', 'overall_losses(home)', 'home_wins(home)', 'home_losses(home)', 'road_wins(home)',
                'road_losses(home)']] = nfl_df.apply(
            lambda row: pd.Series(adjust_record(row['home_records'], "home", row['home_winner'])), axis=1
        )
        nfl_df[['overall_wins(away)', 'overall_losses(away)', 'home_wins(away)', 'home_losses(away)', 'road_wins(away)',
                'road_losses(away)']] = nfl_df.apply(
            lambda row: pd.Series(adjust_record(row['away_records'], "away", row['away_winner'])), axis=1
        )
        nfl_df = nfl_df[nfl_df['season_type'] != 1]
        nfl_df.rename(
            columns={'home_abbreviation': 'Home_Team', 'away_abbreviation': 'Away_Team', 'season': 'Year', 'week': 'Week'},
            inplace=True)
        
        # *Some older team names are different* This is to change them
        home_away_list = ['Home', 'Away']

        for item in home_away_list:
            nfl_df = change_old_teams(nfl_df, f'{item}_Team')
        
        nfl_df = nfl_df[nfl_df['Year'] >= 2002]
        
        # Don't really care for Postponed status games
        nfl_df = nfl_df[nfl_df['status_type_description'] != 'Postponed']
        
        # Ensure the dataframe is sorted by year and week to process games chronologically
        nfl_df = nfl_df.sort_values(by=['Year', 'Week']).reset_index(drop=True)

        # Remove ties
        nfl_df = nfl_df[nfl_df['home_winner'] != nfl_df['away_winner']]

        nfl_df['status_type_description'] = nfl_df['status_type_description'].astype(str)

        return nfl_df

    def merge_games_and_teams(df, to_merge):
        numeric_cols = ['Home_Score', 'Away_Score', 'Offense_Total_Yrds', 'Offense_Pass_Yrds', 'Offense_Rush_Yrds',
                    'Turnovers_Lost', 'Total_Yrds_Allowed', 'Pass_Yrds_Allowed',
                    'Rush_Yrds_Allowed', 'Turnovers_Gained', 'Offense_Expected_Points',
                    'Defense_Expected_Points', 'Spteams_Expected_Points']

        df = df.merge(to_merge[['Home_Team', 'Year', 'Week', 'season_type'] + numeric_cols],
                      left_on=['Home_Team', 'Year', 'Week', 'season_type'],
                      right_on=['Home_Team', 'Year', 'Week', 'season_type'],
                      how='left', suffixes=('_home_stats', '')).copy()
    
        df = df.merge(to_merge[['Home_Team', 'Year', 'Week', 'season_type'] + numeric_cols],
                      left_on=['Away_Team', 'Year', 'Week', 'season_type'],
                      right_on=['Home_Team', 'Year', 'Week', 'season_type'],
                      how='left', suffixes=('', '_away_stats')).copy()
    
        return df

    main_df = pd.read_json(Path(f'{GAMES_DIR}/nfl_2002_{curr_year}.json'))
    nfl_df = preprocess_nfl_games(main_df)
    total_team_stats = preprocess_team_stats(nfl_df)
    main_df = merge_games_and_teams(nfl_df, total_team_stats)
    main_df.to_json(Path(f'../../data/preprocessed_data/preprocessed_2002_{curr_year}.json'))
    print("Data preprocessing completed!")
    return