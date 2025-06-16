import pandas as pd
import os
from datetime import datetime
import sportsdataverse.nfl as sdv
from pathlib import Path
from dotenv import load_dotenv
from src.scraper.RLManager import RLManager

BASE_PATH = '../../data/'
TEAM_DIR = f'{BASE_PATH}/just_in_case_data/team_jsons'
TEST_DIR = f'{BASE_PATH}test_jsons'
GAMES_DIR = f'{BASE_PATH}/nfl_games'

def import_games(start_year, end_year=None) -> pd.DataFrame:
    years = range(start_year, end_year + 1) if end_year else [start_year]
    all_games = pd.concat([sdv.espn_nfl_schedule(dates=year, return_as_pandas=True) for year in years], axis=0, ignore_index=True)
    return all_games

def get_game_stats(team, start_year=int(datetime.now().year), end_year=None) -> pd.DataFrame:
    team = team.lower()
    years = range(start_year, end_year + 1) if end_year else [start_year]
    team_df = pd.DataFrame()
    scraper = None

    load_dotenv(f'{BASE_PATH}/keys/keys.env')
    api_key = os.getenv('WEBSHARE_API_KEY')
    
    for year in years:
        url = f'https://www.pro-football-reference.com/teams/{team}/{year}.htm'
        scraper = RLManager(url, max_retries=6, base_delay=1.0,
                            webshare_token=api_key)

        try:
            df = scraper.read_html()[1]
            df['Year'] = year
            team_df = pd.concat([team_df, df], axis=0, ignore_index=True)
        except Exception as e:
            print(f"Error retrieving data for {team} in {year}: {e}")

    scraper.close()

    return team_df


def get_data(dir_name) -> None:
    curr_year = datetime.now().year

    if dir_name == 'team_jsons':
        os.makedirs(Path(TEST_DIR), exist_ok=True)

        teams = [
            "SFO", "CHI", "CIN", "BUF", "DEN", "CLE", "TAM", "CRD", "SDG", "KAN",
            "CLT", "WAS", "DAL", "MIA", "PHI", "ATL", "NYG", "JAX", "NYJ", "DET",
            "GNB", "CAR", "NWE", "RAI", "RAM", "RAV", "NOR", "SEA", "PIT", "HTX",
            "OTI", "MIN"
        ]

        team_conversions = {
            'CLT': 'IND', 'OTI': 'TEN', 'NOR': 'NO', 'WAS': 'WSH', 'RAM': 'LAR', 'TAM': 'TB', 'RAV': 'BAL',
            'GNB': 'GB', 'NWE': 'NE', 'SFO': 'SF', 'KAN': 'KC', 'CRD': 'ARI', 'HTX': 'HOU', 'SDG': 'LAC',
            'RAI': 'LV'
        }

        for team in teams:
            converted_team = team_conversions.get(team, team)
            curr_team_file = f'{converted_team}_2002_{curr_year}.json'
            curr_file = Path(f'{TEST_DIR}/{curr_team_file}')

            if curr_file.exists():
                file_edited = curr_team_file.replace('.json', '')
                last_updated_year = int(file_edited.split('_')[-1])
                new_team_data = get_game_stats(team, last_updated_year, curr_year)

                if new_team_data is not None and not new_team_data.empty:
                    data_df = pd.read_json(
                        Path(f'{BASE_PATH}/{dir_name}/{converted_team}_2002_{last_updated_year}.json'))
                    data_df = pd.concat([data_df, new_team_data], axis=0, ignore_index=True)
                    data_df.to_json(Path(f'{TEST_DIR}/{converted_team}_2002_{curr_year}.json'), orient='records')
                    print(f"Updated {team} ({converted_team}) data through {curr_year}")
                else:
                    print(f"No new data available for {team} ({converted_team})")
            else:
                last_updated_year = 2001
                new_team_data = get_game_stats(team, last_updated_year + 1, curr_year)

                if new_team_data is not None and not new_team_data.empty:
                    new_team_data.to_json(Path(f'{TEST_DIR}/{converted_team}_2002_{curr_year}.json'), orient='records')
                    print(f"Updated {team} ({converted_team}) data through {curr_year}")
                else:
                    print(f"No new data available for {team} ({converted_team})")

    else:
        files = os.listdir(Path(GAMES_DIR))
        os.makedirs(Path(f'{BASE_PATH}/nfl_games'), exist_ok=True)

        if not files:
            print("No files found in directory")
            return

        files_edited = files[0].replace('.json', '')
        last_updated_year = int(files_edited.split('_')[-1])

        print(f"Processing NFL games from year {last_updated_year} to {curr_year}")

        new_data = import_games(last_updated_year, curr_year)

        if new_data is not None and not new_data.empty:
            data_df = pd.read_json(Path(f'../../data/{dir_name}/nfl_2002_{last_updated_year}.json'))
            data_df = pd.concat([data_df, new_data], axis=0, ignore_index=True)
            data_df.to_json(Path(f'../../data/nfl_games/nfl_2002_{curr_year}.json'), orient='records')
            print(f"Updated NFL games through {curr_year}")
        else:
            print("No new NFL game data available")

def get_upcoming_data() -> None:
    print("Starting data update process...")
    print("Rate limit: 20 requests per minute")
    
    get_data('nfl_games')
    get_data('team_jsons')
    
    print("Data update process completed!")
    return