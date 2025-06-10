import pandas as pd
import random
import os
import io
import time
import requests
from datetime import datetime
import sportsdataverse.nfl as sdv
from collections import deque

class RateLimiter:
    def __init__(self, max_requests=20, time_window=60):  # 20 per minute
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()
    
    def wait_if_needed(self):
        now = time.time()
        # Remove requests older than 1 minute
        while self.requests and now - self.requests[0] >= self.time_window:
            self.requests.popleft()
        
        # If at limit, wait until oldest request is 1 minute old
        if len(self.requests) >= self.max_requests:
            sleep_time = self.time_window - (now - self.requests[0]) + 1
            print(f"Rate limit reached. Waiting {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
        
        self.requests.append(now)

rate_limiter = RateLimiter(20, 60)

def polite_fetch_table(url):
    rate_limiter.wait_if_needed()
    tables = pd.read_html(url)
    return tables

def import_games(start_year, end_year=None):
    years = range(start_year, end_year + 1) if end_year else [start_year]
    all_games = pd.concat([sdv.espn_nfl_schedule(dates=year, return_as_pandas=True) for year in years], axis=0, ignore_index=True)
    rate_limiter.wait_if_needed()
    return all_games

def get_game_stats(team, start_year=int(datetime.now().year), end_year=None):
    team = team.lower()
    years = range(start_year, end_year + 1) if end_year else [start_year]
    team_df = pd.DataFrame()
    
    for year in years:
        url = f'https://www.pro-football-reference.com/teams/{team}/{year}.htm'
        try:
            df = polite_fetch_table(url)[1]
            team_df = pd.concat([team_df, df], axis=0, ignore_index=True)
        except Exception as e:
            print(f"Error retrieving data for {team} in {year}: {e}")
    
    return team_df

def get_data(dir_name):
    files = os.listdir(f'../../data/{dir_name}')
    curr_year = datetime.now().year
    
    if dir_name == 'team_jsons':
        os.makedirs('../../data/test_jsons', exist_ok=True)
        
        for idx, file in enumerate(files):
            file_edited = file.replace('.json', '')
            last_updated_year, team = int(file_edited.split('_')[-1]), file_edited.split('_')[0]
            
            print(f"Processing team {team} from year {last_updated_year + 1} to {curr_year}")
            
            new_team_data = get_game_stats(team, last_updated_year + 1, curr_year)
            
            if new_team_data is not None and not new_team_data.empty:
                data_df = pd.read_json(f'../../data/{dir_name}/{file}')
                
                data_df = pd.concat([data_df, new_team_data], axis=0, ignore_index=True)
                
                data_df.to_json(f'../../data/test_jsons/{team}_2002_{curr_year}.json', orient='records')
                print(f"Updated {team} data through {curr_year}")
            else:
                print(f"No new data available for {team}")
    else:
        os.makedirs('../../data/nfl_games', exist_ok=True)
        
        if not files:
            print("No files found in directory")
            return
            
        files_edited = files[0].replace('.json', '')
        last_updated_year = int(files_edited.split('_')[-1])
        
        print(f"Processing NFL games from year {last_updated_year} to {curr_year}")
        
        new_data = import_games(last_updated_year, curr_year)
        
        if new_data is not None and not new_data.empty:
            data_df = pd.read_json(f'../../data/{dir_name}/nfl_2002_{last_updated_year}.json')
            data_df = pd.concat([data_df, new_data], axis=0, ignore_index=True)
            data_df.to_json(f'../../data/nfl_games/nfl_2002_{curr_year}.json', orient='records')
            print(f"Updated NFL games through {curr_year}")
        else:
            print("No new NFL game data available")

def get_upcoming_data():
    print("Starting data update process...")
    print("Rate limit: 20 requests per minute")
    
    get_data('nfl_games')
    get_data('team_jsons')
    
    print("Data update process completed!")
    return