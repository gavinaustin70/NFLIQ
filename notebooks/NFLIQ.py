#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sportsdataverse.nfl as sdv
import numpy as np
import pandas as pd
import time
import requests
import random
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif, SelectFromModel
from hyperopt import hp, fmin, tpe, Trials


# In[4]:


pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_colwidth')


# In[5]:


def import_games(start_year, end_year=None):
    years = range(start_year, end_year + 1) if end_year else [start_year]
    all_games = pd.concat([sdv.espn_nfl_schedule(dates=year, return_as_pandas=True) for year in years], axis=0, ignore_index=True)
    return all_games


# In[6]:


"""
main_df = import_games(2002, 2023)
main_df.to_json(f'nfl_2002_2023.json', orient='records')
"""


# In[7]:


main_df = pd.read_json(f'nfl_2002_2023.json')
main_df


# In[8]:


pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_columns', None)


# In[9]:


main_df.columns


# In[10]:


def get_game_stats(team, start_year=2002, end_year=2024):
    team = team.lower()
    years = range(start_year, end_year + 1) if end_year else [start_year]
    team_df = pd.DataFrame()

    for year in years:
        url = f'https://www.pro-football-reference.com/teams/{team}/{year}.htm'

        try:
            df = pd.read_html(url)[1]
        except:
            return None

        team_df = pd.concat([team_df, df], axis=0, ignore_index=True)
        time.sleep(random.uniform(3.1, 3.3))

    return team_df


# In[11]:


team_list = main_df['home_abbreviation'].unique().tolist()

"""
teams_dict = {}
for team in team_list:
    teams_dict[team] = get_game_stats(team, 2002, 2024)
    teams_dict[team].to_json(f'./team_jsons2/{teems[team]}_2002_2024.json')

team_list
"""


# In[12]:


"""
for team in teams_dict.keys():
    try:
        teams_dict[team].to_json(f'./team_jsons2/{team}_2002_2024.json')
    except:
        continue
"""


# In[13]:


"""
teems = {'CLT':'IND', 'OTI':'TEN', 'NOR': 'NO', 'WAS':'WSH', 'RAM': 'LAR', 'TAM':'TB', 'RAV':'BAL', 'GNB':'GB', 'NWE':'NE', 'SFO':'SF', 'KAN':'KC', 'CRD':'ARI', 'HTX':'HOU', 'SDG':'LAC', 'RAI':'LV'}
for team in teems:
    get_game_stats(team).to_json(f'./team_jsons2/{teems[team]}_2002_2024.json')
"""


# In[14]:


def get_team_jsons(team_dict):
    for team, df in teams_dict.items():
        if isinstance(df, pd.DataFrame):
            df.to_json(f'./team_jsons/{team}_2002_2024.json')
        else:
            print(f"Skipping {team}: Data is None or not a DataFrame")


# In[15]:


def assign_year(df, team, start_year=2001):
    df = df.rename(columns={'(\'Unnamed: 0_level_0\', \'Week\')': 'Week', '(\'Unnamed: 1_level_0\', \'Day\')': 'Day', "('Unnamed: 4_level_0', 'Unnamed: 4_level_1')": "boxscore",
                            "('Unnamed: 9_level_0', 'Opp')": "Opponents", "('Score', 'Tm')": "Home_Score",	"('Score', 'Opp')": "Away_Score",
                            "('Offense', '1stD')": "Offensive_1sts", "('Offense', 'TotYd')": "Offense_Total_Yrds", "('Offense', 'PassY')": "Offense_Pass_Yrds",
                            "('Offense', 'RushY')": "Offense_Rush_Yrds", "('Offense', 'TO')": "Turnovers_Lost", "('Defense', '1stD')": "Defense_1st_Allowed",
                            "('Defense', 'TotYd')": "Total_Yrds_Allowed", "('Defense', 'PassY')": "Pass_Yrds_Allowed", "('Defense', 'RushY')": "Rush_Yrds_Allowed",
                            "('Defense', 'TO')": "Turnovers_Gained", "('Expected Points', 'Offense')": "Offense_Expected_Points",
                            "('Expected Points', 'Defense')": "Defense_Expected_Points", "('Expected Points', 'Sp. Tms')": "Spteams_Expected_Points"})
    df.drop(columns=df.columns[[2, 3, 5, 6, 7, 8]], inplace=True)
    df['Week'] = df['Week'].replace({'Wild Card': -1, 'Division': -2, 'Conf. Champ.': -3, 'SuperBowl': -4})
    df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
    df['season_type'] = np.where(df['Week'] < 0, 3, 2)
    
    current_year = int(start_year)
    prev_week = 0
    years = []

    for week in df['Week']:
        if (week == 1) or ((prev_week == 17) and (week == 2) and (team == 'TB') and (current_year == 2016)):
            current_year += 1
        years.append(current_year)
        prev_week = week

    df["Year"] = years
    df['Team'] = team
    df.loc[(df['Year'] > 2008) & (df['Week'] == -4), 'Week'] = 5
    df['Week'] = df['Week'].abs()
    
    return df


# In[16]:


"""
def assign_year(df, team, start_year=2001):
    df = df.rename(columns={'(\'Unnamed: 0_level_0\', \'Week\')': 'Week', '(\'Unnamed: 1_level_0\', \'Day\')': 'Day', 
                           '(\'Unnamed: 2_level_0\', \'Date\')': 'Date', '(\'Unnamed: 3_level_0\', \'Unnamed: 3_level_1\')': 'Time',
                           '(\'Unnamed: 5_level_0\', \'Unnamed: 5_level_1\')': 'Result', '(\'Unnamed: 6_level_0\', \'OT\')': 'OT',
                           '(\'Unnamed: 7_level_0\', \'Rec\')': 'Record', '(\'Unnamed: 8_level_0\', \'Unnamed: 8_level_1\')': '@'})
    df['Team'] = team
    df['Week'] = pd.to_numeric(df['Week'], errors='coerce')
    df['Week'] = df['Week'].fillna(0).astype(int)
    current_year = start_year
    years = []

    for week in df['Week']:
        if week == 1:
            current_year += 1
        years.append(current_year)

    df["Year"] = years
    return df
"""


# In[17]:


teams_dict = {}

for team in team_list:
    try:
        df = pd.read_json(f'./team_jsons2/{team}_2002_2024.json')
        df = assign_year(df, team)
        teams_dict[team] = df
    except:
        continue


# In[18]:


teams_dict['NYJ'].head(25)


# In[19]:


def adjust_record(record, team_type, winner):
    overall_summary = next((item['summary'] for item in record if item['name'] == 'overall'), None)
    home_summary = next((item['summary'] for item in record if item['name'] == 'Home'), None)
    road_summary = next((item['summary'] for item in record if item['name'] == 'Road'), None)

    def parse_summary(summary):
        if summary and '-' in summary:
            try:
                wins, losses = map(int, summary.split('-'))
                return wins, losses
            except ValueError:
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


# In[20]:


def change_old_teams(df, team):
    df = df[df[team] != "AFC"]
    df = df[df[team] != "NFC"]
    df = df[df[team] != "RIC"]
    df = df[df[team] != "CTR"]
    df = df[df[team] != "IRV"]

    replace_dict = {'STL': 'LAR', 'OAK': 'LV', 'SD': 'LAC'}
    df[team] = df[team].replace(replace_dict)
    
    return df


# In[21]:


def arrange_team_json(df, team):
    df = df.copy()
    df = df.dropna(subset=['Opponents'])
    df.rename(columns={"Team": "Home_Team"}, inplace=True)
    
    return df


# In[22]:


tb_df = arrange_team_json(teams_dict['TB'], 'TB')
sea_df = arrange_team_json(teams_dict['SEA'], 'SEA')
tb_df[tb_df['Year'] == 2020]


# In[23]:


"""
def get_rolling_df(df):
    numeric_cols = ['Home_Score', 'Away_Score','Offense_Total_Yrds', 'Offense_Pass_Yrds', 'Offense_Rush_Yrds', 
                'Turnovers_Lost', 'Total_Yrds_Allowed', 'Pass_Yrds_Allowed', 
                'Rush_Yrds_Allowed', 'Turnovers_Gained', 'Offense_Expected_Points', 
                'Defense_Expected_Points', 'Spteams_Expected_Points']

    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df['Year'] = df[]

    team_stats = df[['Home_Team', 'Year', 'Week', 'season_type'] + numeric_cols].copy()
    team_stats[numeric_cols] = (team_stats
                                     .groupby(['Home_Team', 'Year'])[numeric_cols]
                                     .expanding()
                                     .mean()
                                     .shift(1)
                                     .reset_index(level=[0,1], drop=True))

    return team_stats
"""


# In[24]:


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


# In[25]:


def update_elo(home_elo, away_elo, home_score, away_score, k_factor=30):
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


# In[26]:


included_cols = [
    "game_id", "season", "season_type", "week", "venue_id", "venue_indoor", "neutral_site",
    "home_id", "home_name", "home_abbreviation", "home_score", "home_winner",
    "home_records", "home_linescores",
    "away_id", "away_name", "away_abbreviation", "away_score", "away_winner",
    "away_records", "away_linescores",
    "broadcast_market", "broadcast_name", "status_type_description"
]

nfl_df = main_df[included_cols]
nfl_df = nfl_df.copy()
nfl_df["broadcast_name"] = nfl_df["broadcast_name"].replace("", "local")
nfl_df["broadcast_market"] = nfl_df["broadcast_market"].replace("", "local")
nfl_df[['overall_wins(home)', 'overall_losses(home)','home_wins(home)', 'home_losses(home)', 'road_wins(home)', 'road_losses(home)']] = nfl_df.apply(
    lambda row: pd.Series(adjust_record(row['home_records'], "home", row['home_winner'])), axis=1
)
nfl_df[['overall_wins(away)', 'overall_losses(away)','home_wins(away)', 'home_losses(away)', 'road_wins(away)', 'road_losses(away)']] = nfl_df.apply(
    lambda row: pd.Series(adjust_record(row['away_records'], "away", row['away_winner'])), axis=1
)
nfl_df = nfl_df[nfl_df['season_type'] != 1]
nfl_df.rename(columns={'home_abbreviation': 'Home_Team', 'away_abbreviation': 'Away_Team', 'season': 'Year', 'week': 'Week'}, inplace=True)

# *Some older team names are different* This is to change them
nfl_df = change_old_teams(nfl_df, 'Home_Team')
nfl_df = change_old_teams(nfl_df, 'Away_Team')


nfl_df = nfl_df[(nfl_df['Year'] >= 2002) & (nfl_df['Year'] < 2024)]

# Don't really care for Postponed status games
nfl_df = nfl_df[nfl_df['status_type_description'] != 'Postponed']

# Incorporate Elo Ratings
initial_elo = 1500
team_elos = {}  # Dictionary to store Elo ratings for each team

# Ensure the dataframe is sorted by year and week to process games chronologically
nfl_df = nfl_df.sort_values(by=['Year', 'Week']).reset_index(drop=True)

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

# Remove ties
nfl_df = nfl_df[nfl_df['home_winner'] != nfl_df['away_winner']]


# In[27]:


nfl_df


# In[28]:


def merge_dfs(df, to_merge):
    numeric_cols = ['Home_Score', 'Away_Score','Offense_Total_Yrds', 'Offense_Pass_Yrds', 'Offense_Rush_Yrds', 
                'Turnovers_Lost', 'Total_Yrds_Allowed', 'Pass_Yrds_Allowed', 
                'Rush_Yrds_Allowed', 'Turnovers_Gained', 'Offense_Expected_Points', 
                'Defense_Expected_Points', 'Spteams_Expected_Points']
    
    df = df.merge(to_merge[['Home_Team', 'Year', 'Week', 'season_type'] + numeric_cols], 
                                      left_on=['Home_Team', 'Year', 'Week', 'season_type'], 
                                      right_on=['Home_Team', 'Year', 'Week', 'season_type'], 
                                      how='left', suffixes=('_home', ''))
    
    df = df.merge(to_merge[['Home_Team', 'Year', 'Week', 'season_type'] + numeric_cols], 
                                      left_on=['Away_Team', 'Year', 'Week', 'season_type'], 
                                      right_on=['Home_Team', 'Year', 'Week', 'season_type'], 
                                      how='left', suffixes=('', '_away'))

    return df


# In[29]:


team_list = nfl_df['Home_Team'].unique().tolist()
all_team_stats = pd.DataFrame()

for team in team_list:
    df = arrange_team_json(teams_dict[team], team)
    df = get_rolling_df(df)
    all_team_stats = pd.concat([all_team_stats, df], axis=0, ignore_index=True)


# In[30]:


all_team_stats[(all_team_stats['Home_Team'] == 'TB') & (all_team_stats['Year'].isin(range(2002,2004)))].head(40)


# In[31]:


nfl_df = merge_dfs(nfl_df, all_team_stats)
nfl_df['turnover_differential'] = (nfl_df['Turnovers_Gained'] - nfl_df['Turnovers_Lost']) / (nfl_df['Turnovers_Gained_away'] - nfl_df['Turnovers_Lost_away']).replace(0, np.nan)
nfl_df


# In[32]:


nfl_df[(nfl_df['Home_Score'].isna()) & (nfl_df['season_type'] == 2)]


# In[33]:


pd.reset_option('display.max_rows')


# In[34]:


def average_terms(df, term1, term2):
    return df[term1] + df[term2] / 2


# In[35]:


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
    "Turnovers_Gained_away", "Offense_Expected_Points_away", "Defense_Expected_Points_away", "Spteams_Expected_Points_away"
]

# Create interaction terms (multiply home and away features)
offense_terms = ["Pass", "Rush", "Total"]
home_away_terms = ['', '_away']

for term in offense_terms:
    for status in home_away_terms:
        other_status = ''

        if status == '':
            other_status = '_away'
        
        nfl_df[f'Expected_{term}_Yrds{status}'] = average_terms(nfl_df, f'Offense_{term}_Yrds{status}', f'{term}_Yrds_Allowed{other_status}')

# Create Elo diff
nfl_df['elo_diff'] = nfl_df['elo_home'] - nfl_df['elo_away']


# In[36]:


nfl_df


# In[37]:


"""
selected_features = home_features
selected_features.append("home_winner")
sns.pairplot(nfl_df[selected_features], hue ='home_winner')
plt.show()
"""


# In[38]:


sns.boxplot(x='home_winner', y='elo_diff', data=nfl_df)


# In[39]:


target_col = "home_winner"
all_cols = nfl_df.columns[24:].drop('Home_Team_away').tolist()
print(all_cols)

categorical_cols = []

numerical_cols = all_cols


# Boxplots of all num columns
"""
for col in numerical_cols:
    sns.boxplot(x=X_train[col])
    plt.show()
"""

nfl_df = nfl_df.dropna(subset=[target_col])

num_wins = (nfl_df['home_winner'] == 1).sum()
num_losses = (nfl_df['home_winner'].shape[0] - num_wins)
class_weights = num_losses / num_wins
X = nfl_df[numerical_cols + categorical_cols]
y = nfl_df[target_col].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", KNNImputer(n_neighbors=3)),
            ("scaler", StandardScaler()),
        ]), numerical_cols),
    ]
)

#Forest (Best Model So Far)
"""
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=500,        # Increase the number of trees for more diversity
        random_state=42, 
        max_depth=7,             # Limit tree depth to avoid overfitting
        min_samples_split=15,    # Increase min_samples_split to reduce tree complexity
        min_samples_leaf=4,      # Increase min_samples_leaf for more generalization
    ))
])
"""

#GradientBooster (2nd)
"""
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=150,        # Use a smaller number of trees for better generalization
        random_state=42, 
        max_depth=5,             # Depth to avoid overfitting
        min_samples_split=30,    # Higher value to prevent overly deep trees
        min_samples_leaf=6,      # Higher value to reduce model complexity
        learning_rate=0.01      # Smaller learning rate to avoid overfitting
    ))
])
"""

#XGBoost
xgboost_classifier = xgb.XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.01706675223214993,
        subsample=0.7640307727051777,
        colsample_bytree=0.8994869541803375,
        gamma=3.3490721414012086,
        min_child_weight=8,
        random_state=42,
        eval_metric="logloss"
    )

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", xgboost_classifier)
])

#GridSearch For Forest
"""
param_grid = {
    'classifier__n_estimators': [100, 300, 500],  # Number of trees in the forest (more trees tend to improve accuracy)
    'classifier__max_depth': [5, 10, 20],  # Depth of the trees; None allows unlimited depth
    'classifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'classifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'classifier__max_features': ['sqrt', 'log2'],  # Number of features to consider when looking for the best split
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
"""

#RandomizedSearch For Forest
"""
param_dist = {
    'classifier__n_estimators': [100, 300, 500],  # Number of trees in the forest
    'classifier__max_depth': [10, 20, 30, None],  # Depth of trees (None allows unlimited depth)
    'classifier__min_samples_split': [2, 5, 10],  # Number of samples required to split a node
    'classifier__min_samples_leaf': [1, 2, 4],  # Number of samples required at a leaf node
    'classifier__max_features': ['sqrt', 'log2'],  # Number of features to consider per split
    'classifier__bootstrap': [True, False]  # Whether to use bootstrap sampling
}

random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, cv=5, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print("Best parameters found: ", random_search.best_params_)

# Evaluate the new model
train_accuracy = random_search.best_estimator_.score(X_train, y_train)
test_accuracy = random_search.best_estimator_.score(X_test, y_test)
cross_val_scores = cross_val_score(random_search.best_estimator_, X, y, cv=5, scoring="accuracy")

print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"Cross-validated accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
"""

#GridSearch For GradientBooster
"""
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [3, 5, 7],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
"""

#GridSearch For XGBoost
"""
param_grid = {
    'classifier__n_estimators': [50, 100, 200],  # Number of trees
    'classifier__max_depth': [3, 5, 7],           # Depth of each tree
    'classifier__learning_rate': [0.01, 0.05, 0.1],  # Learning rate
    'classifier__subsample': [0.8, 1.0],          # Fraction of samples used per tree
    'classifier__colsample_bytree': [0.8, 1.0],   # Fraction of features used per tree
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)

best_model = grid_search.best_estimator_

train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
"""

#hyperopt For Forest
"""
space = {
    'classifier__n_estimators': hp.choice('n_estimators', [50, 100, 200]),
    'classifier__max_depth': hp.choice('max_depth', [10, 20, None]),
    'classifier__min_samples_split': hp.choice('min_samples_split', [2, 5, 10]),
    'classifier__min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 4]),
    'classifier__max_features': hp.choice('max_features', [None, 'sqrt', 'log2'])
}

def objective(params):
    pipeline.set_params(**params)
    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
    return -score

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

print("Best parameters:", best)
"""

#hyperopt for XGBoost
"""
space = {
    'classifier__n_estimators': hp.choice('n_estimators', [150, 300, 500, 700]),
    'classifier__max_depth': hp.choice('max_depth', [4, 5, 6]),
    'classifier__learning_rate': hp.uniform('learning_rate', 0.005, 0.02),
    'classifier__subsample': hp.uniform('subsample', 0.7, 0.9),
    'classifier__colsample_bytree': hp.uniform('colsample_bytree', 0.7, 0.9),
    'classifier__gamma': hp.uniform('gamma', 2, 4),
    'classifier__min_child_weight': hp.choice('min_child_weight', [8, 10, 12])
}

def objective(params):
    pipeline.set_params(**params)
    score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy').mean()
    return -score

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
"""

# For GradBoost and Forest

pipeline.fit(X_train, y_train)

train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)

print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-validated accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

predictions = pipeline.predict(X_test)

print(classification_report(y_test, predictions))

fitted_classifier = pipeline.named_steps["classifier"]
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

thresholds = np.sort(fitted_classifier.feature_importances_)

for thresh in thresholds:
    selection = SelectFromModel(fitted_classifier, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train_transformed)
    selection_model = clone(fitted_classifier)
    selection_model.fit(select_X_train, y_train)
    
    select_X_test = selection.transform(X_test_transformed)
    predictions = selection_model.predict(select_X_test)
    accuracy = accuracy_score(y_test, predictions)
    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))


# In[40]:


best_params = {
    'classifier__n_estimators': [300, 500, 700][best['n_estimators']],
    'classifier__max_depth': [4, 5, 6][best['max_depth']],
    'classifier__min_child_weight': [8, 10, 12][best['min_child_weight']],
    'classifier__learning_rate': best['learning_rate'],
    'classifier__subsample': best['subsample'],
    'classifier__colsample_bytree': best['colsample_bytree'],
    'classifier__gamma': best['gamma'],
}
print("Best converted parameters:", best_params)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script NFLIQ.ipynb')

