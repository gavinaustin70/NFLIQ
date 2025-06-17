import numpy as np
import pandas as pd

class Team:
    RENAME_COLUMNS = {
        "('Unnamed: 0_level_0', 'Week')": 'Week',
        "('Unnamed: 1_level_0', 'Day')": 'Day',
        "('Unnamed: 4_level_0', 'Unnamed: 4_level_1')": 'boxscore',
        "('Unnamed: 9_level_0', 'Opp')": 'Opponents',
        "('Score', 'Tm')": 'Home_Score',
        "('Score', 'Opp')": 'Away_Score',
        "('Offense', '1stD')": 'Offensive_1sts',
        "('Offense', 'TotYd')": 'Offense_Total_Yrds',
        "('Offense', 'PassY')": 'Offense_Pass_Yrds',
        "('Offense', 'RushY')": 'Offense_Rush_Yrds',
        "('Offense', 'TO')": 'Turnovers_Lost',
        "('Defense', '1stD')": 'Defense_1st_Allowed',
        "('Defense', 'TotYd')": 'Total_Yrds_Allowed',
        "('Defense', 'PassY')": 'Pass_Yrds_Allowed',
        "('Defense', 'RushY')": 'Rush_Yrds_Allowed',
        "('Defense', 'TO')": 'Turnovers_Gained',
        "('Expected Points', 'Offense')": 'Offense_Expected_Points',
        "('Expected Points', 'Defense')": 'Defense_Expected_Points',
        "('Expected Points', 'Sp. Tms')": 'Spteams_Expected_Points',
        r"('Year', '')": 'Year'
    }

    def __init__(self, df, team_name):
        self.df = df
        self.team = team_name

    def preprocess_team(self):
        self._rename_and_drop_columns()
        self._convert_week_column()
        self._postprocess_week_and_team()
        return self
    
    def _rename_and_drop_columns(self):
        self.df = self.df.rename(columns=self.RENAME_COLUMNS)
        self.df.drop(columns=self.df.columns[[2, 3, 5, 6, 7, 8]], inplace=True)
        self.df.dropna(subset=['Opponents'], inplace=True)
    
    def _convert_week_column(self):
        self.df['Week'] = self.df['Week'].replace({
            'Wild Card': -1,
            'Division': -2,
            'Conf. Champ.': -3,
            'SuperBowl': -4
        })
        self.df['Week'] = pd.to_numeric(self.df['Week'], errors='coerce')
        self.df['season_type'] = np.where(self.df['Week'] < 0, 3, 2)
    
    def _postprocess_week_and_team(self):
        self.df['Home_Team'] = self.team
        self.df.loc[(self.df['Year'] > 2008) & (self.df['Week'] == -4), 'Week'] = 5
        self.df['Week'] = self.df['Week'].abs()

    def get_df(self):
        return self.df