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
        "('Expected Points', 'Sp. Tms')": 'Spteams_Expected_Points'
    }

    def __init__(self, df, team_name):
        self.df = df
        self.team = team_name

    def assign_year(self, start_year=2001):
        self._rename_and_drop_columns()
        self._convert_week_column()
        self._assign_year_by_week(start_year)
        self._postprocess_week_and_team()
        return self
    
    def _rename_and_drop_columns(self):
        self.df = self.df.rename(columns=self.RENAME_COLUMNS)
        self.df.drop(columns=self.df.columns[[2, 3, 5, 6, 7, 8]], inplace=True)
    
    def _convert_week_column(self):
        self.df['Week'] = self.df['Week'].replace({
            'Wild Card': -1,
            'Division': -2,
            'Conf. Champ.': -3,
            'SuperBowl': -4
        })
        self.df['Week'] = pd.to_numeric(self.df['Week'], errors='coerce')
        self.df['season_type'] = np.where(self.df['Week'] < 0, 3, 2)
    
    def _assign_year_by_week(self, start_year):
        current_year = int(start_year)
        prev_week = 0
        years = []
    
        for week in self.df['Week']:
            if (week == 1) or ((prev_week == 17) and (week == 2) and (self.team == 'TB') and (current_year == 2016)):
                current_year += 1
            years.append(current_year)
            prev_week = week
    
        self.df["Year"] = years
    
    def _postprocess_week_and_team(self):
        self.df['Team'] = self.team
        self.df.loc[(self.df['Year'] > 2008) & (self.df['Week'] == -4), 'Week'] = 5
        self.df['Week'] = self.df['Week'].abs()

    def arrange_team_json(self):
        df = self.df.copy()
        df.dropna(subset=['Opponents'], inplace=True)
        df.rename(columns={'Team': 'Home_Team'}, inplace=True)
        self.df = df
        return self

    def get_df(self):
        return self.df