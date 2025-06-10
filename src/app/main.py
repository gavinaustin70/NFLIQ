# ================================================================
#                                                                |
#    |||\  |||    |||||||||  |||        ||||||   ///|||\\\       |
#    |||\\ |||    |||        |||          ||    ///     \\\      |
#    ||| \\|||    ||||||     |||          ||    |||      |||     |
#    |||  \\||    |||        |||          ||    \\\   \\\///     |
#    |||   |||    |||        |||||||||  ||||||   \\\|||\\\\\\\   |
#                                                                |
# ================================================================
# ======================
#                      |
# Author: Gavin Austin |
#                      |
#=======================

from src.scraper.fetch_upcoming_games import get_upcoming_data
from src.preprocessing.preprocessing import preprocess_data

if __name__ == "__main__":
    get_upcoming_data()
    preprocess_data()