import os
import pandas as pd
import numpy as np

# Paths
BASE_DIR = "/Users/raufkutayakyildiz/Desktop/Football Transfer Prediction"
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')

print('Loading datasets...')
players_df = pd.read_csv(os.path.join(DATA_DIR, 'players_with_fifa.csv'), low_memory=False)
appearances_df = pd.read_csv(os.path.join(DATA_DIR, 'appearances.csv'), low_memory=False)
player_valuations_df = pd.read_csv(os.path.join(DATA_DIR, 'player_valuations.csv'), low_memory=False)
clubs_df = pd.read_csv(os.path.join(DATA_DIR, 'clubs.csv'), low_memory=False)
competitions_df = pd.read_csv(os.path.join(DATA_DIR, 'competitions.csv'), low_memory=False)
print('Data loaded successfully!')

print('Parsing dates and extracting targets...')
player_valuations_df['date'] = pd.to_datetime(player_valuations_df['date'])
appearances_df['date'] = pd.to_datetime(appearances_df['date'])
players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], errors='coerce')

player_valuations_df = player_valuations_df.sort_values(by=['player_id', 'date'])
player_valuations_df['previous_market_value'] = player_valuations_df.groupby('player_id')['market_value_in_eur'].shift(1)

target_df = player_valuations_df.groupby('player_id').last().reset_index()
target_df = target_df.rename(columns={'date': 'valuation_date', 'market_value_in_eur': 'target_market_value', 'current_club_id': 'club_id_at_valuation'})
target_df['previous_market_value'] = target_df['previous_market_value'].fillna(100000)

df = players_df.merge(target_df[['player_id', 'valuation_date', 'target_market_value', 'previous_market_value', 'club_id_at_valuation']], on='player_id', how='inner')

# Demographics
df['age_at_valuation'] = (df['valuation_date'] - df['date_of_birth']).dt.days / 365.25
df['age_at_valuation'] = df['age_at_valuation'].fillna(df['age_at_valuation'].median())
df['valuation_year'] = df['valuation_date'].dt.year
df['contract_expiration_date'] = pd.to_datetime(df['contract_expiration_date'], errors='coerce')
df['years_left_on_contract_at_valuation'] = (df['contract_expiration_date'] - df['valuation_date']).dt.days / 365.25
df['years_left_on_contract_at_valuation'] = df['years_left_on_contract_at_valuation'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)

# Time-Aware Performance
apps_merged = appearances_df.merge(target_df[['player_id', 'valuation_date']], on='player_id', how='inner')
career_apps = apps_merged[apps_merged['date'] < apps_merged['valuation_date']]
career_agg = career_apps.groupby('player_id').agg(
    career_appearances=('appearance_id', 'count'), career_goals=('goals', 'sum'),
    career_assists=('assists', 'sum'), career_minutes=('minutes_played', 'sum'),
    career_yellow_cards=('yellow_cards', 'sum'), career_red_cards=('red_cards', 'sum')
).reset_index()

recent_apps = career_apps[(career_apps['valuation_date'] - career_apps['date']).dt.days <= 365]
recent_agg = recent_apps.groupby('player_id').agg(
    recent_appearances=('appearance_id', 'count'), recent_goals=('goals', 'sum'),
    recent_assists=('assists', 'sum'), recent_minutes=('minutes_played', 'sum')
).reset_index()

df = df.merge(career_agg, on='player_id', how='left').merge(recent_agg, on='player_id', how='left')
df.fillna({'career_appearances':0, 'career_goals':0, 'career_assists':0, 'career_minutes':0, 'career_yellow_cards':0, 'career_red_cards':0, 'recent_appearances':0, 'recent_goals':0, 'recent_assists':0, 'recent_minutes':0}, inplace=True)

# Club Data
clubs_comp = clubs_df[['club_id', 'domestic_competition_id', 'squad_size', 'average_age', 'foreigners_number', 'foreigners_percentage', 'national_team_players', 'stadium_seats']]
clubs_comp = clubs_comp.merge(competitions_df[['competition_id', 'sub_type']], left_on='domestic_competition_id', right_on='competition_id', how='left')
clubs_comp.rename(columns={'sub_type': 'competition_tier'}, inplace=True)
df = df.merge(clubs_comp, left_on='club_id_at_valuation', right_on='club_id', how='left')

# Advanced & FIFA Features
df['height_in_cm'] = df['height_in_cm'].fillna(df['height_in_cm'].median())
df['career_goals_per_90'] = (df['career_goals'] / (df['career_minutes'] + 1)) * 90
df['recent_goals_per_90'] = (df['recent_goals'] / (df['recent_minutes'] + 1)) * 90
df['is_veteran'] = (df['age_at_valuation'] >= 34).astype(int)
df['peak_age_decay'] = df['age_at_valuation'].apply(lambda age: max(0, age - 31))
df['age_contract_fragility'] = df['peak_age_decay'] / (df['years_left_on_contract_at_valuation'] + 0.1)

df['fifa_reputation'] = df['fifa_reputation'].fillna(1.0)
df['fifa_overall'] = df['fifa_overall'].fillna(df['fifa_overall'].median())
df['fifa_potential'] = df['fifa_potential'].fillna(df['fifa_overall'])
df['potential_growth'] = df['fifa_potential'] - df['fifa_overall']

print('Feature engineering completed!')

# Descriptive stats
cat_cols = ['sub_position', 'position', 'foot', 'country_of_citizenship', 'competition_tier']
num_cols = ['height_in_cm', 'age_at_valuation', 'valuation_year', 'years_left_on_contract_at_valuation',
            'career_appearances', 'career_goals', 'career_assists', 'career_minutes', 'career_yellow_cards', 'career_red_cards', 
            'recent_appearances', 'recent_goals', 'recent_assists', 'recent_minutes',
            'career_goals_per_90', 'recent_goals_per_90', 'is_veteran', 'peak_age_decay', 'age_contract_fragility',
            'squad_size', 'average_age', 'foreigners_number', 'foreigners_percentage', 'national_team_players', 'stadium_seats',
            'previous_market_value', 'fifa_reputation', 'potential_growth', 'target_market_value']

stats_df = df[num_cols].describe().transpose()
stats_df.to_csv("/Users/raufkutayakyildiz/Desktop/Football Transfer Prediction/data/descriptive_statistics.csv")
print('Descriptive stats saved to data/descriptive_statistics.csv')
