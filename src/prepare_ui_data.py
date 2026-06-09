import os
import pandas as pd
import joblib
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data')

def main():
    print("Preparing UI Data for End-to-End Model...")
    players_df = pd.read_csv(os.path.join(DATA_DIR, 'players_with_fifa.csv'), low_memory=False)
    appearances_df = pd.read_csv(os.path.join(DATA_DIR, 'appearances.csv'), low_memory=False)
    player_valuations_df = pd.read_csv(os.path.join(DATA_DIR, 'player_valuations.csv'), low_memory=False)
    clubs_df = pd.read_csv(os.path.join(DATA_DIR, 'clubs.csv'), low_memory=False)
    competitions_df = pd.read_csv(os.path.join(DATA_DIR, 'competitions.csv'), low_memory=False)
    
    player_valuations_df['date'] = pd.to_datetime(player_valuations_df['date'])
    appearances_df['date'] = pd.to_datetime(appearances_df['date'])
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], errors='coerce')
    
    player_valuations_df = player_valuations_df.sort_values(by=['player_id', 'date'])
    player_valuations_df['previous_market_value'] = player_valuations_df.groupby('player_id')['market_value_in_eur'].shift(1)
    
    target_df = player_valuations_df.groupby('player_id').last().reset_index()
    target_df = target_df.rename(columns={'date': 'valuation_date', 'market_value_in_eur': 'target_market_value', 'current_club_id': 'club_id_at_valuation'})
    target_df['previous_market_value'] = target_df['previous_market_value'].fillna(100000)
    
    # Keep the relevant features
    df = players_df.merge(target_df[['player_id', 'valuation_date', 'target_market_value', 'previous_market_value', 'club_id_at_valuation']], on='player_id', how='inner')
    
    df['age_at_valuation'] = (df['valuation_date'] - df['date_of_birth']).dt.days / 365.25
    df['age_at_valuation'] = df['age_at_valuation'].fillna(df['age_at_valuation'].median())
    df['valuation_year'] = df['valuation_date'].dt.year
    df['contract_expiration_date'] = pd.to_datetime(df['contract_expiration_date'], errors='coerce')
    df['years_left_on_contract_at_valuation'] = (df['contract_expiration_date'] - df['valuation_date']).dt.days / 365.25
    df['years_left_on_contract_at_valuation'] = df['years_left_on_contract_at_valuation'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)
    
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
    
    df = df.merge(career_agg, on='player_id', how='left')
    df = df.merge(recent_agg, on='player_id', how='left')
    perf_cols = ['career_appearances', 'career_goals', 'career_assists', 'career_minutes', 'career_yellow_cards', 'career_red_cards', 'recent_appearances', 'recent_goals', 'recent_assists', 'recent_minutes']
    df[perf_cols] = df[perf_cols].fillna(0)
    
    clubs_comp = clubs_df[['club_id', 'domestic_competition_id', 'squad_size', 'average_age', 'foreigners_number', 'foreigners_percentage', 'national_team_players', 'stadium_seats']]
    clubs_comp = clubs_comp.merge(competitions_df[['competition_id', 'sub_type']], left_on='domestic_competition_id', right_on='competition_id', how='left')
    clubs_comp['foreigners_percentage'] = pd.to_numeric(clubs_comp['foreigners_percentage'], errors='coerce')
    clubs_comp.rename(columns={'sub_type': 'competition_tier'}, inplace=True)
    
    df = df.merge(clubs_comp, left_on='club_id_at_valuation', right_on='club_id', how='left')
    df['competition_tier'] = df['competition_tier'].fillna('Unknown')
    
    df['height_in_cm'] = df['height_in_cm'].fillna(df['height_in_cm'].median())
    df['career_goals_per_90'] = (df['career_goals'] / (df['career_minutes'] + 1)) * 90
    df['recent_goals_per_90'] = (df['recent_goals'] / (df['recent_minutes'] + 1)) * 90
    
    df['is_veteran'] = (df['age_at_valuation'] >= 34).astype(int)
    df['peak_age_decay'] = df['age_at_valuation'].apply(lambda age: max(0, age - 31))
    df['age_contract_fragility'] = df['peak_age_decay'] / (df['years_left_on_contract_at_valuation'] + 0.1)
    
    # Handle missing FIFA data
    df['fifa_reputation'] = df['fifa_reputation'].fillna(1.0)
    df['fifa_overall'] = df['fifa_overall'].fillna(df['fifa_overall'].median())
    df['fifa_potential'] = df['fifa_potential'].fillna(df['fifa_overall'])
    df['potential_growth'] = df['fifa_potential'] - df['fifa_overall']
    
    cat_cols = ['sub_position', 'position', 'foot', 'country_of_citizenship', 'competition_tier']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    
    top_countries = df['country_of_citizenship'].value_counts().head(20).index
    df['country_of_citizenship'] = df['country_of_citizenship'].apply(lambda x: x if x in top_countries else 'Other')
    
    # We prep the features for inference
    num_cols = ['height_in_cm', 'age_at_valuation', 'valuation_year', 'years_left_on_contract_at_valuation',
                'career_appearances', 'career_goals', 'career_assists', 'career_minutes', 
                'career_yellow_cards', 'career_red_cards', 
                'recent_appearances', 'recent_goals', 'recent_assists', 'recent_minutes',
                'career_goals_per_90', 'recent_goals_per_90',
                'is_veteran', 'peak_age_decay', 'age_contract_fragility',
                'squad_size', 'average_age', 'foreigners_number', 
                'foreigners_percentage', 'national_team_players', 'stadium_seats',
                'previous_market_value', 'fifa_reputation', 'potential_growth']
    
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    encoded_df = pd.get_dummies(df[num_cols + cat_cols], drop_first=True, columns=cat_cols)
    
    # Store exact model features per player
    features_list = joblib.load(os.path.join(MODEL_DIR, 'model_features.pkl'))
    # Add any missing one-hot encoded columns
    for col in features_list:
        if col not in encoded_df.columns:
            encoded_df[col] = 0
            
    encoded_df = encoded_df[features_list] # Ordered perfectly for inference
    
    # Fast filtering and JSON mapping
    df = df.sort_values(by='target_market_value', ascending=False)
    ui_df = df.head(5000).copy()
    
    player_features_map = {}
    for _, row in ui_df.iterrows():
        p_id = row['player_id']
        feature_vector = encoded_df.loc[row.name].to_dict()
        player_features_map[str(p_id)] = {
            'features': feature_vector,
            'stats': {
                'name': row['name'],
                'image_url': row['image_url'],
                'actual_value': row['target_market_value'],
                'position': row['position'],
                'club': row['current_club_name'],
                'age': round(row['age_at_valuation'], 1),
                'recent_goals': int(row['recent_goals']),
                'recent_assists': int(row['recent_assists']),
                'contract_years': round(row['years_left_on_contract_at_valuation'], 1),
                'career_minutes': int(row['career_minutes']),
                'reputation': float(row['fifa_reputation']),
                'potential': float(row['fifa_potential']),
                'overall': float(row['fifa_overall'])
            }
        }
        
    output_path = os.path.join(PROCESSED_DATA_DIR, 'ui_player_data.json')
    with open(output_path, 'w') as f:
        json.dump(player_features_map, f)
        
    print("UI Data saved. Total profiles:", len(player_features_map))

if __name__ == "__main__":
    main()
