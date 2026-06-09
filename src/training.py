import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging

# Configure logging for professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

def load_datasets():
    """Loads all necessary raw datasets for feature engineering."""
    logging.info("Loading datasets...")
    try:
        players_df = pd.read_csv(os.path.join(DATA_DIR, 'players_with_fifa.csv'), low_memory=False)
        appearances_df = pd.read_csv(os.path.join(DATA_DIR, 'appearances.csv'), low_memory=False)
        player_valuations_df = pd.read_csv(os.path.join(DATA_DIR, 'player_valuations.csv'), low_memory=False)
        clubs_df = pd.read_csv(os.path.join(DATA_DIR, 'clubs.csv'), low_memory=False)
        competitions_df = pd.read_csv(os.path.join(DATA_DIR, 'competitions.csv'), low_memory=False)
        return players_df, appearances_df, player_valuations_df, clubs_df, competitions_df
    except FileNotFoundError as e:
        logging.error(f"Dataset not found: {e}")
        raise

def preprocess_and_engineer_features(players_df, appearances_df, player_valuations_df, clubs_df, competitions_df):
    """Processes raw data, creates valuation targets, and engineers historical/performance features."""
    logging.info("Parsing dates...")
    player_valuations_df['date'] = pd.to_datetime(player_valuations_df['date'])
    appearances_df['date'] = pd.to_datetime(appearances_df['date'])
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], errors='coerce')
    
    logging.info("Extracting Valuation Events...")
    player_valuations_df = player_valuations_df.sort_values(by=['player_id', 'date'])
    player_valuations_df['previous_market_value'] = player_valuations_df.groupby('player_id')['market_value_in_eur'].shift(1)
    
    target_df = player_valuations_df.groupby('player_id').last().reset_index()
    target_df = target_df.rename(columns={
        'date': 'valuation_date',
        'market_value_in_eur': 'target_market_value',
        'current_club_id': 'club_id_at_valuation'
    })
    
    target_df['previous_market_value'] = target_df['previous_market_value'].fillna(100000)
    target_df = target_df[['player_id', 'valuation_date', 'target_market_value', 'previous_market_value', 'club_id_at_valuation']]
    logging.info(f"Total Valuation Targets identified: {len(target_df)}")
    
    df = players_df.merge(target_df, on='player_id', how='inner')
    
    logging.info("Calculating Demographics at Valuation Time...")
    df['age_at_valuation'] = (df['valuation_date'] - df['date_of_birth']).dt.days / 365.25
    df['age_at_valuation'] = df['age_at_valuation'].fillna(df['age_at_valuation'].median())
    
    df['valuation_year'] = df['valuation_date'].dt.year
    df['contract_expiration_date'] = pd.to_datetime(df['contract_expiration_date'], errors='coerce')
    df['years_left_on_contract_at_valuation'] = (df['contract_expiration_date'] - df['valuation_date']).dt.days / 365.25
    df['years_left_on_contract_at_valuation'] = df['years_left_on_contract_at_valuation'].apply(lambda x: max(0, x) if pd.notnull(x) else 0)
    
    logging.info("Processing Appearances using Time-Aware Filtering...")
    apps_merged = appearances_df.merge(target_df[['player_id', 'valuation_date']], on='player_id', how='inner')
    
    career_apps = apps_merged[apps_merged['date'] < apps_merged['valuation_date']]
    career_agg = career_apps.groupby('player_id').agg(
        career_appearances=('appearance_id', 'count'),
        career_goals=('goals', 'sum'),
        career_assists=('assists', 'sum'),
        career_minutes=('minutes_played', 'sum'),
        career_yellow_cards=('yellow_cards', 'sum'),
        career_red_cards=('red_cards', 'sum')
    ).reset_index()
    
    recent_apps = career_apps[(career_apps['valuation_date'] - career_apps['date']).dt.days <= 365]
    recent_agg = recent_apps.groupby('player_id').agg(
        recent_appearances=('appearance_id', 'count'),
        recent_goals=('goals', 'sum'),
        recent_assists=('assists', 'sum'),
        recent_minutes=('minutes_played', 'sum')
    ).reset_index()
    
    df = df.merge(career_agg, on='player_id', how='left')
    df = df.merge(recent_agg, on='player_id', how='left')
    
    perf_cols = ['career_appearances', 'career_goals', 'career_assists', 'career_minutes', 
                 'career_yellow_cards', 'career_red_cards', 
                 'recent_appearances', 'recent_goals', 'recent_assists', 'recent_minutes']
    df[perf_cols] = df[perf_cols].fillna(0)
    
    logging.info("Joining Club & League Data...")
    clubs_comp = clubs_df[['club_id', 'domestic_competition_id', 'squad_size', 'average_age', 'foreigners_number', 
                           'foreigners_percentage', 'national_team_players', 'stadium_seats']]
    clubs_comp = clubs_comp.merge(competitions_df[['competition_id', 'sub_type']], left_on='domestic_competition_id', right_on='competition_id', how='left')
    clubs_comp['foreigners_percentage'] = pd.to_numeric(clubs_comp['foreigners_percentage'], errors='coerce')
    clubs_comp.rename(columns={'sub_type': 'competition_tier'}, inplace=True)
    
    df = df.merge(clubs_comp, left_on='club_id_at_valuation', right_on='club_id', how='left')
    df['competition_tier'] = df['competition_tier'].fillna('Unknown')
    
    logging.info("Finalizing Feature Engineering (Including FIFA integrations)...")
    df['height_in_cm'] = df['height_in_cm'].fillna(df['height_in_cm'].median())
    
    df['career_goals_per_90'] = (df['career_goals'] / (df['career_minutes'] + 1)) * 90
    df['recent_goals_per_90'] = (df['recent_goals'] / (df['recent_minutes'] + 1)) * 90
    
    df['is_veteran'] = (df['age_at_valuation'] >= 34).astype(int)
    df['peak_age_decay'] = df['age_at_valuation'].apply(lambda age: max(0, age - 31))
    df['age_contract_fragility'] = df['peak_age_decay'] / (df['years_left_on_contract_at_valuation'] + 0.1)
    
    # Handle missing FIFA features gracefully
    df['fifa_reputation'] = df['fifa_reputation'].fillna(1.0)
    df['fifa_overall'] = df['fifa_overall'].fillna(df['fifa_overall'].median())
    df['fifa_potential'] = df['fifa_potential'].fillna(df['fifa_overall'])
    df['potential_growth'] = df['fifa_potential'] - df['fifa_overall']
    
    cat_cols = ['sub_position', 'position', 'foot', 'country_of_citizenship', 'competition_tier']
    for col in cat_cols:
        df[col] = df[col].fillna('Unknown')
    
    top_countries = df['country_of_citizenship'].value_counts().head(20).index
    df['country_of_citizenship'] = df['country_of_citizenship'].apply(lambda x: x if x in top_countries else 'Other')
    
    num_cols = ['height_in_cm', 'age_at_valuation', 'valuation_year', 'years_left_on_contract_at_valuation',
                'career_appearances', 'career_goals', 'career_assists', 'career_minutes', 
                'career_yellow_cards', 'career_red_cards', 
                'recent_appearances', 'recent_goals', 'recent_assists', 'recent_minutes',
                'career_goals_per_90', 'recent_goals_per_90',
                'is_veteran', 'peak_age_decay', 'age_contract_fragility',
                'squad_size', 'average_age', 'foreigners_number', 
                'foreigners_percentage', 'national_team_players', 'stadium_seats',
                'previous_market_value',
                'fifa_reputation', 'potential_growth']
    
    final_df = df[num_cols + cat_cols].copy()
    
    for col in num_cols:
        final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
        final_df[col] = final_df[col].fillna(final_df[col].median())
        
    final_df = pd.get_dummies(final_df, drop_first=True, columns=cat_cols)
    
    X = final_df
    y = np.log1p(df['target_market_value']) # Log-transforming the target variable for stability
    
    return X, y, df

def train_and_evaluate(X, y):
    """Splits data, trains the gradient boosting model, and evaluates performance."""
    logging.info(f"Final Features Data Shape: {X.shape}")
    logging.info("Splitting Data into Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logging.info("Training Model (HistGradientBoostingRegressor)...")
    model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    logging.info("Evaluating Model Performance...")
    y_pred_log = model.predict(X_test)
    
    # Reverse log transform to interpret metrics in real EUR
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)
    
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2 = r2_score(y_test_real, y_pred_real)
    
    logging.info(f"Test RMSE: {rmse:,.2f} EUR")
    logging.info(f"Test R-squared (R2): {r2:.4f}")
    
    return model

def validate_outliers(model, X, df):
    """Tests the model on known high-value outliers as a sanity check."""
    logging.info("Running Outlier Validation Tests...")
    players_to_check = ['Arda Güler', 'Barış Alper Yılmaz', 'Mauro Icardi']
    
    for player_name in players_to_check:
        player_data = df[df['name'].str.contains(player_name, na=False, case=False)]
        if not player_data.empty:
            idx = player_data.index[0]
            actual = player_data.iloc[0]['target_market_value']
            pred = np.expm1(model.predict(X.loc[[idx]]))[0]
            logging.info(f"Validation ({player_name}) - Actual: {actual:,.0f} EUR | Predicted: {pred:,.0f} EUR")

def main():
    """Main execution flow for the training pipeline."""
    logging.info("Initiating Training Pipeline...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    players_df, appearances_df, player_valuations_df, clubs_df, competitions_df = load_datasets()
    
    X, y, df = preprocess_and_engineer_features(players_df, appearances_df, player_valuations_df, clubs_df, competitions_df)
    
    trained_model = train_and_evaluate(X, y)
    
    # Save artifacts
    logging.info("Exporting Model Artifacts...")
    model_path = os.path.join(MODEL_DIR, 'transfer_value_model.pkl')
    features_path = os.path.join(MODEL_DIR, 'model_features.pkl')
    
    joblib.dump(trained_model, model_path)
    joblib.dump(list(X.columns), features_path)
    logging.info("Pipeline Execution Completed Successfully.")
    
    # Optional Validation Run
    validate_outliers(trained_model, X, df)

if __name__ == "__main__":
    main()
