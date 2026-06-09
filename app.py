# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import json
import pandas as pd
from typing import Dict, Any, List

app = FastAPI(title="Transfer Value AI Predictor")

# Load model and UI data at startup
print("Loading model and UI data...")
try:
    model = joblib.load('models/transfer_value_model.pkl')
    model_features = joblib.load('models/model_features.pkl')
    with open('data/ui_player_data.json', 'r') as f:
        player_data = json.load(f)
    print(f"Loaded {len(player_data)} players successfully.")
except Exception as e:
    print(f"Error loading assets: {e}")
    model = None
    player_data = {}

class PredictRequest(BaseModel):
    player_id: str
    overrides: Dict[str, float]

@app.get("/api/search")
def search_players(q: str = ""):
    q = q.lower()
    results = []
    for pid, data in player_data.items():
        if q in data['stats']['name'].lower():
            results.append({
                "id": pid,
                "name": data['stats']['name'],
                "image_url": data['stats']['image_url'],
                "actual_value": data['stats']['actual_value'],
                "club": data['stats']['club'],
                "position": data['stats']['position']
            })
            if len(results) >= 20:
                break
    return {"results": results}

@app.get("/api/player/{player_id}")
def get_player(player_id: str):
    if player_id not in player_data:
        raise HTTPException(status_code=404, detail="Player not found")
    # Return their stats and baseline prediction
    base_features = player_data[player_id]['features']
    df = pd.DataFrame([base_features], columns=model_features)
    import numpy as np
    final_pred = np.expm1(model.predict(df)[0])
    
    return {
        "stats": player_data[player_id]['stats'],
        "predicted_value": float(final_pred)
    }

@app.post("/api/predict")
def predict_value(req: PredictRequest):
    if req.player_id not in player_data:
        raise HTTPException(status_code=404, detail="Player not found")
        
    base_features = player_data[req.player_id]['features'].copy()
    
    # Apply overrides
    for key, val in req.overrides.items():
        if key in base_features:
            base_features[key] = val
            
            # Recompute per_90 metrics if applicable
            if key == 'recent_goals' or key == 'recent_minutes':
                mins = base_features.get('recent_minutes', 0)
                goals = base_features.get('recent_goals', 0)
                base_features['recent_goals_per_90'] = (goals / (mins + 1)) * 90
                
            if key == 'fifa_potential' or key == 'fifa_overall':
                pot = base_features.get('fifa_potential', 65)
                ovr = base_features.get('fifa_overall', 65)
                base_features['potential_growth'] = pot - ovr
                
            if key == 'career_goals' or key == 'career_minutes':
                mins = base_features.get('career_minutes', 0)
                goals = base_features.get('career_goals', 0)
                base_features['career_goals_per_90'] = (goals / (mins + 1)) * 90
                
            # Recompute Age and Contract related features
            if key == 'age_at_valuation' or key == 'years_left_on_contract_at_valuation':
                age = base_features.get('age_at_valuation', 25)
                contract = base_features.get('years_left_on_contract_at_valuation', 2)
                base_features['is_veteran'] = 1 if age >= 34 else 0
                decay = max(0, age - 31)
                base_features['peak_age_decay'] = decay
                base_features['age_contract_fragility'] = decay / (contract + 0.1)
                
    # Create DataFrame with exact column order
    df = pd.DataFrame([base_features], columns=model_features)
    import numpy as np
    final_pred = np.expm1(model.predict(df)[0])
    
    return {"predicted_value": float(final_pred)}

# Mount static files to serve the frontend
import os
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
