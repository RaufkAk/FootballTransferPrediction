import pandas as pd
import numpy as np
import os
import unicodedata
from collections import defaultdict

DATA_DIR = '/Users/raufkutayakyildiz/Desktop/Football Transfer Prediction/archive'

def clean_name(name):
    if not isinstance(name, str): return ""
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('utf-8')
    return name.lower().strip()

def main():
    print("Loading datasets...")
    # Load Transfermarkt Players
    players_df = pd.read_csv(os.path.join(DATA_DIR, 'players.csv'), low_memory=False)
    # Load FIFA Players
    fifa_df = pd.read_csv(os.path.join(DATA_DIR, 'male_players.csv'), low_memory=False)
    
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], errors='coerce')
    fifa_df['dob'] = pd.to_datetime(fifa_df['dob'], errors='coerce')
    
    original_len = len(players_df)
    
    players_df['clean_name'] = players_df['name'].apply(clean_name)
    fifa_df['clean_long_name'] = fifa_df['long_name'].apply(clean_name)
    fifa_df['clean_short_name'] = fifa_df['short_name'].apply(clean_name)
    
    # Build dictionary grouped by DOB for O(1) matching speed
    fifa_by_dob = defaultdict(list)
    for _, row in fifa_df.dropna(subset=['dob']).iterrows():
        fifa_by_dob[row['dob']].append({
            'long_name': row['clean_long_name'],
            'short_name': row['clean_short_name'],
            'potential': row['potential'],
            'overall': row['overall'],
            'international_reputation': row['international_reputation']
        })
        
    matched_potentials = []
    matched_overalls = []
    matched_reputations = []
    
    print("Fuzzy Matching Players by DOB and Name...")
    matched_count = 0
    
    for idx, row in players_df.iterrows():
        dob = row['date_of_birth']
        name = row['clean_name']
        
        best_match = None
        if pd.notnull(dob) and dob in fifa_by_dob:
            candidates = fifa_by_dob[dob]
            for cand in candidates:
                # Basic string inclusion Logic (matches Kylian Mbappe to Kylian Mbappe Lottin)
                c_long = cand['long_name']
                c_short = cand['short_name']
                
                # Try exact word matches
                parts = name.split()
                if name in c_long or name in c_short or c_short in name:
                    best_match = cand
                    break
                elif len(parts) >= 2 and (parts[-1] in c_long and parts[0] in c_long):
                    best_match = cand
                    break
        
        if best_match:
            matched_potentials.append(best_match['potential'])
            matched_overalls.append(best_match['overall'])
            matched_reputations.append(best_match['international_reputation'])
            matched_count += 1
        else:
            matched_potentials.append(np.nan)
            matched_overalls.append(np.nan)
            matched_reputations.append(np.nan)
            
    players_df['fifa_potential'] = matched_potentials
    players_df['fifa_overall'] = matched_overalls
    players_df['fifa_reputation'] = matched_reputations
    
    print(f"Successfully matched {matched_count} players out of {original_len}!")
    
    # Save the merged dataset
    output_path = os.path.join(DATA_DIR, 'players_with_fifa.csv')
    players_df.to_csv(output_path, index=False)
    print(f"Saved merged data to {output_path}")

if __name__ == '__main__':
    main()
