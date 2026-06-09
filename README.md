# AI Football Transfer Value Predictor (V2 Model)

An end-to-end Machine Learning web application designed to predict and simulate football player market values. The system combines player performance metrics, demographics, club stats, and FIFA datasets to estimate market valuations and provides an interactive "What-If" simulator for real-time adjustments.

## 🚀 Key Features

*   **Predictive Engine:** Uses an optimized `HistGradientBoostingRegressor` model trained on player historical records and appearances.
*   **"What-If" Simulator:** An interactive dashboard where users can adjust player parameters (Age, Recent Goals, Recent Assists, Contract length, and Minutes Played) and see how the AI dynamically updates their valuation.
*   **Auto-Complete Search:** Easily search across thousands of players in the indexed database.
*   **Glassmorphic UI:** Modern web dashboard with dark mode aesthetics, clean gradients, and responsive layouts.
*   **Comprehensive Data Pipeline:** Automated scripts for merging databases, feature engineering (log-transformed targets, decay functions, fragility indexes), and model validation.

---

## 🎬 Demo

Here is a quick demonstration of the web application interface and the real-time "What-If" simulator in action:

[📥 **Click here to download / watch the App Demo Video (MP4)**](https://github.com/RaufkAk/FootballTransferPrediction/raw/main/app_demo.mp4)

---

## 🛠️ Technology Stack

*   **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python), Pandas, NumPy, Scikit-Learn, Joblib, Uvicorn
*   **Frontend:** Vanilla HTML5, CSS3 (Custom Glassmorphism, Google Fonts `Outfit`, animations), Javascript (ES6 Fetch API)
*   **Data Analysis & Modeling:** Jupyter Notebooks, Matplotlib, Seaborn, Scikit-Learn Ensemble methods

---

## 📂 Repository Structure

```text
├── app.py                      # FastAPI Web Application & API Gateway
├── requirements.txt            # Python environment dependencies
├── .gitignore                  # Git exclusion rules (ignores venv and large CSVs)
├── data/                       # Preprocessed statistics and UI cache
│   ├── ui_player_data.json     # Player indexed features for fast UI search
│   └── descriptive_statistics.csv
├── models/                     # Trained ML models and feature mapping
│   ├── transfer_value_model.pkl
│   └── model_features.pkl
├── notebooks/                  # Research, analysis, and prototyping
│   ├── EDA.ipynb               # Exploratory Data Analysis
│   └── Model_Training.ipynb    # Model training and hyperparameter tuning
├── src/                        # Data preparation & training pipelines
│   ├── training.py             # Model training and outlier validation pipeline
│   ├── prepare_ui_data.py      # Transforms processed records to web UI structure
│   └── merge_fifa_data.py      # Merges Transfermarkt stats with FIFA database
├── static/                     # Web Frontend Assets
│   ├── index.html              # Dashboard Layout
│   ├── style.css               # Styling and Animations
│   └── script.js               # Event handlers and API integration
└── eda_outputs/                # Data visualizations generated during analysis
    ├── correlation_heatmap.png
    ├── target_distribution.png
    ├── feature_importance_solid.png
    └── pca_cumulative_variance.png
```

---

## 📊 Machine Learning Pipeline

### 1. Feature Engineering
The pipeline engineers predictive features across several domains:
*   **Demographics:** Age at valuation, height, and citizenship.
*   **Contractual Status:** Years remaining on contract, veteran flags (age $\ge$ 34), and age-contract fragility.
*   **Performance Metrics:** Career and recent (last 365 days) appearances, goals, assists, minutes played, and goals per 90 minutes.
*   **Club and League Details:** Squad size, average age, stadium capacity, and competition tier.
*   **FIFA Stats:** Player overall rating, potential rating, potential growth, and international reputation.

### 2. Modeling
*   **Target Variable:** Log-transformed market value (`np.log1p(target_market_value)`) to handle right-skewness and stabilize predictions.
*   **Algorithm:** `HistGradientBoostingRegressor` (Scikit-Learn) tuned for robust handling of categorical variables and missing entries.
*   **Evaluation Metrics:** R-squared ($R^2$) and Root Mean Squared Error (RMSE) interpreted back in real Euros.

---

## ⚡ Getting Started

### Prerequisites
*   Python 3.8 or higher installed on your system.
*   To train the model from scratch, raw data files (`players.csv`, `appearances.csv`, etc.) must be placed in the `data/raw/` directory (excluded from Git).

### Installation
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/RaufkAk/FootballTransferPrediction.git
    cd FootballTransferPrediction
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Web App
Start the FastAPI server:
```bash
uvicorn app:app --reload
```

Once running, navigate to **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

---

## 📈 Model Training
To retrain the model and regenerate artifacts:
```bash
python src/training.py
```
This will train the regressor, output the test validation scores (RMSE & R²), validate predictions against reference outliers (e.g. Arda Güler, Mauro Icardi), and export model binaries under `models/`.
