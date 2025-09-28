# Tennis Match Outcome Predictor 🎾

This project predicts the outcome of professional tennis matches using historical ATP match data.  
It demonstrates an **end-to-end data science and machine learning workflow** — from data cleaning and exploratory data analysis (EDA), to feature engineering, model building, and deployment of a lightweight predictive model.

---

## Project Structure

data/
  raw/ # Original ATP match data
  processed/ # Cleaned data used for modeling
models/
  tennis_match_predictor.pkl # High-accuracy model (not predictive, includes in-match stats)
  light_model.pkl # Lightweight predictive model
metadata/ # Saved metadata (feature columns, win rates, etc.)
01_data_cleaning.ipynb
02_data_processing.ipynb
03_EDA.ipynb
04_feature_engineering_and_modeling.ipynb
scripts/
  predict.py # Script to test lightweight model predictions
requirements.txt
README.md


---

## Features

- **Exploratory Data Analysis (EDA):** Player performance, surface differences, and match outcome patterns.
- **Feature Engineering:** Player win rates, surface performance, head-to-head stats, and rank differences.
- **Modeling:**
  - **Full model (~99% accuracy):** Demonstrates pipelines and feature engineering, but not predictive since it uses in-match stats.
  - **Lightweight model:** Uses pre-match features for realistic predictions.
- **Prediction Function:** Run matchups between any two players on any surface.

---

## Usage
Option 1: Run prediction script
python scripts/predict.py

Option 2: Import function in Python
from scripts.predict import predict_match

prob = predict_match("Nadal", "Federer", "Clay")
print(f"Probability Nadal beats Federer on Clay: {prob:.2f}")

Example prediction
Input:  Player = "Nadal", Opponent = "Federer", Surface = "Clay"
Output: Probability Nadal wins ≈ 0.72

Notes:
The full model demonstrates end-to-end ML pipelines but isn’t predictive.
The lightweight model is intended for real-world pre-match predictions.
EDA highlighted important trends like surface effects, dominance of top players, and ranking differences.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/tennis-predictor.git
cd tennis-predictor
pip install -r requirements.txt
