import pandas as pd
import joblib

# Load model + metadata
light_model = joblib.load("models/light_model.pkl")
df_model = pd.read_csv("metadata/df_model_light.csv")

# Load dictionaries (player_win_pct, h2h_wins, etc.)
player_win_pct = joblib.load("metadata/player_win_pct.pkl")
h2h_wins = joblib.load("metadata/h2h_wins.pkl")
h2h_matches = joblib.load("metadata/h2h_matches.pkl")
win_surface = joblib.load("metadata/win_surface.pkl")
matches_surface = joblib.load("metadata/matches_surface.pkl")
feature_cols = joblib.load("metadata/feature_cols.pkl")


# === Prediction function (unchanged) ===
def predict_match(player, opponent, surface, df_model=df_model, model=light_model):
    """Predicts probability of player winning against opponent on given surface."""

    def safe_lookup(mapping, key, default=0.5):
        return mapping.get(key, default)

    player_win = safe_lookup(player_win_pct, player)
    opponent_win = safe_lookup(player_win_pct, opponent)
    h2h = safe_lookup(h2h_wins, (player, opponent)) / safe_lookup(h2h_matches, (player, opponent), 1)
    player_surface = safe_lookup(win_surface, (player, surface)) / safe_lookup(matches_surface, (player, surface), 1)
    opponent_surface = safe_lookup(win_surface, (opponent, surface)) / safe_lookup(matches_surface, (opponent, surface), 1)

    # Rank difference
    player_rank_val = df_model.loc[df_model['player_name'] == player, 'winner_rank'].min()
    opponent_rank_val = df_model.loc[df_model['player_name'] == opponent, 'winner_rank'].min()
    if pd.isna(player_rank_val): player_rank_val = 1000
    if pd.isna(opponent_rank_val): opponent_rank_val = 1000
    rank_diff_val = player_rank_val - opponent_rank_val

    X_new = pd.DataFrame([[
        player_win, opponent_win, h2h, player_surface, opponent_surface, rank_diff_val
    ]], columns=feature_cols)

    prob = model.predict_proba(X_new)[0][1]  # Probability of player winning
    return prob



if __name__ == "__main__":
    player = input("Enter player name: ")
    opponent = input("Enter opponent name: ")
    surface = input("Enter surface (Hard/Clay/Grass): ")

    prob = predict_match(player, opponent, surface)
    print(f"\nPredicted probability of {player} beating {opponent} on {surface}: {prob:.2%}")
