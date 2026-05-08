import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "points_predictor.pkl"
DATA_PATH = BASE_DIR / "data" / "nba_player_stats.csv"


FEATURES = [
    "minutes_played",
    "field_goal_attempts",
    "field_goal_percentage",
    "rebounds",
    "assists",
    "usage_rate",
    "last_5_games_avg_points",
]


st.set_page_config(
    page_title="NBA Player Performance Predictor",
    page_icon="🏀",
    layout="centered"
)

st.title("🏀 NBA Player Performance Predictor")
st.write(
    "Predict how many points an NBA player may score based on historical performance stats."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please run `python src/train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)

st.header("Enter Player Stats")

minutes_played = st.slider("Minutes Played", 0, 48, 32)
field_goal_attempts = st.slider("Field Goal Attempts", 0, 40, 16)
field_goal_percentage = st.slider("Field Goal Percentage", 0.0, 100.0, 47.5)
rebounds = st.slider("Rebounds", 0, 25, 6)
assists = st.slider("Assists", 0, 20, 5)
usage_rate = st.slider("Usage Rate", 0.0, 50.0, 26.0)
last_5_games_avg_points = st.slider("Last 5 Games Avg Points", 0.0, 60.0, 22.0)

input_data = pd.DataFrame(
    [[
        minutes_played,
        field_goal_attempts,
        field_goal_percentage,
        rebounds,
        assists,
        usage_rate,
        last_5_games_avg_points,
    ]],
    columns=FEATURES
)

if st.button("Predict Points"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Points: {prediction:.1f}")

st.divider()

st.header("Dataset Preview")

if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH)
    st.dataframe(df.head())

    st.subheader("Points Distribution")
    st.bar_chart(df["points"])
else:
    st.info("Dataset not found.")
