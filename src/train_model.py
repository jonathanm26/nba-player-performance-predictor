import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "nba_player_stats.csv"
MODEL_PATH = BASE_DIR / "models" / "points_predictor.pkl"


FEATURES = [
    "minutes_played",
    "field_goal_attempts",
    "field_goal_percentage",
    "rebounds",
    "assists",
    "usage_rate",
    "last_5_games_avg_points",
]

TARGET = "points"


def load_data(path: Path) -> pd.DataFrame:
    """Load NBA player stats dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset by removing missing values and keeping required columns."""
    required_columns = FEATURES + [TARGET]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    df = df.dropna(subset=required_columns)
    return df


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Evaluate a regression model."""
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results = {
        "model": name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2_score": r2,
    }

    print(f"\n{name} Results")
    print("-" * 30)
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

    return results


def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        results.append(evaluate_model(name, model, X_test, y_test))

    best_result = min(results, key=lambda x: x["rmse"])
    best_model_name = best_result["model"]
    best_model = trained_models[best_model_name]

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nBest model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
