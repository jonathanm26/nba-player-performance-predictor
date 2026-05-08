# NBA Player Performance Predictor

A machine learning project that predicts NBA player scoring performance using historical player statistics, data preprocessing, regression models, and an interactive Streamlit dashboard.

## Project Overview

This project uses NBA player statistics to predict how many points a player may score based on performance-related features such as minutes played, field goal attempts, shooting percentage, rebounds, assists, offensive usage, and recent game trends.

The goal is to build a clean, portfolio-ready AI/ML project that demonstrates:

- Data cleaning and preprocessing
- Exploratory data analysis
- Machine learning regression models
- Model evaluation
- Prediction workflow
- Interactive Streamlit dashboard

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Repository Structure

```text
nba-player-performance-predictor/
│
├── data/
│   └── nba_player_stats.csv
│
├── notebooks/
│   └── nba_model_training.ipynb
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── points_predictor.pkl
│
├── src/
│   └── train_model.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Features

- Clean NBA player performance data
- Train Linear Regression and Random Forest models
- Compare model performance using MAE, MSE, RMSE, and R² score
- Save the best model as a `.pkl` file
- Predict player points using an interactive dashboard
- Display basic data visualizations

## Target Variable

The model predicts:

```text
points
```

## Initial Input Features

```text
minutes_played
field_goal_attempts
field_goal_percentage
rebounds
assists
usage_rate
last_5_games_avg_points
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/nba-player-performance-predictor.git
cd nba-player-performance-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

```bash
# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python src/train_model.py
```

### 5. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

## Example Prediction

The user enters player statistics such as:

- Minutes played: 34
- Field goal attempts: 18
- Field goal percentage: 48.5
- Rebounds: 7
- Assists: 5
- Usage rate: 29.4
- Last 5 games average points: 25.8

The app returns an estimated points prediction for the player's next game.

## Machine Learning Models

The first version compares:

1. Linear Regression
2. Random Forest Regressor

The best-performing model is saved and used in the Streamlit dashboard.

## Future Improvements

- Add real NBA data using an API
- Add opponent defensive rating
- Add home/away game feature
- Add back-to-back game indicator
- Add injury and minutes restriction data
- Add player-specific models
- Deploy the app online using Streamlit Cloud

## Author

Daniel Mezrahi

Computer Science student interested in AI/ML, sports analytics, finance, and software development.
