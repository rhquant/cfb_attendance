# CFB Attendance Predictor

A statistical model that predicts college football home game attendance, with an interactive Streamlit app for exploring 2026 Texas Longhorns scenarios.

## What It Does

- **Predicts attendance** for any home game based on factors like win/loss record, opponent quality, kickoff time, weather, rivalries, and TV coverage
- **Interactive app** lets you toggle game outcomes, adjust conditions, and see how projected attendance changes in real time
- **Compares to actuals** with a cumulative attendance chart benchmarked against the 2024 season

## Model Overview

| Metric | Value |
|--------|-------|
| Training Data | 3,947 home games across 67 teams (2015-2023) |
| R-squared | 0.715 |
| Holdout Accuracy (2024) | Within ~7.3% of actual attendance |
| Typical Error | ~8,300 fans per game |

Key drivers: season record, opponent drawing power, rivalry games, kickoff time, weather, and TV coverage. Notably, losses hurt attendance 3x more than wins help it.

## Project Structure

```
cfb_attendance/
  cfb_hierarchical_regression.py   # Data pipeline + model training
  cfb_attendance_analysis.py       # Attendance visualizations (one-sheeter)
  texas_prediction_app.py          # Streamlit prediction app
  model_coefficients.json          # Exported model coefficients
  model_data.csv                   # Training dataset
  hierarchical_results.txt         # Full model results report
```

## Running the App

```bash
pip install streamlit pandas numpy altair
streamlit run texas_prediction_app.py
```

## Running the Model Pipeline

Requires a [CollegeFootballData.com](https://collegefootballdata.com/) API key:

```bash
export CFB_API_KEY="your_api_key_here"
python cfb_hierarchical_regression.py
```

## Data Source

Game data sourced from the [CollegeFootballData.com API](https://collegefootballdata.com/), covering attendance, rankings, weather, TV coverage, and coaching changes.
