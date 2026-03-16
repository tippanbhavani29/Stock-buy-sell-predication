📈 ML-Based Stock Buy/Sell Signal Generator with MLflow

A machine learning system that predicts next-day stock movement using technical indicators and generates buy/sell trading signals while tracking experiments using MLflow.

📌 Project Overview

Stock markets generate large volumes of financial data daily. Traders often rely on manual technical analysis to identify opportunities, which can be inconsistent and time-consuming.

This project builds a machine learning-based trading signal generator that:

Collects historical stock data

Generates technical indicators

Trains ML models to predict price movement

Tracks experiments using MLflow

Evaluates strategies with backtesting metrics

Generates buy/sell signals with confidence scores

🚀 Key Features

✔ Automatic stock data collection
✔ Technical indicator feature engineering
✔ Multiple ML models comparison
✔ MLflow experiment tracking
✔ Backtesting trading strategy
✔ Signal generation system
✔ Interactive visualization dashboard

📊 System Architecture
Stock Data Source
(Yahoo Finance API)
        │
        ▼
Data Ingestion
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
(SMA, RSI, MACD, Volatility)
        │
        ▼
Dataset Creation
(Target Variable)
        │
        ▼
Machine Learning Models
(Logistic Regression / Random Forest / XGBoost)
        │
        ▼
MLflow Experiment Tracking
        │
        ▼
Model Evaluation
        │
        ▼
Backtesting Engine
        │
        ▼
Trading Signal Generator
        │
        ▼
Streamlit Dashboard
🧠 Machine Learning Pipeline
Data Collection
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Dataset Preparation
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
MLflow Experiment Logging
      │
      ▼
Model Evaluation
      │
      ▼
Backtesting Strategy
      │
      ▼
Trading Signal Generation
📊 Example Output
Stock	Prediction	Confidence	Signal
RELIANCE	Up	0.82	Strong Buy
TCS	Up	0.74	Buy
INFY	Down	0.40	Sell
📈 Technical Indicators Used
Indicator	Description
SMA	Simple Moving Average
RSI	Relative Strength Index
MACD	Momentum Indicator
Volatility	Standard deviation of returns
Lagged Returns	Previous day returns
🤖 Machine Learning Models

The following models are trained and compared:

Model	Purpose
Logistic Regression	Baseline model
Random Forest	Tree ensemble model
XGBoost	High-performance boosting model
📉 Trading Strategy Evaluation

Backtesting metrics used:

Metric	Description
Sharpe Ratio	Risk-adjusted returns
Win Rate	Percentage of profitable trades
Max Drawdown	Maximum loss from peak
Cumulative Returns	Total strategy return
🔬 MLflow Experiment Tracking

MLflow tracks:

Model parameters

Training metrics

Model artifacts

Experiment runs

Example experiment log:

Model: RandomForest
n_estimators: 200
max_depth: 10
Accuracy: 0.66
Sharpe Ratio: 1.42
Win Rate: 62%

MLflow UI allows easy comparison between models.

Run MLflow UI:

mlflow ui

Open:

http://127.0.0.1:5000
📁 Project Folder Structure
stock-signal-mlflow-project

data
│
├── raw
│     stock_data.csv
│
├── processed
│     features.csv

notebooks
│
├── EDA.ipynb

src
│
├── data_loader.py
├── feature_engineering.py
├── dataset_builder.py
├── train_model.py
├── evaluate_model.py
├── backtesting.py
├── signal_generator.py

models
│
├── trained_models.pkl

mlruns
│
├── MLflow experiment logs

app
│
├── streamlit_app.py

requirements.txt
README.md
🛠 Technology Stack
Tool	Purpose
Python	Programming
Pandas	Data processing
NumPy	Numerical computation
yfinance	Stock market data
TA library	Technical indicators
Scikit-learn	ML models
XGBoost	Boosting model
MLflow	Experiment tracking
Streamlit	Dashboard
Matplotlib / Plotly	Visualization
⚙ Installation

Install dependencies:

pip install -r requirements.txt

Example dependencies:

pandas
numpy
scikit-learn
xgboost
mlflow
yfinance
ta
streamlit
matplotlib
plotly
▶ Running the Project

Step 1 — Train models

python src/train_model.py

Step 2 — Start MLflow UI

mlflow ui

Step 3 — Run dashboard

streamlit run app/streamlit_app.py
📊 Dashboard Features

The dashboard will show:

Stock price charts

Technical indicators

Model predictions

Trading signals

Model performance metrics

⚠ Limitations

Stock market is highly volatile

Model relies only on historical data

External factors (news, macro events) not included

Backtesting does not guarantee future profits

🔮 Future Improvements

Possible enhancements:

News sentiment analysis using NLP

LSTM / Transformer deep learning models

Reinforcement learning trading agents

Real-time prediction system

Portfolio optimization

🎯 Skills Demonstrated

This project demonstrates:

Machine Learning

Financial Data Analysis

Feature Engineering

Model Evaluation

MLflow (MLOps)

Experiment Tracking

Backtesting Strategies

Streamlit Dashboard