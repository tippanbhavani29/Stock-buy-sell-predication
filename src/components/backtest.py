import pandas as pd
import pickle
import numpy as np


def run_backtest():

    print("Loading model...")

    model = pickle.load(open("artifacts/model.pkl", "rb"))

    print("Loading test data...")

    df = pd.read_csv("artifacts/test.csv")

    # Remove non-numeric columns
    drop_cols = ["Date", "Ticker", "Target"]

    features = [col for col in df.columns if col not in drop_cols]

    X = df[features]

    y = df["Target"]

    # Generate predictions
    df["Prediction"] = model.predict(X)

    # Strategy return
    df["Strategy_Return"] = df["Return"] * df["Prediction"]

    # Total return
    total_return = df["Strategy_Return"].sum()

    # Win rate
    win_rate = (df["Strategy_Return"] > 0).mean()

    # Sharpe ratio
    sharpe = df["Strategy_Return"].mean() / df["Strategy_Return"].std()

    # Max drawdown
    cumulative = (1 + df["Strategy_Return"]).cumprod()

    peak = cumulative.cummax()

    drawdown = (cumulative - peak) / peak

    max_drawdown = drawdown.min()

    print("\n===== Backtest Results =====")

    print("Total Return:", round(total_return, 4))

    print("Win Rate:", round(win_rate, 4))

    print("Sharpe Ratio:", round(sharpe, 4))

    print("Max Drawdown:", round(max_drawdown, 4))


if __name__ == "__main__":

    run_backtest()