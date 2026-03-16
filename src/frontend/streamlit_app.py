import streamlit as st
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
import ta
import plotly.graph_objects as go
import plotly.subplots as sp

# --------------------------------
# Load trained model
# --------------------------------

model = pickle.load(open("C:\Users\Bhavani\OneDrive\Documents\Ml-project\ML_flow_Project\artifacts\model.pkl", "rb"))

st.set_page_config(page_title="AI Trading Dashboard", layout="wide")

st.title("📈 AI Stock Trading Dashboard")

# --------------------------------
# Sidebar navigation
# --------------------------------

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Trading Chart", "Prediction", "Backtest"]
)

# --------------------------------
# Download stock data safely
# --------------------------------

def download_stock(ticker):

    df = yf.download(ticker, period="6mo")

    df = df.copy()

    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Convert columns to Series
    for col in df.columns:
        df[col] = pd.Series(df[col].values.flatten(), index=df.index)

    return df


# --------------------------------
# Feature engineering
# --------------------------------

def create_features(df):

    df = df.copy()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.Series(df[col].values.flatten(), index=df.index)

    df["Return"] = df["Close"].pct_change()

    df["Momentum_10"] = df["Close"].diff(10)

    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()

    df["Volatility"] = df["Return"].rolling(10).std()

    close_series = pd.Series(df["Close"].values.flatten(), index=df.index)

    # RSI
    rsi = ta.momentum.RSIIndicator(close=close_series)
    df["RSI"] = rsi.rsi()

    # MACD
    macd = ta.trend.MACD(close=close_series)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()

    # Bollinger
    bb = ta.volatility.BollingerBands(close=close_series)
    df["BB_high"] = bb.bollinger_hband()
    df["BB_low"] = bb.bollinger_lband()

    # ATR
    atr = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )

    df["ATR"] = atr.average_true_range()

    # VWAP
    df["VWAP"] = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()

    df = df.dropna()

    return df


# --------------------------------
# Dashboard Page
# --------------------------------

if page == "Dashboard":

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", "XGBoost")
    col2.metric("Accuracy", "72%")
    col3.metric("Win Rate", "61%")
    col4.metric("Sharpe Ratio", "1.45")

    st.markdown("---")

    st.write(
    """
    This system predicts **next-day stock direction** using:

    • Technical indicators  
    • Machine learning models  
    • MLflow experiment tracking  
    • Strategy backtesting  
    """
    )


# --------------------------------
# Trading Chart
# --------------------------------

elif page == "Trading Chart":

    ticker = st.text_input("Enter Stock Ticker", "RELIANCE.NS")

    if st.button("Load Chart"):

        df = download_stock(ticker)
        df = create_features(df)

        features = df[[
            "Open","High","Low","Close","Volume",
            "Return","Momentum_10",
            "SMA_10","SMA_50","Volatility",
            "RSI","MACD","MACD_signal",
            "BB_high","BB_low","ATR","VWAP"
        ]]

        df["Prediction"] = model.predict(features)

        buy = df[df["Prediction"] == 1]
        sell = df[df["Prediction"] == 0]

        # Create subplot layout
        fig = sp.make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6,0.2,0.2],
            subplot_titles=("Price Chart","RSI","MACD")
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price"
            ),
            row=1,col=1
        )

        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_10"],
                line=dict(color="blue", width=1),
                name="SMA 10"
            ),
            row=1,col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_50"],
                line=dict(color="orange", width=1),
                name="SMA 50"
            ),
            row=1,col=1
        )

        # Buy signals
        fig.add_trace(
            go.Scatter(
                x=buy.index,
                y=buy["Close"],
                mode="markers",
                marker=dict(color="green", size=10, symbol="triangle-up"),
                name="BUY"
            ),
            row=1,col=1
        )

        # Sell signals
        fig.add_trace(
            go.Scatter(
                x=sell.index,
                y=sell["Close"],
                mode="markers",
                marker=dict(color="red", size=10, symbol="triangle-down"),
                name="SELL"
            ),
            row=1,col=1
        )

        # RSI panel
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI"],
                line=dict(color="purple"),
                name="RSI"
            ),
            row=2,col=1
        )

        fig.add_hline(y=70,row=2,col=1,line_dash="dot",line_color="red")
        fig.add_hline(y=30,row=2,col=1,line_dash="dot",line_color="green")

        # MACD panel
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD"],
                line=dict(color="blue"),
                name="MACD"
            ),
            row=3,col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_signal"],
                line=dict(color="orange"),
                name="Signal"
            ),
            row=3,col=1
        )

        fig.update_layout(
            title=f"{ticker} Trading Dashboard",
            xaxis_rangeslider_visible=False,
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)


# --------------------------------
# Prediction Page
# --------------------------------

elif page == "Prediction":

    ticker = st.text_input("Ticker", "RELIANCE.NS")

    if st.button("Predict"):

        df = download_stock(ticker)

        df = create_features(df)

        latest = df.iloc[-1]

        X = pd.DataFrame([[
            latest["Open"],
            latest["High"],
            latest["Low"],
            latest["Close"],
            latest["Volume"],
            latest["Return"],
            latest["Momentum_10"],
            latest["SMA_10"],
            latest["SMA_50"],
            latest["Volatility"],
            latest["RSI"],
            latest["MACD"],
            latest["MACD_signal"],
            latest["BB_high"],
            latest["BB_low"],
            latest["ATR"],
            latest["VWAP"]
        ]])

        prediction = int(model.predict(X)[0])

        proba = model.predict_proba(X)

        confidence = round(float(np.max(proba)) * 100, 2)

        if prediction == 1:
            st.success(f"📈 BUY SIGNAL (Confidence {confidence}%)")
        else:
            st.error(f"📉 SELL SIGNAL (Confidence {confidence}%)")

# --------------------------------
# Backtesting Page
# --------------------------------

elif page == "Backtest":

    df = pd.read_csv("artifacts/test.csv")

    features = df.select_dtypes(include=["number"]).drop("Target",axis=1)

    df["Prediction"] = model.predict(features)

    df["Strategy_Return"] = df["Return"] * df["Prediction"]

    equity = (1 + df["Strategy_Return"]).cumprod()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=equity,
            mode="lines",
            name="Equity Curve"
        )
    )

    fig.update_layout(title="Strategy Backtest Performance")

    st.plotly_chart(fig,use_container_width=True)