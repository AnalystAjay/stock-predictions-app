import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
import datetime

st.set_page_config(layout="wide")
st.title("🚀 Professional Stock Dashboard (Candlestick + Prediction)")

# --- Session state for auto-refresh ---
if 'last_update' not in st.session_state:
    st.session_state.last_update = pd.Timestamp.now()

refresh_interval = st.number_input("Auto-refresh interval (seconds)", min_value=30, max_value=600, value=300)

st.write(f"⏱ Last updated: {st.session_state.last_update}")

# --- Fetch stock data ---
def fetch_stock_data(ticker, days=60):
    df = yf.download(ticker, period=f"{days}d", interval="1d", progress=False)
    df.reset_index(inplace=True)
    if not df.empty:
        df['Tomorrow'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
    return df

# --- Predict next day ---
def predict_next_day(df, split_ratio=0.9):
    if df.empty:
        return None, None, None
    split_index = int(len(df)*split_ratio)
    X_train = np.arange(split_index).reshape(-1,1)
    y_train = df['Close'].iloc[:split_index].values
    X_test = np.arange(split_index,len(df)).reshape(-1,1)
    y_test = df['Close'].iloc[split_index:].values
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred_next = float(model.predict(np.array([[len(df)]]))[0])
    accuracy = None
    if len(y_test)>0:
        pred_test = model.predict(X_test)
        accuracy = 1 - abs(pred_test[-1]-y_test[-1])/y_test[-1]
    return pred_next, model, accuracy

# --- Main App ---
tickers_input = st.text_input("Enter Stock Tickers (comma separated)", "AAPL, MSFT, ^GSPC")
days_input = st.number_input("Days of History for chart", min_value=20, max_value=365, value=60)

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    all_pred_data = []

    for ticker in tickers:
        df = fetch_stock_data(ticker, days_input)
        if df.empty:
            st.warning(f"No data for {ticker}")
            continue

        st.subheader(f"📈 Last {days_input} Days Data: {ticker}")
        st.dataframe(df[['Date','Open','High','Low','Close','Volume']])

        pred, model, accuracy = predict_next_day(df)
        if pred is None:
            st.warning(f"Prediction not available for {ticker}")
            continue

        # Safe metric display
        if accuracy is None or (isinstance(accuracy,float) and np.isnan(accuracy)):
            accuracy_display = "N/A"
        else:
            accuracy_display = f"{float(accuracy)*100:.2f}%"
        st.metric(label=f"{ticker} Predicted Next Day Close", value=pred, delta=f"Accuracy: {accuracy_display}")

        # --- Candlestick Chart ---
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df['Date'],
            open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
            name="Candlestick"
        ))
        # --- Predicted vs Actual Line ---
        X_numeric = np.arange(len(df)).reshape(-1,1)
        df['Predicted'] = list(model.predict(X_numeric))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Close', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], mode='lines', name='Predicted Close', line=dict(color='orange', dash='dot')))
        # --- Volume bars ---
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume', yaxis='y2', opacity=0.3))
        # Layout
        fig.update_layout(
            title=f"{ticker} Candlestick + Predicted vs Actual Close",
            xaxis_title="Date",
            yaxis_title="_
