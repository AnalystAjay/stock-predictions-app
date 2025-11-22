import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🚀 Ultimate Stock Prediction Dashboard (No MySQL)")

# --- Fetch stock data from yfinance ---
def fetch_stock_data(ticker, days=30):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    df.reset_index(inplace=True)
    if not df.empty:
        df['Tomorrow'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
    return df

# --- Predict next day using Linear Regression ---
def predict_next_day(df):
    if df.empty:
        return None, None
    X = df.index.values.reshape(-1,1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    next_day_index = [[len(df)]]
    pred = float(model.predict(next_day_index)[0])
    return round(pred,2), model

# --- Accuracy calculation ---
def calculate_accuracy(pred, actual):
    if actual is None:
        return None
    return round(1 - abs(pred - actual)/actual, 4)

# --- Main App ---
tickers_input = st.text_input("Enter Stock Tickers (comma separated)", "^GSPC, AAPL, MSFT")
days_input = st.number_input("Days of History", min_value=10, max_value=365, value=30)

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    all_pred_data = []
    
    for ticker in tickers:
        df = fetch_stock_data(ticker, days_input)
        
        if df.empty:
            st.warning(f"No data available for {ticker}. Skipping.")
            continue
        
        st.subheader(f"📈 Last {days_input} Days Data: {ticker}")
        st.dataframe(df[['Date','Open','High','Low','Close','Volume']])
        
        pred, model = predict_next_day(df)
        if pred is None:
            st.warning(f"Prediction not available for {ticker}")
            continue
        
        actual_today = df['Close'].iloc[-1] if not df.empty else None
        accuracy = calculate_accuracy(pred, actual_today)
        
        st.metric(label=f"{ticker} Predicted Next Day Close", value=pred, delta=f"Accuracy: {accuracy*100 if accuracy else 'N/A'}%")
        
        # Prepare dataframe for plotting
        df_plot = df[['Date','Close']].copy()
        df_plot.rename(columns={'Close':'Actual'}, inplace=True)
        df_plot['Predicted'] = list(model.predict(df.index.values.reshape(-1,1)))
        df_plot['stock_id'] = ticker
        all_pred_data.append(df_plot)
    
    # --- Interactive Multi-Line Chart ---
    if
