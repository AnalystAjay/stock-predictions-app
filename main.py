import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("🚀 Ultimate Stock Prediction Dashboard (Fixed, No MySQL)")

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
    X = np.arange(len(df)).reshape(-1,1)  # use numeric index
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    next_day_index = np.array([[len(df)]])
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
        if pred is None or model is None:
            st.warning(f"Prediction not available for {ticker}")
            continue
        
        actual_today = df['Close'].iloc[-1] if not df.empty else None
        accuracy = calculate_accuracy(pred, actual_today)
        
        st.metric(label=f"{ticker} Predicted Next Day Close", value=pred, delta=f"Accuracy: {accuracy*100 if accuracy else 'N/A'}%")
        
        # --- Prepare dataframe for plotting ---
        df_plot = df[['Date','Close']].copy()
        df_plot.rename(columns={'Close':'Actual'}, inplace=True)
        X_numeric = np.arange(len(df)).reshape(-1,1)
        df_plot['Predicted'] = list(model.predict(X_numeric))
        df_plot['stock_id'] = ticker
        all_pred_data.append(df_plot)
    
    # --- Interactive Multi-Line Chart ---
    if all_pred_data:
        fig = go.Figure()
        for df_plot in all_pred_data:
            ticker_name = df_plot['stock_id'].iloc[0]
            # Actual line
            fig.add_trace(go.Scatter(
                x=df_plot['Date'], y=df_plot['Actual'],
                mode='lines+markers', name=f"{ticker_name} Actual"
            ))
            # Predicted line
            fig.add_trace(go.Scatter(
                x=df_plot['Date'], y=df_plot['Predicted'],
                mode='lines+markers', name=f"{ticker_name} Predicted"
            ))
        fig.update_layout(title="📊 Predicted vs Actual Close Prices",
                          xaxis_title="Date", yaxis_title="Close Price",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
