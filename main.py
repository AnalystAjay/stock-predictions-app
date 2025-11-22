import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title(" Stock Prediction Dashboard ")

# --- Fetch stock data ---
def fetch_stock_data(ticker, days=30):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    df.reset_index(inplace=True)
    if not df.empty:
        df['Tomorrow'] = df['Close'].shift(-1)
        df.dropna(inplace=True)
    return df

# --- Predict next day using train/test split ---
def predict_next_day(df, split_ratio=0.9):
    if df.empty:
        return None, None, None
    
    split_index = int(len(df) * split_ratio)
    
    # Training
    X_train = np.arange(split_index).reshape(-1,1)
    y_train = df['Close'].iloc[:split_index].values
    
    # Test
    X_test = np.arange(split_index, len(df)).reshape(-1,1)
    y_test = df['Close'].iloc[split_index:].values
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict next day
    pred_next = float(model.predict(np.array([[len(df)]]))[0])
    
    # Accuracy on last test point
    accuracy = None
    if len(y_test) > 0:
        pred_test = model.predict(X_test)
        accuracy = 1 - abs(pred_test[-1] - y_test[-1]) / y_test[-1]
    
    return pred_next, model, accuracy

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
        
        pred, model, accuracy = predict_next_day(df)
        if pred is None or model is None:
            st.warning(f"Prediction not available for {ticker}")
            continue
        
        # --- Safe metric display (TypeError & ValueError fix) ---
        if accuracy is None or (isinstance(accuracy, float) and np.isnan(accuracy)):
            accuracy_display = "N/A"
        else:
            accuracy_display = f"{float(accuracy)*100:.2f}%"
        
        st.metric(label=f"{ticker} Predicted Next Day Close", value=pred, delta=f"Accuracy: {accuracy_display}")
        
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
