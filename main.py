import streamlit as st
import yfinance as yf
import pandas as pd
import mysql.connector
from mysql.connector import Error
from sklearn.linear_model import LinearRegression
from datetime import datetime
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title(" Stock Prediction Dashboard (MySQL Fixed)")

# --- MySQL connection ---
def get_connection():
    try:
        conn = mysql.connector.connect(
            host='127.0.0.1',        # Use 127.0.0.1 instead of 'localhost'
            user='root',
            password='Sumo$%23',
            database='sp500 database',
            port=3306
        )
        if conn.is_connected():
            st.success("🎉 Connected to MySQL successfully!")
            return conn
    except Error as e:
        st.error(f"❌ Error connecting to MySQL: {e}")
        return None

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

# --- Save prediction to MySQL ---
def save_prediction(conn, ticker, pred, actual=None):
    cursor = conn.cursor()
    date_today = datetime.today().strftime('%Y-%m-%d')
    accuracy = 1.0 if actual is None else round(1 - abs(pred-actual)/actual,4)
    query = """
    INSERT INTO predictions (stock_id, date, predicted_close, actual_close, model_name, accuracy)
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (ticker, date_today, pred, actual, 'LinearRegression', accuracy))
    conn.commit()

# --- Main App ---
tickers_input = st.text_input("Enter Stock Tickers (comma separated)", "^GSPC, AAPL, MSFT")
days_input = st.number_input("Days of History", min_value=10, max_value=365, value=30)

if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    conn = get_connection()
    
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
        
        st.metric(label=f"{ticker} Predicted Next Day Close", value=pred)
        
        # Save prediction
        if conn:
            actual_today = df['Close'].iloc[-1] if not df.empty else None
            save_prediction(conn, ticker, pred, actual_today)
            
            # Fetch all predictions for chart
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE stock_id=%s ORDER BY date ASC", (ticker,))
            rows = cursor.fetchall()
            if rows:
                df_pred = pd.DataFrame(rows, columns=[i[0] for i in cursor.description])
                all_pred_data.append(df_pred)
    
    # --- Interactive Multi-Line Chart ---
    if all_pred_data:
        fig = go.Figure()
        for df_pred in all_pred_data:
            ticker_name = df_pred['stock_id'].iloc[0]
            # Predicted line
            fig.add_trace(go.Scatter(
                x=df_pred['date'], y=df_pred['predicted_close'],
                mode='lines+markers', name=f"{ticker_name} Predicted"
            ))
            # Actual line if available
            if 'actual_close' in df_pred.columns:
                fig.add_trace(go.Scatter(
                    x=df_pred['date'], y=df_pred['actual_close'],
                    mode='lines+markers', name=f"{ticker_name} Actual"
                ))
        fig.update_layout(title="📊 Predicted vs Actual Close Prices",
                          xaxis_title="Date", yaxis_title="Close Price",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    
    if conn:
        conn.close()
