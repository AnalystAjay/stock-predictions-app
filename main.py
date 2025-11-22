import streamlit as st
import yfinance as yf
import pandas as pd
import mysql.connector
from mysql.connector import Error
from sklearn.linear_model import LinearRegression
from datetime import datetime
import plotly.express as px

st.set_page_config(layout="wide")
st.title("🚀 Pro Real-Time Stock Prediction Dashboard")

# --- MySQL connection ---
def get_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='ajay',
            password='Ajay@123',
            database='stock_app'
        )
        return conn
    except Error as e:
        st.error(f"MySQL Connection Error: {e}")
        return None

# --- Fetch stock data from yfinance ---
def fetch_stock_data(ticker, days=30):
    df = yf.download(ticker, period=f"{days}d", interval="1d")
    df.reset_index(inplace=True)
    df['Tomorrow'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    return df

# --- Train Linear Regression and predict next day ---
def predict_next_day(df):
    X = df.index.values.reshape(-1,1)
    y = df['Close'].values
    model = LinearRegression()
    model.fit(X, y)
    next_day_index = [[len(df)]]
    pred = model.predict(next_day_index)[0]
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
if tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    conn = get_connection()
    
    all_pred_data = []
    for ticker in tickers:
        df = fetch_stock_data(ticker)
        st.subheader(f"📈 Last 30 Days: {ticker}")
        st.dataframe(df[['Date','Open','High','Low','Close','Volume']])
        
        pred, model = predict_next_day(df)
        st.metric(label=f"{ticker} Predicted Next Day Close", value=pred)
        
        if conn:
            save_prediction(conn, ticker, pred)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE stock_id=%s ORDER BY date DESC", (ticker,))
            rows = cursor.fetchall()
            if rows:
                df_pred = pd.DataFrame(rows, columns=[i[0] for i in cursor.description])
                all_pred_data.append(df_pred)
    
    # --- Combined Interactive Chart ---
    if all_pred_data:
        combined_df = pd.concat(all_pred_data)
        fig = px.line(combined_df, x="date", y="predicted_close", color="stock_id",
                      title="📊 Predicted Close Prices for Multiple Stocks")
        st.plotly_chart(fig, use_container_width=True)
    
    if conn:
        conn.close()
