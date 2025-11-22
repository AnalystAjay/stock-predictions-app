import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import mysql.connector
from mysql.connector import Error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
RAW_CSV_URL = "https://raw.githubusercontent.com/AnalystAjay/stock-predictions-app/main/sp500.csv"

MYSQL_CONFIG = {
    "host": st.secrets["mysql"]["Mysql@localhost"],
    "port": st.secrets["mysql"]["3306"],
    "user": st.secrets["mysql"]["root"],
    "password": st.secrets["mysql"]["Sumo$%23"],
    "database": st.secrets["mysql"]["sp500 database"],
}

MYSQL_TABLE = st.secrets["mysql"]["table"]


# -------------------------------------------------------
# LOAD DATA LOGIC
# -------------------------------------------------------
def load_from_mysql():
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        query = f"SELECT * FROM {MYSQL_TABLE}"
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            raise ValueError("MySQL table is empty")

        st.success("Loaded data from MySQL successfully ✔️")
        return df

    except Exception as e:
        st.warning(f"MySQL load failed: {e}")
        return None


def load_from_csv():
    try:
        st.info("Loading fallback CSV from GitHub…")
        df = pd.read_csv(RAW_CSV_URL)
        st.success("Loaded CSV from GitHub successfully ✔️")
        return df
    except Exception as e:
        st.error(f"Failed to load fallback CSV: {e}")
        return None


def ensure_datetime(df):
    """Fixes date parsing warnings"""
    if "Date" in df.columns:
        try:
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
        except:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# -------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------
def add_features(df):
    df = df.sort_values("Date")
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Volatility"] = df["Close"].pct_change().rolling(10).std()

    df = df.dropna()
    return df


# -------------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------------
def train_model(df):
    X = df[["Close", "MA5", "MA20", "Volatility"]]
    y = df["Tomorrow"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, mse, r2, X_test.index, preds


# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("📈 Stock Prediction Dashboard")
st.write("Powered by MySQL + Streamlit + Machine Learning")

# Load data
df = load_from_mysql()
if df is None:
    df = load_from_csv()

if df is None:
    st.error("❌ No data available from MySQL or CSV. Cannot run the app.")
    st.stop()

df = ensure_datetime(df)
df = add_features(df)

# Show data preview
with st.expander("🔍 View Raw Data"):
    st.dataframe(df)

# Train model
model, mse, r2, idx, preds = train_model(df)

st.subheader("📊 Model Performance")
st.write(f"**MSE:** {mse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")

# Plot predictions
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.loc[idx, "Date"], y=preds, name="Predicted"))
fig.add_trace(go.Scatter(x=df.loc[idx, "Date"], y=df.loc[idx, "Tomorrow"], name="Actual"))
fig.update_layout(title="Prediction vs Actual", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig, use_container_width=True)
