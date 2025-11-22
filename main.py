# main.py — MySQL + CSV fallback (FIXED VERSION)

import os
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

# ----------------------------------------------------
# Streamlit Config
# ----------------------------------------------------
st.set_page_config(page_title="Stock Prediction App (MySQL)", layout="wide")
st.title("📈 Stock Prediction App (MySQL + CSV fallback)")

st.write("This app loads stock data from **MySQL**. If MySQL fails, it will load fallback CSV from GitHub.")

# ----------------------------------------------------
# FIXED: Correct fallback CSV URL
# ----------------------------------------------------
FALLBACK_CSV = "https://raw.githubusercontent.com/AnalystAjay/stock-predictions-app/main/sp500.csv"

# ----------------------------------------------------
# Database Credentials (Environment Variables)
# ----------------------------------------------------
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "3306")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "sp500")
DB_TABLE = os.environ.get("DB_TABLE", "")

def make_engine():
    url = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url, connect_args={"connect_timeout": 5})

# ----------------------------------------------------
# MySQL Loader
# ----------------------------------------------------
@st.cache_data
def load_data_from_mysql():
    try:
        engine = make_engine()
    except Exception as e:
        return None, f"engine_error:{e}"

    # Try user-provided table first, then fallback names
    candidates = []
    if DB_TABLE.strip():
        candidates.append(DB_TABLE.strip())
    candidates.extend(["sp500_data", "sp500", "stock_data", "stocks", "sp500_table"])

    seen = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    for tbl in candidates:
        try:
            df = pd.read_sql(f"SELECT * FROM `{tbl}` LIMIT 1000000", con=engine)
            if df is not None and len(df) > 0:
                return df, f"mysql:{tbl}"
        except Exception:
            continue

    return None, "no_table_found"

# ----------------------------------------------------
# CSV Fallback Loader (GitHub RAW)
# ----------------------------------------------------
def load_fallback_csv(path=FALLBACK_CSV):
    try:
        return pd.read_csv(path)
    except:
        return pd.read_csv(path, sep="\t")

# ----------------------------------------------------
# Prepare Data
# ----------------------------------------------------
def normalize_and_prepare(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Find date column
    date_col = None
    for c in df.columns:
        if c.lower() == "date" or "date" in c.lower():
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).set_index(date_col)

    # Find Close column
    close_col = None
    for cand in ["Close", "close", "Adj Close", "Adj_Close", "ClosePrice", "close_price"]:
        if cand in df.columns:
            close_col = cand
            break

    if close_col is None:
        for c in df.columns:
            if c.lower() == "close":
                close_col = c
                break

    if close_col is None:
        raise ValueError("No 'Close' column found in dataset.")

    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    # Ensure base columns exist
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # Convert numeric
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Features
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["Return"] = df["Close"].pct_change()

    df = df.dropna()
    return df

# ----------------------------------------------------
# Train Model
# ----------------------------------------------------
def train_model(df):
    features = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "Return"]
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df["Tomorrow"]

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, X_test.index, preds, mse, r2

# ----------------------------------------------------
# Main Execution
# ----------------------------------------------------
st.info("Attempting to load data from MySQL...")

with st.spinner("Connecting to MySQL..."):
    df, status = load_data_from_mysql()

if df is None:
    st.warning(f"MySQL load failed: {status}. Loading fallback CSV...")
    try:
        df = load_fallback_csv()
        source = "csv_fallback"
        st.success("Loaded fallback CSV successfully!")
    except Exception as e:
        st.error(f"CSV load failed: {e}")
        st.stop()
else:
    source = status
    st.success(f"Loaded data from {source}")

# Prepare
try:
    df_prepared = normalize_and_prepare(df)
except Exception as e:
    st.error(f"Data preparation failed: {e}")
    st.write("Available columns:", df.columns)
    st.stop()

st.subheader("Data preview")
st.dataframe(df_prepared.head())

# Train
with st.spinner("Training model..."):
    model, pred_idx, preds, mse, r2 = train_model(df_prepared)

st.subheader("Model Metrics")
c1, c2 = st.columns(2)
c1.metric("MSE", f"{mse:.4f}")
c2.metric("R² Score", f"{r2:.4f}")

# Plot
st.subheader("Predicted vs Actual")
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_idx, y=df_prepared.loc[pred_idx, "Tomorrow"], name="Actual"))
fig.add_trace(go.Scatter(x=pred_idx, y=preds, name="Predicted"))
fig.update_layout(height=600)
st.plotly_chart(fig, use_container_width=True)

# Predict next day
st.subheader("Next Day Prediction")
last_row = df_prepared.iloc[-1:]
features = [f for f in ["Open","High","Low","Close","Volume","MA_5","MA_10","Return"] if f in last_row]
next_pred = model.predict(last_row[features])[0]

st.success(f"Predicted next day Close Price: {next_pred:.2f}")

st.info(f"Data loaded from: {source}")
