# main.py — MySQL-native version (with CSV fallback)
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

# -----------------------------
# Config / defaults
# -----------------------------
st.set_page_config(page_title="Stock Prediction App (MySQL)", layout="wide")
st.title("📈 Stock Prediction App (MySQL)")
st.write("This app loads stock data from your **MySQL** database (preferred). If DB fails, it will fallback to a local CSV.")

# default fallback CSV (your uploaded file)
FALLBACK_CSV = "/mnt/data/26514515-6823-4304-bb53-2dee7e12ef12.csv"

# Read DB credentials from environment (set these in Streamlit Secrets or env vars)
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "3306")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "Sumo$%23")  # default from earlier message
DB_NAME = os.environ.get("DB_NAME", "sp500")
DB_TABLE = os.environ.get("DB_TABLE", "")  # if empty, code will try common names

# helper to build sqlalchemy URL
def make_engine():
    user = DB_USER
    pw = DB_PASSWORD
    host = DB_HOST
    port = DB_PORT
    db = DB_NAME
    url = f"mysql+mysqlconnector://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, connect_args={"connect_timeout": 5})

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data_from_mysql():
    """
    Attempts to load data from MySQL. Returns (df, source) on success, or (None, error) on failure.
    If DB_TABLE env var is blank, we try a list of common table names.
    """
    try:
        engine = make_engine()
    except Exception as e:
        return None, f"engine_error:{e}"

    candidates = []
    if DB_TABLE and DB_TABLE.strip():
        candidates.append(DB_TABLE.strip())

    # common fallback table names
    candidates.extend(["sp500_data", "sp500", "stock_data", "stocks", "sp500_table"])

    # dedupe
    candidates = list(dict.fromkeys(candidates))

    for tbl in candidates:
        try:
            query = f"SELECT * FROM `{tbl}` LIMIT 1000000"
            df = pd.read_sql(query, con=engine)
            if df is not None and len(df) > 0:
                return df, f"mysql:{tbl}"
        except SQLAlchemyError:
            # table doesn't exist or permission denied — try next
            continue
        except Exception:
            continue

    return None, "no_table_found"

def load_fallback_csv(path=FALLBACK_CSV):
    # try comma, then tab
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, sep="\t")
    return df

# -----------------------------
# Data normalization + features
# -----------------------------
def normalize_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # fix column whitespace and common names
    df.columns = [c.strip() for c in df.columns]

    # find date column
    date_col = None
    for c in df.columns:
        if c.lower() == "date" or "date" in c.lower():
            date_col = c
            break

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col)
        df = df.set_index(date_col)
    else:
        # keep whatever index exists, or set RangeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

    # Find Close-like column
    close_col = None
    for cand in ["Close", "close", "Adj Close", "Adj_Close", "ClosePrice", "close_price"]:
        if cand in df.columns:
            close_col = cand
            break
    # also check lowercase matches
    if close_col is None:
        for c in df.columns:
            if c.lower() == "close":
                close_col = c
                break

    if close_col is None:
        raise ValueError("No 'Close' column found in data. Columns: " + ", ".join(df.columns))

    # unify names for ML pipeline
    if close_col != "Close":
        df = df.rename(columns={close_col: "Close"})

    # ensure Open/High/Low/Volume exist (create NaN columns if not)
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    # convert numeric columns
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # features
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    df["MA_5"] = df["Close"].rolling(5).mean()
    df["MA_10"] = df["Close"].rolling(10).mean()
    df["Return"] = df["Close"].pct_change()

    df = df.dropna()
    return df

# -----------------------------
# Model training
# -----------------------------
def train_model(df):
    features = ["Open", "High", "Low", "Close", "Volume", "MA_5", "MA_10", "Return"]
    features = [f for f in features if f in df.columns]
    X = df[features]
    y = df["Tomorrow"]

    # chronological split
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return model, X_test.index, preds, mse, r2

# -----------------------------
# Main app flow
# -----------------------------
st.sidebar.header("Database settings (env vars)")
st.sidebar.write("If you want to override defaults, set DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_TABLE in your environment or Streamlit Secrets.")

st.info("Attempting to load data from MySQL...")

df = None
source = None

with st.spinner("Connecting to MySQL..."):
    df, status = load_data_from_mysql()

if df is None:
    st.warning(f"MySQL load failed: {status}. Falling back to local CSV at `{FALLBACK_CSV}`.")
    try:
        df = load_fallback_csv(FALLBACK_CSV)
        source = "csv_fallback"
        st.success("Loaded fallback CSV.")
    except Exception as e:
        st.error(f"Failed to load fallback CSV: {e}")
        st.stop()
else:
    source = status  # e.g. "mysql:sp500_data"
    st.success(f"Loaded data from {source}")

# prepare data
try:
    df_prepared = normalize_and_prepare(df)
except Exception as e:
    st.error(f"Data preparation failed: {e}")
    st.write("Columns available:", df.columns)
    st.stop()

st.subheader("Data preview (prepared)")
st.dataframe(df_prepared.head())

# train
with st.spinner("Training model..."):
    model, pred_idx, preds, mse, r2 = train_model(df_prepared)

st.subheader("Model metrics")
c1, c2 = st.columns(2)
c1.metric("MSE", f"{mse:.4f}")
c2.metric("R²", f"{r2:.4f}")

# plot
st.subheader("Predicted vs Actual (test set)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=pred_idx, y=df_prepared.loc[pred_idx, "Tomorrow"], mode="lines", name="Actual"))
fig.add_trace(go.Scatter(x=pred_idx, y=preds, mode="lines", name="Predicted"))
fig.update_layout(xaxis_title="Date", yaxis_title="Price", height=600)
st.plotly_chart(fig, use_container_width=True)

# predict next day
st.subheader("Predict next day")
last_row = df_prepared.iloc[-1:]
needed = [f for f in ["Open","High","Low","Close","Volume","MA_5","MA_10","Return"] if f in last_row.columns]
next_pred = model.predict(last_row[needed])[0]
st.success(f"Predicted next day close: {next_pred:.2f}")

st.info(f"Data source: {source}")
