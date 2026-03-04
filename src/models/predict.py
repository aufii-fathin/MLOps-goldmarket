import pandas as pd
import yfinance as yf
import joblib

# LOAD MODEL & SCALER
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# FETCH LATEST DATA
df = yf.download("GC=F", period="30d", interval="1d")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
    df = df.reset_index()

# FEATURE ENGINEERING (SAME AS PREPROCESS)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")
df = df.drop_duplicates(subset="Date")

df["return"] = df["Close"].pct_change()

df["lag_1"] = df["Close"].shift(1)
df["lag_2"] = df["Close"].shift(2)
df["lag_3"] = df["Close"].shift(3)
df["lag_5"] = df["Close"].shift(5)
df["lag_7"] = df["Close"].shift(7)
df["lag_10"] = df["Close"].shift(10)

df["rolling_mean_7"] = df["Close"].rolling(7).mean()
df["rolling_std_7"] = df["Close"].rolling(7).std()

df["volatility_20"] = df["return"].rolling(20).std()

df = df.dropna()

# TAKE LATEST ROW
latest = df.iloc[-1:]

X = latest.drop(columns=["Date","Close"])

# SCALE
X_scaled = scaler.transform(X)

# PREDICT
prediction = model.predict(X_scaled)

print("\nLatest gold price:", latest["Close"].values[0])
print("Predicted next price:", prediction[0])