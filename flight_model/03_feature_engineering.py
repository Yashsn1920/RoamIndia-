"""
03_feature_engineering.py
--------------------------
Turns the cleaned CSV into a model-ready feature matrix.
Run this AFTER 01_clean_data.py.

Key transforms:
  - Target encoding for high-cardinality categoricals (city, airline)
  - Season + holiday features from month
  - Log-transform the target (price_inr) for better regression
  - Train/test split (temporal where possible, else random)

Outputs:
  data/processed/X_train.csv
  data/processed/X_test.csv
  data/processed/y_train.csv
  data/processed/y_test.csv
  models/encoders.pkl   ← save encoders so prediction uses same transforms
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split

DATA_PATH = "data/processed/flights_clean.csv"
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows for feature engineering\n")


# ─── 1. DATE-DERIVED FEATURES ─────────────────────────────────────────────────
def get_season(month):
    """India-specific season buckets."""
    if month in [10, 11, 12, 1, 2, 3]:
        return "peak"      # winter tourism peak
    elif month in [4, 5, 6]:
        return "summer"
    else:
        return "monsoon"

HOLIDAY_MONTHS = [10, 11, 12, 1]   # Diwali, Christmas, New Year
df["season"] = df["journey_month"].apply(
    lambda m: get_season(m) if m > 0 else "unknown"
)
df["is_holiday_month"] = df["journey_month"].apply(
    lambda m: 1 if m in HOLIDAY_MONTHS else 0
)

# Advance booking buckets (feature for the model + interpretability)
def advance_bucket(d):
    if d < 0:   return "unknown"
    if d <= 3:  return "last_minute"
    if d <= 14: return "short"
    if d <= 45: return "medium"
    return "long"

df["advance_bucket"] = df["days_advance"].apply(advance_bucket)


# ─── 2. ROUTE FEATURE ─────────────────────────────────────────────────────────
df["route"] = df["source_city"] + "__" + df["destination_city"]


# ─── 3. TARGET: LOG-TRANSFORM PRICE ──────────────────────────────────────────
# XGBoost works better when the target is more normally distributed.
# We'll predict log(price+1) and exponentiate at inference.
df["log_price"] = np.log1p(df["price_inr"])


# ─── 4. TRAIN / TEST SPLIT ────────────────────────────────────────────────────
# DS1 has no journey date — use random split.
# Ideally: older dates → train, newer → test (temporal split).
# Here we just do 80/20 stratified by airline to ensure coverage.
train_df, test_df = train_test_split(
    df, test_size=0.20, random_state=42,
    stratify=df["airline"].where(df["airline"].isin(
        df["airline"].value_counts()[lambda x: x > 10].index
    ), other="Other")
)
print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")


# ─── 5. TARGET ENCODING (fit on train only, transform both) ──────────────────
# Replaces city/airline/route strings with their mean log_price.
# This handles high cardinality without exploding feature dimensions.

class TargetEncoder:
    """Simple mean target encoder with smoothing."""
    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.global_mean = None
        self.mapping = {}

    def fit(self, X: pd.Series, y: pd.Series):
        self.global_mean = y.mean()
        stats = pd.DataFrame({"col": X, "target": y}).groupby("col")["target"]
        count = stats.count()
        mean  = stats.mean()
        # Apply smoothing: weight towards global mean for rare categories
        smooth = (count * mean + self.smoothing * self.global_mean) / (count + self.smoothing)
        self.mapping = smooth.to_dict()
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        return X.map(self.mapping).fillna(self.global_mean)

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        self.fit(X, y)
        return self.transform(X)


COLS_TO_ENCODE = ["airline", "source_city", "destination_city", "route"]
encoders = {}

for col in COLS_TO_ENCODE:
    enc = TargetEncoder(smoothing=10)
    train_df[f"{col}_enc"] = enc.fit_transform(train_df[col], train_df["log_price"])
    test_df[f"{col}_enc"]  = enc.transform(test_df[col])
    encoders[col] = enc
    print(f"  Encoded: {col}  ({train_df[col].nunique()} unique values)")


# ─── 6. ORDINAL ENCODING ──────────────────────────────────────────────────────
season_map   = {"peak": 2, "summer": 1, "monsoon": 0, "unknown": -1}
advance_map  = {"last_minute": 0, "short": 1, "medium": 2, "long": 3, "unknown": -1}

for split in [train_df, test_df]:
    split["season_enc"]   = split["season"].map(season_map)
    split["advance_enc"]  = split["advance_bucket"].map(advance_map)


# ─── 7. FINAL FEATURE LIST ────────────────────────────────────────────────────
FEATURE_COLS = [
    # Target-encoded categoricals
    "airline_enc",
    "source_city_enc",
    "destination_city_enc",
    "route_enc",
    # Numerical
    "total_stops",
    "duration_hrs",
    "dep_hour",
    "is_business",
    "days_advance",          # raw value (model handles -1 as unknown)
    # Date-derived
    "journey_month",         # raw month (1–12, or -1 if unknown)
    "journey_day",
    "season_enc",
    "advance_enc",
    "is_holiday_month",
]

X_train = train_df[FEATURE_COLS].copy()
X_test  = test_df[FEATURE_COLS].copy()
y_train = train_df["log_price"]
y_test  = test_df["log_price"]

# Fill any remaining NaNs (journey_month == -1 is intentional — keep as is)
X_train = X_train.fillna(-1)
X_test  = X_test.fillna(-1)

print(f"\nFeature matrix shape: train={X_train.shape}, test={X_test.shape}")
print(f"Target (log_price): mean={y_train.mean():.3f}, std={y_train.std():.3f}")

# Save
X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv",   index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv",   index=False)

# Save encoders — MUST be saved so prediction script uses same mappings
joblib.dump(encoders, "models/encoders.pkl")
joblib.dump(FEATURE_COLS, "models/feature_cols.pkl")

print("\n[SAVED] data/processed/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("[SAVED] models/encoders.pkl")
print("[SAVED] models/feature_cols.pkl")
print("\nFeature columns:")
for i, c in enumerate(FEATURE_COLS):
    print(f"  {i+1:2d}. {c}")