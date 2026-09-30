"""
tourism_model/02_train_model.py
--------------------------------
Trains a Random Forest regression model on top of the city-level
activity cost lookup table.

HOW THIS MODEL WORKS (two-layer approach):
  Layer 1 — City lookup table (from 01_clean_data.py):
    Gives base daily_activity_cost_inr per city.

  Layer 2 — ML multiplier model (this file):
    Adjusts the base cost based on:
      - traveller type (solo vs family vs group)
      - trip purpose (leisure vs adventure vs pilgrimage)
      - season (peak vs off-peak)
      - destination type (beach, hill, heritage, etc.)
      - number of days (economies of scale for longer trips)
    Output: multiplier (e.g. 1.4 = 40% above base)

  Final cost = base_daily_cost × multiplier × num_travellers × days

Since we don't have direct "actual spend per trip" data (no dataset has
that), we build synthetic training data using well-established rules
from India tourism spending research and scale it to the lookup table.
This is a reasonable approach when ground-truth spend data is unavailable.

FIX (v2): Replaced multiplicative compounding of all factors with a
weighted additive blend. Previously, stacking adventure × hill × peak
could reach 2.9–3.5×, and pilgrimage × religious × family could crash
to 0.3×. Now each factor contributes an additive delta to a base of
1.0, capped by a final realistic range.

OUTPUT:
  models/tourism_model.pkl
  models/tourism_meta.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/eda_charts", exist_ok=True)

# ─── LOAD CITY LOOKUP ─────────────────────────────────────────────────────────
city_lookup = pd.read_csv("data/processed/city_activity_cost.csv")
print(f"City lookup loaded: {len(city_lookup)} cities")

# ─── SYNTHETIC TRAINING DATA ──────────────────────────────────────────────────
# We generate realistic (city, traveller_type, season, purpose, days) → spend
# combinations using known Indian tourism spending patterns.
# References:
#   - India Tourism Statistics 2023 (Ministry of Tourism)
#   - Average domestic tourist spend: ₹4,500–8,000/person/trip
#   - Adventure tourism premium: ~1.5–2× leisure (not 3×)
#   - Peak season premium: ~20–35% above off-peak

np.random.seed(42)
N_SAMPLES = 20000

TRAVELLER_TYPES  = ["solo", "couple", "family", "group"]
SEASONS          = ["peak", "summer", "monsoon"]
TRIP_PURPOSES    = ["leisure", "adventure", "pilgrimage", "business"]
DEST_TYPES       = ["beach", "hill", "heritage", "religious", "nature", "adventure", "city", "other"]

# ─── ADDITIVE DELTA APPROACH ──────────────────────────────────────────────────
# Each factor contributes an additive delta to a base multiplier of 1.0.
# Final multiplier = 1.0 + Δtraveller + Δseason + Δpurpose + Δdest_type + Δduration
# Then clipped to [0.55, 2.20] — a realistic real-world range for India.
#
# This prevents two "low" factors (e.g. pilgrimage + religious) from
# compounding into an impossibly small number, and two "high" factors
# (e.g. adventure + adventure dest + peak) from blowing past 3×.

TRAVELLER_DELTA = {
    "solo":   +0.15,   # solo pays more per head (no room-sharing)
    "couple":  0.00,   # baseline
    "family": -0.10,   # families economise slightly
    "group":  -0.18,   # groups share costs / get discounts
}
SEASON_DELTA = {
    "peak":    +0.25,
    "summer":  -0.05,
    "monsoon": -0.20,
}
PURPOSE_DELTA = {
    "leisure":     0.00,
    "adventure":  +0.55,   # guide fees, permits, gear rental
    "pilgrimage": -0.25,   # frugal travel; was dragging to 0.6× × 0.7× = 0.42
    "business":   +0.10,
}
DEST_TYPE_DELTA = {
    "beach":     +0.20,
    "hill":      +0.15,
    "heritage":  +0.10,
    "religious": -0.10,   # budget-conscious destinations; was 0.7× compounding
    "nature":    +0.25,
    "adventure": +0.40,   # capped by final clip, won't double-dip to 3×
    "city":       0.00,
    "other":     -0.05,
}

REALISTIC_MIN = 0.55   # ~pilgrimage family monsoon — still above destitute
REALISTIC_MAX = 2.20   # ~adventure peak solo — premium but not luxury international


def duration_delta(days):
    """Longer trips lower per-day spend (fixed costs spread out)."""
    if days <= 2:   return +0.10
    if days <= 4:   return  0.00
    if days <= 7:   return -0.08
    return -0.15


def generate_sample(city_row):
    ttype     = np.random.choice(TRAVELLER_TYPES, p=[0.15, 0.40, 0.30, 0.15])
    season    = np.random.choice(SEASONS, p=[0.45, 0.30, 0.25])
    purpose   = np.random.choice(TRIP_PURPOSES, p=[0.55, 0.15, 0.20, 0.10])
    days      = np.random.choice([1,2,3,4,5,6,7,8,10,14],
                                  p=[0.05,0.10,0.15,0.20,0.20,0.10,0.10,0.05,0.03,0.02])
    dest_type = city_row["top_dest_type"]
    base      = city_row["daily_activity_cost_inr"]

    mult = (
        1.0
        + TRAVELLER_DELTA[ttype]
        + SEASON_DELTA[season]
        + PURPOSE_DELTA[purpose]
        + DEST_TYPE_DELTA.get(dest_type, 0.0)
        + duration_delta(days)
        + np.random.normal(0, 0.07)   # ±7% noise
    )
    mult = float(np.clip(mult, REALISTIC_MIN, REALISTIC_MAX))
    daily_spend = base * mult

    return {
        "dest_type_enc":      DEST_TYPES.index(dest_type) if dest_type in DEST_TYPES else 7,
        "traveller_type_enc": TRAVELLER_TYPES.index(ttype),
        "season_enc":         SEASONS.index(season),
        "purpose_enc":        TRIP_PURPOSES.index(purpose),
        "trip_days":          days,
        "num_spots":          float(city_row["num_spots"]),
        "popularity_score":   float(city_row["popularity_score"]),
        "base_daily_cost":    base,
        "multiplier":         mult,         # TARGET
        "daily_spend_inr":    daily_spend,  # alternative TARGET
    }


# Sample cities weighted by popularity
city_weights = city_lookup["popularity_score"].values
city_weights = city_weights / city_weights.sum()

rows = []
for _ in range(N_SAMPLES):
    city_idx = np.random.choice(len(city_lookup), p=city_weights)
    rows.append(generate_sample(city_lookup.iloc[city_idx]))

df = pd.DataFrame(rows)
print(f"Synthetic training data: {len(df):,} rows")
print(f"Multiplier range: {df['multiplier'].min():.2f} – {df['multiplier'].max():.2f}")
print(f"Multiplier mean:  {df['multiplier'].mean():.2f}  std: {df['multiplier'].std():.2f}")

# ─── EXPECTED SANITY CHECK ────────────────────────────────────────────────────
# Print expected multipliers for the four problem test cases so we can
# verify the training distribution looks right before fitting.
def expected_mult(ttype, season, purpose, dest, days):
    m = (1.0
         + TRAVELLER_DELTA[ttype]
         + SEASON_DELTA[season]
         + PURPOSE_DELTA[purpose]
         + DEST_TYPE_DELTA.get(dest, 0.0)
         + duration_delta(days))
    return float(np.clip(m, REALISTIC_MIN, REALISTIC_MAX))

print("\n[SANITY] Expected base multipliers for test cases:")
cases = [
    ("couple",  "peak",    "leisure",    "beach",    7,  "Goa"),
    ("couple",  "peak",    "adventure",  "hill",     5,  "Manali"),
    ("family",  "peak",    "pilgrimage", "religious",3,  "Varanasi"),
    ("solo",    "peak",    "leisure",    "heritage", 4,  "Jaipur"),
]
for ttype, season, purpose, dest, days, city in cases:
    m = expected_mult(ttype, season, purpose, dest, days)
    print(f"  {city:10s} → {m:.2f}×")

# ─── FEATURES + TARGET ────────────────────────────────────────────────────────
FEATURE_COLS = [
    "dest_type_enc",
    "traveller_type_enc",
    "season_enc",
    "purpose_enc",
    "trip_days",
    "num_spots",
    "popularity_score",
]
TARGET = "multiplier"

X = df[FEATURE_COLS]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─── TRAIN ────────────────────────────────────────────────────────────────────
model = GradientBoostingRegressor(
    n_estimators  = 400,
    max_depth     = 4,
    learning_rate = 0.05,
    subsample     = 0.8,
    random_state  = 42,
)
model.fit(X_train, y_train)

pred_train = model.predict(X_train)
pred_test  = model.predict(X_test)

mae_train = mean_absolute_error(y_train, pred_train)
mae_test  = mean_absolute_error(y_test,  pred_test)
r2_train  = r2_score(y_train, pred_train)
r2_test   = r2_score(y_test,  pred_test)

print(f"\n[TRAIN]  MAE={mae_train:.4f}  R²={r2_train:.4f}")
print(f"[TEST]   MAE={mae_test:.4f}  R²={r2_test:.4f}")

cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"[CV]     R² mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

# ─── FEATURE IMPORTANCE PLOT ──────────────────────────────────────────────────
imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
fig, ax = plt.subplots(figsize=(7, 4))
imp.plot(kind="barh", ax=ax, color="#D85A30", edgecolor="white")
ax.set_title("Feature importances — Tourism multiplier model")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("data/eda_charts/tourism_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("[CHART]  data/eda_charts/tourism_importance.png")

# ─── SAVE ─────────────────────────────────────────────────────────────────────
meta = {
    "feature_cols":    FEATURE_COLS,
    "dest_types":      DEST_TYPES,
    "traveller_types": TRAVELLER_TYPES,
    "seasons":         SEASONS,
    "purposes":        TRIP_PURPOSES,
    "city_lookup":     city_lookup.to_dict(orient="records"),
    "realistic_min":   REALISTIC_MIN,
    "realistic_max":   REALISTIC_MAX,
    # Fallback daily costs if city not found in lookup
    "fallback_daily": {
        "budget":   600,
        "midrange": 1200,
        "luxury":   3000,
    }
}

joblib.dump(model, "models/tourism_model.pkl")
joblib.dump(meta,  "models/tourism_meta.pkl")
print("\n[SAVED]  models/tourism_model.pkl")
print("[SAVED]  models/tourism_meta.pkl")