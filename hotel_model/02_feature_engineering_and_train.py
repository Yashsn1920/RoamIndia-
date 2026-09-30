"""
hotel_model/02_feature_engineering_and_train.py
------------------------------------------------
Feature engineering + XGBoost training for the hotel cost model.
Run AFTER 01_clean_data.py.

TARGET: price_per_night (₹) — log-transformed for training
        At inference we multiply by stay_nights × num_rooms to get total cost.

OUTPUTS:
  data/processed/hotel_X_train.csv
  data/processed/hotel_X_test.csv
  data/processed/hotel_y_train.csv
  data/processed/hotel_y_test.csv
  models/hotel_model.pkl
  models/hotel_meta.pkl
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/eda_charts", exist_ok=True)

# ─── LOAD ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/hotel_clean.csv")
print(f"Loaded {len(df):,} hotels\n")

# ─── TARGET ENCODING FOR CITY ─────────────────────────────────────────────────
# Cities are high-cardinality — use mean log-price encoding
df["log_price"] = np.log1p(df["price_per_night"])
city_mean       = df.groupby("city")["log_price"].mean()
df["city_enc"]  = df["city"].map(city_mean).fillna(df["log_price"].mean())

# Source encoding (which dataset the row came from — mild data source effect)
source_map = {"mmt_primary": 0, "google_2023": 1, "promptcloud_mmt": 2}
df["source_enc"] = df["source"].map(source_map).fillna(0)

# ─── FEATURE LIST ─────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "tier_enc",           # 0=budget  1=midrange  2=luxury  ← strongest signal
    "star_rating",        # 1–5 stars
    "city_enc",           # target-encoded city mean log-price
    "city_price_pct",     # within-city price percentile (0→1)
    "amenity_score",      # 0–5 count of amenities
    "has_pool",
    "has_ac",
    "has_restaurant",
    "has_wifi",
    "has_gym",
    "source_enc",         # dataset origin
]

X = df[FEATURE_COLS].fillna(0)
y = df["log_price"]

# ─── TRAIN / TEST SPLIT ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42,
    stratify=df["hotel_tier"]   # stratify so each tier is equally represented
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─── TRAIN XGBoost ────────────────────────────────────────────────────────────
model = xgb.XGBRegressor(
    n_estimators         = 700,
    max_depth            = 5,
    learning_rate        = 0.05,
    subsample            = 0.80,
    colsample_bytree     = 0.80,
    min_child_weight     = 8,
    gamma                = 0.05,
    reg_alpha            = 0.1,
    reg_lambda           = 1.0,
    random_state         = 42,
    tree_method          = "hist",
    eval_metric          = "rmse",
    early_stopping_rounds= 30,
    verbosity            = 0,
)
print("\nTraining hotel price model...")
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          verbose=50)
print(f"Best iteration: {model.best_iteration}")

# ─── EVALUATE ─────────────────────────────────────────────────────────────────
def evaluate(X_s, y_s, label):
    pred_log = model.predict(X_s)
    pred_inr = np.expm1(pred_log)
    true_inr = np.expm1(y_s)
    mae  = mean_absolute_error(true_inr, pred_inr)
    rmse = np.sqrt(mean_squared_error(true_inr, pred_inr))
    r2   = r2_score(true_inr, pred_inr)
    mape = np.mean(np.abs((true_inr - pred_inr) / true_inr.clip(lower=1))) * 100
    print(f"[{label}]  MAE=₹{mae:,.0f}/night  RMSE=₹{rmse:,.0f}  R²={r2:.4f}  MAPE={mape:.1f}%")
    return pred_inr, true_inr

print()
pred_train, true_train = evaluate(X_train, y_train, "TRAIN")
pred_test,  true_test  = evaluate(X_test,  y_test,  "TEST ")

# ─── PLOTS ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sample_idx = np.random.choice(len(true_test), min(3000, len(true_test)), replace=False)
axes[0].scatter(np.array(true_test)[sample_idx], np.array(pred_test)[sample_idx],
                alpha=0.2, s=8, color="#1D9E75")
mv = max(true_test.max(), pred_test.max())
axes[0].plot([0, mv], [0, mv], "r--", lw=1)
axes[0].set_title(f"Actual vs Predicted — Hotel (R²={r2_score(true_test, pred_test):.3f})")
axes[0].set_xlabel("Actual ₹/night")
axes[0].set_ylabel("Predicted ₹/night")

imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
imp.plot(kind="barh", ax=axes[1], color="#1D9E75", edgecolor="white")
axes[1].set_title("Feature importances — Hotel model")

plt.tight_layout()
plt.savefig("data/eda_charts/hotel_eval.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n[CHART]  data/eda_charts/hotel_eval.png")

# ─── SAVE ─────────────────────────────────────────────────────────────────────
X_train.to_csv("data/processed/hotel_X_train.csv", index=False)
X_test.to_csv("data/processed/hotel_X_test.csv",   index=False)
y_train.to_csv("data/processed/hotel_y_train.csv", index=False)
y_test.to_csv("data/processed/hotel_y_test.csv",   index=False)

meta = {
    "feature_cols": FEATURE_COLS,
    "city_mean_enc": city_mean.to_dict(),
    "global_mean_log_price": float(df["log_price"].mean()),
    "tier_map": {"budget":0,"midrange":1,"luxury":2},
    "source_map": source_map,
}
joblib.dump(model, "models/hotel_model.pkl")
joblib.dump(meta,  "models/hotel_meta.pkl")
print("[SAVED]  models/hotel_model.pkl")
print("[SAVED]  models/hotel_meta.pkl")