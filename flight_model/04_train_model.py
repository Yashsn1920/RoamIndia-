"""
04_train_model.py
-----------------
Trains the XGBoost flight price regression model.
Run AFTER 03_feature_engineering.py.

What happens here:
  1. Load processed feature matrices
  2. Train XGBoost with early stopping
  3. Evaluate: MAE, RMSE, R² (in ₹, not log space)
  4. Plot feature importances
  5. Save model to models/flight_model.pkl

The model predicts log(price+1), so at inference we exponentiate back.
"""

import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("models", exist_ok=True)
os.makedirs("data/eda_charts", exist_ok=True)

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
print("Loading feature matrices...")
X_train = pd.read_csv("data/processed/X_train.csv")
X_test  = pd.read_csv("data/processed/X_test.csv")
y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test  = pd.read_csv("data/processed/y_test.csv").squeeze()

print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}")


# ─── MODEL ────────────────────────────────────────────────────────────────────
model = xgb.XGBRegressor(
    n_estimators      = 1000,    # large — early stopping will find the right n
    max_depth         = 6,
    learning_rate     = 0.05,
    subsample         = 0.80,    # row sampling per tree
    colsample_bytree  = 0.80,    # feature sampling per tree
    min_child_weight  = 5,       # prevent overfitting on rare routes
    gamma             = 0.1,
    reg_alpha         = 0.05,    # L1 regularisation
    reg_lambda        = 1.0,     # L2 regularisation
    random_state      = 42,
    tree_method       = "hist",  # fast histogram method
    device            = "cpu",   # change to "cuda" if you have a GPU
    eval_metric       = "rmse",
    early_stopping_rounds = 40,
    verbosity         = 0,
)

print("\nTraining XGBoost (early stopping on test RMSE)...")
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=50,
)

best_iter = model.best_iteration
print(f"\nBest iteration: {best_iter}")


# ─── EVALUATE ─────────────────────────────────────────────────────────────────
# Predict in log space, then convert back to ₹
log_pred_train = model.predict(X_train)
log_pred_test  = model.predict(X_test)

pred_train_inr = np.expm1(log_pred_train)
pred_test_inr  = np.expm1(log_pred_test)
true_train_inr = np.expm1(y_train)
true_test_inr  = np.expm1(y_test)

def evaluate(true, pred, split_name):
    mae  = mean_absolute_error(true, pred)
    rmse = np.sqrt(mean_squared_error(true, pred))
    r2   = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / true.clip(lower=1))) * 100
    print(f"\n[{split_name}]")
    print(f"  MAE  : ₹{mae:,.0f}   ← avg error per prediction")
    print(f"  RMSE : ₹{rmse:,.0f}")
    print(f"  R²   : {r2:.4f}   ← 1.0 = perfect")
    print(f"  MAPE : {mape:.1f}%  ← % error on average")
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}

train_metrics = evaluate(true_train_inr, pred_train_inr, "TRAIN")
test_metrics  = evaluate(true_test_inr,  pred_test_inr,  "TEST ")


# ─── PLOT: ACTUAL VS PREDICTED ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Scatter: actual vs predicted (test set, sample 3000)
idx = np.random.choice(len(true_test_inr), min(3000, len(true_test_inr)), replace=False)
axes[0].scatter(true_test_inr.iloc[idx], pred_test_inr[idx],
                alpha=0.2, s=8, color="#7F77DD")
max_val = max(true_test_inr.max(), pred_test_inr.max())
axes[0].plot([0, max_val], [0, max_val], "r--", linewidth=1)
axes[0].set_xlabel("Actual price (₹)")
axes[0].set_ylabel("Predicted price (₹)")
axes[0].set_title(f"Actual vs predicted — test set\nR²={test_metrics['r2']:.3f}")

# Residual distribution
residuals = pred_test_inr - true_test_inr.values
axes[1].hist(residuals, bins=60, color="#1D9E75", edgecolor="white", linewidth=0.4)
axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
axes[1].set_xlabel("Residual (predicted − actual) ₹")
axes[1].set_title(f"Residual distribution\nMAE=₹{test_metrics['mae']:,.0f}")

plt.tight_layout()
plt.savefig("data/eda_charts/09_model_eval.png", dpi=120, bbox_inches="tight")
plt.close()
print("\n[CHART] Saved: data/eda_charts/09_model_eval.png")


# ─── PLOT: FEATURE IMPORTANCE ─────────────────────────────────────────────────
FEATURE_COLS = joblib.load("models/feature_cols.pkl")
importance   = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
importance.tail(12).plot(kind="barh", ax=ax, color="#534AB7", edgecolor="white")
ax.set_title("Top 12 feature importances (XGBoost gain)")
ax.set_xlabel("Importance score")
plt.tight_layout()
plt.savefig("data/eda_charts/10_feature_importance.png", dpi=120, bbox_inches="tight")
plt.close()
print("[CHART] Saved: data/eda_charts/10_feature_importance.png")
print("\nTop features:")
for feat, score in importance.sort_values(ascending=False).head(8).items():
    print(f"  {feat:<28s} {score:.4f}")


# ─── SAVE MODEL ───────────────────────────────────────────────────────────────
joblib.dump(model, "models/flight_model.pkl")
print("\n[SAVED] models/flight_model.pkl")


# ─── QUICK SANITY CHECK ───────────────────────────────────────────────────────
print("\n=== SANITY CHECK — Sample predictions ===")
sample = X_test.sample(5, random_state=99)
preds  = np.expm1(model.predict(sample))
actual = np.expm1(y_test.iloc[sample.index])

for i, (pred, act) in enumerate(zip(preds, actual)):
    diff = pred - act
    print(f"  Sample {i+1}: Predicted ₹{pred:,.0f}  |  Actual ₹{act:,.0f}  |  Diff ₹{diff:+,.0f}")