"""
02_eda.py
---------
Exploratory Data Analysis on the cleaned flight dataset.
Run this AFTER 01_clean_data.py.

Outputs charts to: data/eda_charts/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

DATA_PATH = "data/processed/flights_clean.csv"
OUT_DIR   = "data/eda_charts"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows\n")


# ─── HELPER ───────────────────────────────────────────────────────────────────
def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─── 1. PRICE DISTRIBUTION ────────────────────────────────────────────────────
print("1. Price distribution")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df["price_inr"], bins=60, color="#7F77DD", edgecolor="white", linewidth=0.4)
axes[0].set_title("Price distribution (raw)")
axes[0].set_xlabel("Price (₹)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))

axes[1].hist(np.log1p(df["price_inr"]), bins=60, color="#1D9E75", edgecolor="white", linewidth=0.4)
axes[1].set_title("Price distribution (log scale)")
axes[1].set_xlabel("log(Price + 1)")
savefig("01_price_distribution.png")


# ─── 2. PRICE BY AIRLINE ──────────────────────────────────────────────────────
print("2. Price by airline")
airline_stats = (df.groupby("airline")["price_inr"]
                   .median()
                   .sort_values(ascending=False)
                   .head(10))
fig, ax = plt.subplots(figsize=(10, 4))
airline_stats.plot(kind="bar", ax=ax, color="#534AB7", edgecolor="white")
ax.set_title("Median price by airline")
ax.set_ylabel("Median price (₹)")
ax.set_xlabel("")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
ax.tick_params(axis="x", rotation=30)
savefig("02_price_by_airline.png")


# ─── 3. PRICE BY SOURCE → DESTINATION ────────────────────────────────────────
print("3. Price by route")
route_stats = (df.assign(route=df["source_city"] + " → " + df["destination_city"])
                 .groupby("route")["price_inr"]
                 .median()
                 .sort_values(ascending=False)
                 .head(15))
fig, ax = plt.subplots(figsize=(10, 5))
route_stats.plot(kind="barh", ax=ax, color="#D85A30", edgecolor="white")
ax.set_title("Median price — top 15 routes")
ax.set_xlabel("Median price (₹)")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
savefig("03_price_by_route.png")


# ─── 4. PRICE VS DAYS ADVANCE ─────────────────────────────────────────────────
print("4. Price vs advance booking")
adv = df[df["days_advance"] > 0]  # DS2 has -1 placeholder
if len(adv) > 0:
    bins = pd.cut(adv["days_advance"], bins=[0,7,14,30,60,90,200], right=False,
                  labels=["<7d","7–14d","14–30d","30–60d","60–90d","90d+"])
    adv_stats = adv.groupby(bins, observed=True)["price_inr"].median()
    fig, ax = plt.subplots(figsize=(8, 4))
    adv_stats.plot(kind="bar", ax=ax, color="#1D9E75", edgecolor="white")
    ax.set_title("Median price by advance booking window")
    ax.set_ylabel("Median price (₹)")
    ax.set_xlabel("Days before departure")
    ax.tick_params(axis="x", rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    savefig("04_price_vs_advance.png")
else:
    print("  Skipped — no days_advance data")


# ─── 5. PRICE VS STOPS ────────────────────────────────────────────────────────
print("5. Price vs stops")
stop_stats = df.groupby("total_stops")["price_inr"].median()
fig, ax = plt.subplots(figsize=(6, 4))
stop_stats.plot(kind="bar", ax=ax, color="#BA7517", edgecolor="white")
ax.set_title("Median price by number of stops")
ax.set_ylabel("Median price (₹)")
ax.set_xlabel("Stops")
ax.set_xticklabels(["Non-stop", "1 stop", "2 stops", "3+ stops"][:len(stop_stats)], rotation=0)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
savefig("05_price_vs_stops.png")


# ─── 6. PRICE VS DURATION ─────────────────────────────────────────────────────
print("6. Price vs duration")
sample = df.sample(min(5000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(sample["duration_hrs"], sample["price_inr"],
           alpha=0.25, s=10, color="#7F77DD")
ax.set_title("Price vs flight duration")
ax.set_xlabel("Duration (hours)")
ax.set_ylabel("Price (₹)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
savefig("06_price_vs_duration.png")


# ─── 7. PRICE BY MONTH (DS2 only) ────────────────────────────────────────────
print("7. Price by month")
monthly = df[df["journey_month"] > 0]
if len(monthly) > 0:
    month_stats = monthly.groupby("journey_month")["price_inr"].median()
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(10, 4))
    month_stats.plot(kind="bar", ax=ax, color="#378ADD", edgecolor="white")
    ax.set_title("Median price by month of travel")
    ax.set_ylabel("Median price (₹)")
    ax.set_xlabel("")
    ax.set_xticklabels([months[i-1] for i in month_stats.index], rotation=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1000:.0f}k"))
    savefig("07_price_by_month.png")


# ─── 8. CORRELATION HEATMAP ───────────────────────────────────────────────────
print("8. Correlation heatmap")
num_cols = ["price_inr", "duration_hrs", "total_stops", "dep_hour",
            "days_advance", "is_business", "journey_month"]
corr = df[num_cols].replace(-1, np.nan).corr()

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(num_cols)))
ax.set_yticks(range(len(num_cols)))
ax.set_xticklabels(num_cols, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(num_cols, fontsize=9)
ax.set_title("Feature correlation matrix")
for i in range(len(num_cols)):
    for j in range(len(num_cols)):
        val = corr.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
savefig("08_correlation_heatmap.png")


# ─── SUMMARY STATS ────────────────────────────────────────────────────────────
print("\n=== SUMMARY STATS ===")
print(df["price_inr"].describe().apply(lambda x: f"₹{x:,.0f}"))
print(f"\nTop airlines:\n{df['airline'].value_counts().head(8)}")
print(f"\nTop source cities:\n{df['source_city'].value_counts().head(8)}")
print(f"\nTop destinations:\n{df['destination_city'].value_counts().head(8)}")
print(f"\nMissing values:\n{df.isnull().sum()}")