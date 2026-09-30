"""
hotel_model/01_clean_data.py
-----------------------------
Cleans and merges 3 India-specific hotel datasets into one unified CSV.

DATASETS TO DOWNLOAD:
  DS1 — MakeMyTrip Hotel Prices (primary — real ₹ prices, Indian cities)
        kaggle.com/datasets/andrewgeorgeissac/hotel-price-data-of-cities-in-india-makemytrip
        Save as: data/raw/mmt_hotels.csv

  DS2 — Google Indian Hotel Data 2023 (51 cities, ratings, price range)
        kaggle.com/datasets/alvinmanojalex/google-indian-hotel-data
        Save as: data/raw/google_hotels.csv

  DS3 — Hotels on MakeMyTrip by PromptCloud (20k hotels, broader coverage)
        kaggle.com/datasets/PromptCloudHQ/hotels-on-makemytrip
        Save as: data/raw/promptcloud_hotels.csv

WHY THESE THREE:
  DS1 — has actual ₹ per night per city → primary price signal
  DS2 — adds Google ratings, amenities, 51-city coverage → enriches features
  DS3 — adds 20k hotels for breadth, older price history → helps generalisation
  No EUR→INR conversion needed. All prices are real Indian ₹.

OUTPUT: data/processed/hotel_clean.csv
"""

import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

os.makedirs("data/processed", exist_ok=True)

CITY_MAP = {
    "new delhi": "Delhi", "delhi": "Delhi",
    "mumbai": "Mumbai", "bombay": "Mumbai",
    "bengaluru": "Bengaluru", "bangalore": "Bengaluru", "banglore": "Bengaluru",
    "chennai": "Chennai", "madras": "Chennai",
    "kolkata": "Kolkata", "calcutta": "Kolkata",
    "hyderabad": "Hyderabad",
    "kochi": "Kochi", "cochin": "Kochi",
    "goa": "Goa", "panaji": "Goa", "north goa": "Goa", "south goa": "Goa",
    "jaipur": "Jaipur", "agra": "Agra", "varanasi": "Varanasi",
    "udaipur": "Udaipur", "manali": "Manali", "shimla": "Shimla",
    "darjeeling": "Darjeeling", "mysuru": "Mysuru", "mysore": "Mysuru",
    "pune": "Pune", "ahmedabad": "Ahmedabad", "amritsar": "Amritsar",
    "rishikesh": "Rishikesh", "ooty": "Ooty", "munnar": "Munnar",
    "port blair": "Port Blair",
}

def std_city(name):
    if not isinstance(name, str):
        return "Unknown"
    return CITY_MAP.get(name.strip().lower(), name.strip().title())

def classify_tier(price):
    if pd.isna(price):   return "midrange"
    if price < 2000:     return "budget"
    elif price < 7000:   return "midrange"
    else:                return "luxury"

def parse_price(x):
    if pd.isna(x): return np.nan
    s = str(x).replace("₹","").replace("INR","").replace(",","").strip()
    try:    return float(s)
    except: return np.nan

def extract_amenities(df, amenity_col):
    """Returns 5 binary amenity columns from a text amenities field."""
    s = df[amenity_col].fillna("").str.lower()
    return pd.DataFrame({
        "has_pool":       s.str.contains("pool").astype(int),
        "has_ac":         s.str.contains("ac|air condition").astype(int),
        "has_restaurant": s.str.contains("restaurant|dining").astype(int),
        "has_wifi":       s.str.contains("wifi|wi-fi|internet").astype(int),
        "has_gym":        s.str.contains("gym|fitness").astype(int),
    })

def empty_amenities(n):
    return pd.DataFrame({c: np.zeros(n, dtype=int)
                         for c in ["has_pool","has_ac","has_restaurant","has_wifi","has_gym"]})

def find_col(cols_lower, candidates):
    for c in candidates:
        if c in cols_lower: return cols_lower[c]
    return None


# ─── DS1: MakeMyTrip Hotel Prices ────────────────────────────────────────────
def load_ds1():
    df = pd.read_csv("data/raw/mmt_hotels.csv", on_bad_lines='skip')
    print(f"[DS1] {len(df):,} rows | Columns: {list(df.columns)}")
    cl = {c.lower().strip(): c for c in df.columns}

    city_col   = find_col(cl, ["city","location","hotel_city"])
    price_col  = find_col(cl, ["price","price_per_night","rate","cost","tariff"])
    rating_col = find_col(cl, ["rating","star_rating","stars","hotel_rating","star rating"])
    name_col   = find_col(cl, ["hotel_name","name","hotel","hotel name"])
    amen_col   = find_col(cl, ["amenities","facilities","features"])

    out = pd.DataFrame()
    out["city"]            = df[city_col].apply(std_city) if city_col else "Unknown"
    out["price_per_night"] = df[price_col].apply(parse_price) if price_col else np.nan
    out["star_rating"]     = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan
    out["hotel_name"]      = df[name_col].fillna("Unknown") if name_col else "Unknown"
    amen = extract_amenities(df, amen_col) if amen_col else empty_amenities(len(df))
    out  = pd.concat([out.reset_index(drop=True), amen.reset_index(drop=True)], axis=1)
    out["source"] = "mmt_primary"
    return out.dropna(subset=["price_per_night","city"])


# ─── DS2: Google Indian Hotel Data 2023 ──────────────────────────────────────
def load_ds2():
    df = pd.read_csv("data/raw/google_hotels.csv", on_bad_lines='skip')
    print(f"[DS2] {len(df):,} rows | Columns: {list(df.columns)}")
    cl = {c.lower().strip(): c for c in df.columns}

    city_col   = find_col(cl, ["city","location","area"])
    price_col  = find_col(cl, ["hotel_price","price","price_per_night","cost","rate","min_price","avg_price"])
    rating_col = find_col(cl, ["hotel_rating","rating","star_rating","stars","google_rating"])
    name_col   = find_col(cl, ["hotel_name","name","property_name"])
    amen_col   = find_col(cl, ["amenities","facilities"])

    out = pd.DataFrame()
    out["city"]            = df[city_col].apply(std_city) if city_col else "Unknown"
    out["price_per_night"] = df[price_col].apply(parse_price) if price_col else np.nan
    out["star_rating"]     = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan
    out["hotel_name"]      = df[name_col].fillna("Unknown") if name_col else "Unknown"
    amen = extract_amenities(df, amen_col) if amen_col else empty_amenities(len(df))
    out  = pd.concat([out.reset_index(drop=True), amen.reset_index(drop=True)], axis=1)
    out["source"] = "google_2023"
    return out.dropna(subset=["price_per_night","city"])


# ─── DS3: PromptCloud MakeMyTrip Hotels ──────────────────────────────────────
def load_ds3():
    df = pd.read_csv("data/raw/promptcloud_hotels.csv", on_bad_lines='skip')
    print(f"[DS3] {len(df):,} rows | Columns: {list(df.columns)}")
    cl = {c.lower().strip(): c for c in df.columns}

    city_col   = find_col(cl, ["city","location","hotel_city","property_city"])
    price_col  = find_col(cl, ["price","price_per_night","tariff","room_tariff","rate"])
    rating_col = find_col(cl, ["rating","star_rating","stars","hotel_star"])
    name_col   = find_col(cl, ["hotel_name","name","property_name"])
    amen_col   = find_col(cl, ["amenities","facilities","hotel_amenities"])

    out = pd.DataFrame()
    out["city"]            = df[city_col].apply(std_city) if city_col else "Unknown"
    out["price_per_night"] = df[price_col].apply(parse_price) if price_col else np.nan
    out["star_rating"]     = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else np.nan
    out["hotel_name"]      = df[name_col].fillna("Unknown") if name_col else "Unknown"
    amen = extract_amenities(df, amen_col) if amen_col else empty_amenities(len(df))
    out  = pd.concat([out.reset_index(drop=True), amen.reset_index(drop=True)], axis=1)
    out["source"] = "promptcloud_mmt"
    return out.dropna(subset=["price_per_night","city"])


# ─── MERGE + CLEAN ────────────────────────────────────────────────────────────
def merge_and_clean():
    df1 = load_ds1()
    df2 = load_ds2()
    df3 = load_ds3()

    df = pd.concat([df1, df2, df3], ignore_index=True)
    print(f"\n[MERGED] {len(df):,} total rows")

    df = df[df["city"] != "Unknown"]
    df = df[df["price_per_night"] >= 200]
    hi = df["price_per_night"].quantile(0.99)
    df = df[df["price_per_night"] <= hi]
    print(f"[CLEAN]  After price filter: {len(df):,}  (₹200 – ₹{hi:,.0f})")

    df["hotel_tier"]   = df["price_per_night"].apply(classify_tier)
    df["tier_enc"]     = df["hotel_tier"].map({"budget":0,"midrange":1,"luxury":2})

    tier_star = {"budget":2.0,"midrange":3.0,"luxury":4.5}
    df["star_rating"]  = df.apply(
        lambda r: r["star_rating"] if not pd.isna(r["star_rating"])
                  else tier_star[r["hotel_tier"]], axis=1)

    # Within-city price percentile — how expensive vs other hotels in same city
    df["city_price_pct"] = df.groupby("city")["price_per_night"].rank(pct=True)

    amenity_cols = ["has_pool","has_ac","has_restaurant","has_wifi","has_gym"]
    df["amenity_score"] = df[amenity_cols].sum(axis=1)

    # Deduplicate same hotel name + city (keep highest price = most recent)
    before = len(df)
    df = df.sort_values("price_per_night", ascending=False)
    df = df.drop_duplicates(subset=["hotel_name","city"], keep="first")
    print(f"[DEDUP]  Removed {before-len(df):,} duplicate entries")

    print(f"\n[FINAL]  {len(df):,} hotels | {df['city'].nunique()} cities")
    print(f"         ₹{df['price_per_night'].min():.0f} – ₹{df['price_per_night'].max():.0f}  "
          f"(median ₹{df['price_per_night'].median():.0f}/night)")
    print(f"\n  Tier distribution:\n{df['hotel_tier'].value_counts().to_string()}")
    print(f"\n  Top cities:\n{df['city'].value_counts().head(12).to_string()}")

    df.to_csv("data/processed/hotel_clean.csv", index=False)
    print(f"\n[SAVED]  data/processed/hotel_clean.csv")
    return df

if __name__ == "__main__":
    df = merge_and_clean()
    print("\nSample rows:")
    if len(df) > 0:
        print(df[["city","hotel_tier","price_per_night","star_rating",
                  "amenity_score","source"]].sample(min(8, len(df))).to_string(index=False))
    else:
        print("No valid rows remaining.")