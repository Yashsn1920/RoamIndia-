"""
01_clean_data.py
----------------
Cleans and merges the two flight datasets into one unified CSV
ready for feature engineering.

DATASETS NEEDED (download from Kaggle and place in data/raw/):
  1. Dataset 1 — EaseMyTrip / Shubham Bathwal (300k rows, cleaner)
     URL: https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction
     File: Clean_Dataset.csv
     Columns: airline, flight, source_city, departure_time, stops,
               arrival_time, destination_city, class, duration, days_left, price

  2. Dataset 2 — Classic Kaggle flight fare dataset (10k rows, older)
     URL: https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh
     Files: Data_Train.xlsx, Test_set.xlsx
     Columns: Airline, Date_of_Journey, Source, Destination, Route,
               Dep_Time, Arrival_Time, Duration, Total_Stops, Additional_Info, Price

WHY BOTH?
  Dataset 1 is large and has days_left (advance booking) — crucial feature.
  Dataset 2 has Date_of_Journey — lets us extract month/season features.
  Together they give better generalisation.

OUTPUT: data/processed/flights_clean.csv
"""

import pandas as pd
import numpy as np
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── CITY NAME STANDARDISER ───────────────────────────────────────────────────
CITY_MAP = {
    "bombay": "Mumbai", "mumbai": "Mumbai",
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "banglore": "Bengaluru",
    "delhi": "Delhi", "new delhi": "Delhi",
    "madras": "Chennai", "chennai": "Chennai",
    "calcutta": "Kolkata", "kolkata": "Kolkata",
    "hyderabad": "Hyderabad",
    "cochin": "Kochi", "kochi": "Kochi",
    "goa": "Goa",
}

def standardise_city(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    return CITY_MAP.get(name.strip().lower(), name.strip().title())


# ─── DATASET 1: Clean_Dataset.csv (EaseMyTrip, 300k rows) ────────────────────
def load_dataset1():
    path = os.path.join(RAW_DIR, "Clean_Dataset.csv")
    df = pd.read_csv(path)
    print(f"[DS1] Loaded {len(df):,} rows")

    # Rename to unified schema
    df = df.rename(columns={
        "airline":          "airline",
        "source_city":      "source_city",
        "destination_city": "destination_city",
        "class":            "seat_class",
        "duration":         "duration_hrs",
        "days_left":        "days_advance",
        "price":            "price_inr",
        "stops":            "stops_raw",
    })

    # Encode stops: "zero" → 0, "one" → 1, "two or more" → 2
    stop_map = {"zero": 0, "one": 1, "two or more": 2}
    df["total_stops"] = df["stops_raw"].str.lower().str.strip().map(stop_map).fillna(1).astype(int)

    # Encode seat class
    df["is_business"] = (df["seat_class"].str.lower() == "business").astype(int)

    # Departure time bucket → numeric hour approximation
    time_map = {"early morning": 5, "morning": 8, "afternoon": 13,
                "evening": 17, "night": 21, "late night": 23}
    df["dep_hour"] = df["departure_time"].str.lower().str.strip().map(time_map).fillna(12).astype(int)

    # No journey date in DS1 — we'll set month/season to -1 (unknown)
    df["journey_month"] = -1
    df["journey_day"]   = -1

    df["source_city"]      = df["source_city"].apply(standardise_city)
    df["destination_city"] = df["destination_city"].apply(standardise_city)

    keep = ["airline", "source_city", "destination_city", "total_stops",
            "is_business", "dep_hour", "duration_hrs", "days_advance",
            "journey_month", "journey_day", "price_inr"]
    return df[keep].dropna(subset=["price_inr", "source_city", "destination_city"])


# ─── DATASET 2: Data_Train.xlsx (classic Kaggle, 10k rows) ───────────────────
def load_dataset2():
    path = os.path.join(RAW_DIR, "Data_Train.xlsx")
    df = pd.read_excel(path)
    print(f"[DS2] Loaded {len(df):,} rows")

    # Drop the 1 missing row in Total_Stops
    df = df.dropna(subset=["Total_Stops", "Price"])

    # Parse duration string e.g. "2h 30m" → float hours
    def parse_duration(s):
        try:
            s = str(s).strip()
            hours = int(s.split("h")[0]) if "h" in s else 0
            mins  = int(s.split("h")[-1].replace("m", "").strip()) if "m" in s else 0
            return round(hours + mins / 60, 2)
        except Exception:
            return np.nan

    df["duration_hrs"] = df["Duration"].apply(parse_duration)

    # Parse departure hour
    def parse_hour(t):
        try:
            return int(str(t).split(":")[0])
        except Exception:
            return 12

    df["dep_hour"] = df["Dep_Time"].apply(parse_hour)

    # Stops → int
    stop_map = {"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}
    df["total_stops"] = df["Total_Stops"].str.lower().str.strip().map(stop_map).fillna(1).astype(int)

    # Journey date features
    df["Date_of_Journey"] = pd.to_datetime(df["Date_of_Journey"], dayfirst=True, errors="coerce")
    df["journey_month"] = df["Date_of_Journey"].dt.month
    df["journey_day"]   = df["Date_of_Journey"].dt.day

    # Advance booking: dataset is from 2019 — approximate as unknown
    df["days_advance"] = -1
    df["is_business"]  = 0  # dataset is all economy

    df["source_city"]      = df["Source"].apply(standardise_city)
    df["destination_city"] = df["Destination"].apply(standardise_city)
    df["airline"]          = df["Airline"].str.strip()
    df["price_inr"]        = df["Price"].astype(float)

    keep = ["airline", "source_city", "destination_city", "total_stops",
            "is_business", "dep_hour", "duration_hrs", "days_advance",
            "journey_month", "journey_day", "price_inr"]
    return df[keep].dropna(subset=["price_inr", "duration_hrs"])


# ─── MERGE + CLEAN ────────────────────────────────────────────────────────────
def clean_and_merge():
    df1 = load_dataset1()
    df2 = load_dataset2()

    df = pd.concat([df1, df2], ignore_index=True)
    print(f"\n[MERGED] Total rows before cleaning: {len(df):,}")

    # Remove price outliers (clip at 1st–99th percentile)
    lo = df["price_inr"].quantile(0.01)
    hi = df["price_inr"].quantile(0.99)
    before = len(df)
    df = df[(df["price_inr"] >= lo) & (df["price_inr"] <= hi)]
    print(f"[CLEAN]  Removed {before - len(df):,} price outliers (kept ₹{lo:.0f}–₹{hi:.0f})")

    # Remove duration outliers
    df = df[df["duration_hrs"].between(0.5, 12)]

    # Drop rows where source == destination
    df = df[df["source_city"] != df["destination_city"]]

    # Standardise airline names
    airline_map = {
        "indigo": "IndiGo", "air india": "Air India", "vistara": "Vistara",
        "spicejet": "SpiceJet", "go air": "GoAir", "goair": "GoAir",
        "air asia": "AirAsia", "airasia": "AirAsia",
        "jet airways": "Jet Airways", "multiple carriers": "Other",
        "trujet": "Other", "star air": "Other",
    }
    df["airline"] = df["airline"].str.strip().str.lower().map(
        lambda x: airline_map.get(x, x.title() if isinstance(x, str) else "Other")
    )

    print(f"\n[FINAL]  Clean rows: {len(df):,}")
    print(f"         Price range: ₹{df['price_inr'].min():.0f} – ₹{df['price_inr'].max():.0f}")
    print(f"         Airlines: {sorted(df['airline'].unique())}")
    print(f"         Cities (source): {sorted(df['source_city'].unique())}")

    out_path = os.path.join(OUT_DIR, "flights_clean.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED]  {out_path}")
    return df


if __name__ == "__main__":
    df = clean_and_merge()
    print("\nSample rows:")
    print(df.sample(5).to_string(index=False))