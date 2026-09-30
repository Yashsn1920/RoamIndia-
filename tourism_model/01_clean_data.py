import pandas as pd
import numpy as np
import os
import ast
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
    "jaipur": "Jaipur",
    "agra": "Agra",
    "varanasi": "Varanasi", "banaras": "Varanasi", "varanasi (kashi)": "Varanasi",
    "goa": "Goa", "panaji": "Goa",
    "manali": "Manali",
    "shimla": "Shimla",
    "darjeeling": "Darjeeling",
    "udaipur": "Udaipur",
    "mysuru": "Mysuru", "mysore": "Mysuru",
    "pune": "Pune",
    "ahmedabad": "Ahmedabad",
    "amritsar": "Amritsar",
    "rishikesh": "Rishikesh",
    "haridwar": "Haridwar",
    "ooty": "Ooty", "udhagamandalam": "Ooty",
    "munnar": "Munnar",
    "andaman": "Port Blair", "port blair": "Port Blair",
    "ladakh (leh)": "Ladakh"
}

def std_city(name):
    if not isinstance(name, str):
        return "Unknown"
    # special handling for names like "Ladakh (Leh)"
    n = name.strip().lower()
    return CITY_MAP.get(n, name.strip().title())

TYPE_MAP = {
    "beach": "beach", "coastal": "beach", "island": "beach",
    "hill station": "hill", "hill": "hill", "mountain": "hill", "snow": "hill",
    "forest": "nature", "wildlife": "nature", "national park": "nature", "nature": "nature",
    "heritage": "heritage", "historical": "heritage", "fort": "heritage", "history": "heritage",
    "religious": "religious", "temple": "religious", "pilgrimage": "religious", "spiritual": "religious",
    "adventure": "adventure", "trekking": "adventure", "safari": "adventure",
    "city": "city", "urban": "city", "culture": "city"
}

def classify_type(t):
    if not isinstance(t, str):
        return "other"
    t_lower = t.strip().lower()
    for key, val in TYPE_MAP.items():
        if key in t_lower:
            return val
    return "other"

def parse_activities_cost(row):
    costs = []
    for cat in ["budget_category", "mid_range_category", "luxury_category"]:
        val = row.get(cat)
        if pd.notna(val) and isinstance(val, str):
            try:
                # ast.literal_eval parses the string representation of dict
                d = ast.literal_eval(val)
                if "activities_range" in d:
                    costs.extend(d["activities_range"])
            except:
                pass
    if costs:
        return np.mean(costs)
    return 500.0 # fallback

def main():
    df = pd.read_csv("data/raw/indian_tourism.csv")
    print(f"Loaded Indian Tourism Dataset: {len(df):,} rows")

    out = pd.DataFrame()

    # Extract city name (fallback to destination_name)
    raw_city = df["nearest_major_city"].fillna(df["destination_name"])
    # If the destination name is already a major city, use it
    out["city"] = df["destination_name"].apply(std_city)

    # Calculate daily activity cost from the JSON strings
    out["daily_activity_cost_inr"] = df.apply(parse_activities_cost, axis=1)

    # Classify destination type based on trip_types or primary_attractions
    out["top_dest_type"] = df["trip_types"].fillna(df["primary_attractions"]).apply(classify_type)

    # Num spots proxy from primary_attractions length
    out["num_spots"] = df["primary_attractions"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 3)

    # Popularity score (0-1)
    pop = pd.to_numeric(df["popularity_score"], errors="coerce").fillna(5.0)
    out["popularity_score"] = pop / 10.0

    # If there are duplicate cities, group them and take mean
    city_stats = out.groupby("city").agg({
        "daily_activity_cost_inr": "mean",
        "top_dest_type": lambda x: x.mode()[0] if len(x)>0 else "other",
        "num_spots": "sum",
        "popularity_score": "mean"
    }).reset_index()

    # Re-normalize popularity_score based on total num_spots across rows
    max_spots = city_stats["num_spots"].max()
    city_stats["popularity_score"] = (city_stats["num_spots"] / max_spots).clip(0.1, 1.0)

    # Save the lookup table
    city_stats.to_csv("data/processed/city_activity_cost.csv", index=False)

    print(f"\n[SAVED]  data/processed/city_activity_cost.csv ({len(city_stats)} cities)")
    print("\nTop 10 cities by daily activity cost:")
    print(city_stats.nlargest(10, "daily_activity_cost_inr")[
        ["city", "top_dest_type", "daily_activity_cost_inr", "num_spots"]
    ].to_string(index=False))

if __name__ == "__main__":
    main()