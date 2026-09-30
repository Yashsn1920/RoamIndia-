"""
hotel_model/03_predict.py  →  rename to predict_03.py
-------------------------------------------------------
Inference function for hotel cost prediction.
Predicts price_per_night then multiplies by stay_nights.

Usage:
    from hotel_model.predict_03 import predict_hotel_cost
"""

import numpy as np
import pandas as pd
import joblib
from datetime import date
from pathlib import Path

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL = joblib.load(_MODEL_DIR / "hotel_model.pkl")
META  = joblib.load(_MODEL_DIR / "hotel_meta.pkl")

FEATURE_COLS      = META["feature_cols"]
CITY_ENC          = META["city_mean_enc"]
GLOBAL_MEAN_LOG   = META["global_mean_log_price"]
TIER_MAP          = META["tier_map"]

HOLIDAY_MONTHS = {10, 11, 12, 1}

# Season price multipliers on top of model output (captures booking-time effects
# not in the static hotel dataset, since dataset has no date column)
SEASON_MULT = {
    "peak":    1.30,
    "summer":  0.90,
    "monsoon": 0.72,
}
def get_season(month):
    if month in [10,11,12,1,2,3]: return "peak"
    if month in [4,5,6]:          return "summer"
    return "monsoon"

# City-level amenity defaults (for cities where we know the typical hotel profile)
CITY_AMENITY_DEFAULTS = {
    "Mumbai":    {"has_pool":1,"has_ac":1,"has_restaurant":1,"has_wifi":1,"has_gym":1},
    "Delhi":     {"has_pool":1,"has_ac":1,"has_restaurant":1,"has_wifi":1,"has_gym":0},
    "Goa":       {"has_pool":1,"has_ac":1,"has_restaurant":1,"has_wifi":1,"has_gym":0},
    "Bengaluru": {"has_pool":0,"has_ac":1,"has_restaurant":1,"has_wifi":1,"has_gym":1},
    "Jaipur":    {"has_pool":1,"has_ac":1,"has_restaurant":1,"has_wifi":1,"has_gym":0},
    "Udaipur":   {"has_pool":1,"has_ac":1,"has_restaurant":1,"has_wifi":1,"has_gym":0},
    "Manali":    {"has_pool":0,"has_ac":0,"has_restaurant":1,"has_wifi":1,"has_gym":0},
    "Shimla":    {"has_pool":0,"has_ac":0,"has_restaurant":1,"has_wifi":1,"has_gym":0},
    "Varanasi":  {"has_pool":0,"has_ac":1,"has_restaurant":1,"has_wifi":1,"has_gym":0},
    "default":   {"has_pool":0,"has_ac":1,"has_restaurant":1,"has_wifi":1,"has_gym":0},
}
TIER_STAR = {"budget":2.0,"midrange":3.0,"luxury":4.5}
TIER_PRICE_PCT = {"budget":0.25,"midrange":0.55,"luxury":0.85}


def predict_hotel_cost(
    destination_city: str,
    hotel_tier:       str,   # "budget" | "midrange" | "luxury"
    checkin_date:     date,
    checkout_date:    date,
    adults:           int  = 2,
    children:         int  = 0,
    num_rooms:        int  = 1,
) -> dict:
    stay_nights = (checkout_date - checkin_date).days
    if stay_nights <= 0:
        raise ValueError("checkout_date must be after checkin_date")

    city    = destination_city.strip().title()
    month   = checkin_date.month
    season  = get_season(month)

    tier_enc     = TIER_MAP.get(hotel_tier.lower(), 1)
    star_rating  = TIER_STAR[hotel_tier.lower()]
    city_enc     = CITY_ENC.get(city, GLOBAL_MEAN_LOG)
    city_pct     = TIER_PRICE_PCT[hotel_tier.lower()]
    amenities    = CITY_AMENITY_DEFAULTS.get(city, CITY_AMENITY_DEFAULTS["default"])
    amenity_score= sum(amenities.values())

    row = {
        "tier_enc":        tier_enc,
        "star_rating":     star_rating,
        "city_enc":        city_enc,
        "city_price_pct":  city_pct,
        "amenity_score":   amenity_score,
        "has_pool":        amenities["has_pool"],
        "has_ac":          amenities["has_ac"],
        "has_restaurant":  amenities["has_restaurant"],
        "has_wifi":        amenities["has_wifi"],
        "has_gym":         amenities["has_gym"],
        "source_enc":      0,   # treat as primary MMT data
    }

    X = pd.DataFrame([row])[FEATURE_COLS]
    log_pred        = MODEL.predict(X)[0]
    base_per_night  = np.expm1(log_pred)

    # Apply season multiplier (static dataset has no booking date)
    seasonal_per_night = base_per_night * SEASON_MULT[season]

    total_mid = round(seasonal_per_night * stay_nights * num_rooms)

    return {
        "total_low":      round(total_mid * 0.88),
        "total_mid":      total_mid,
        "total_high":     round(total_mid * 1.22),
        "per_night_mid":  round(seasonal_per_night),
        "stay_nights":    stay_nights,
        "inputs": {
            "destination": city,
            "tier":        hotel_tier,
            "checkin":     str(checkin_date),
            "checkout":    str(checkout_date),
            "adults":      adults,
            "children":    children,
            "num_rooms":   num_rooms,
            "season":      season,
            "month":       month,
        }
    }


if __name__ == "__main__":
    tests = [
        dict(destination_city="Goa",    hotel_tier="midrange",
             checkin_date=date(2025,12,20), checkout_date=date(2025,12,27)),
        dict(destination_city="Manali", hotel_tier="budget",
             checkin_date=date(2026,1,3),   checkout_date=date(2026,1,8),
             adults=2, children=1),
        dict(destination_city="Mumbai", hotel_tier="luxury",
             checkin_date=date(2025,11,14), checkout_date=date(2025,11,17)),
    ]
    print("=== HOTEL COST PREDICTIONS ===\n")
    for t in tests:
        r = predict_hotel_cost(**t)
        i = r["inputs"]
        print(f"{i['destination']} | {i['tier']} | {r['stay_nights']}n | {i['season']}")
        print(f"  Per night : ₹{r['per_night_mid']:,}")
        print(f"  Total     : ₹{r['total_low']:,} – ₹{r['total_high']:,}  (mid ₹{r['total_mid']:,})")
        print()