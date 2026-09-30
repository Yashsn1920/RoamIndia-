"""
tourism_model/03_predict.py  →  rename to predict_03.py
---------------------------------------------------------
Inference function for tourism/activities cost prediction.

Usage:
    from tourism_model.predict_03 import predict_tourism_cost

    result = predict_tourism_cost(
        destination    = "Goa",
        trip_days      = 7,
        num_travellers = 2,
        hotel_tier     = "midrange",
        traveller_type = "couple",
        trip_purpose   = "leisure",
        travel_month   = 12,
    )
    print(result)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ─── LOAD ARTIFACTS ───────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL = joblib.load(_MODEL_DIR / "tourism_model.pkl")
META  = joblib.load(_MODEL_DIR / "tourism_meta.pkl")

DEST_TYPES      = META["dest_types"]
TRAVELLER_TYPES = META["traveller_types"]
SEASONS         = META["seasons"]
PURPOSES        = META["purposes"]
FEATURE_COLS    = META["feature_cols"]
FALLBACK_DAILY  = META["fallback_daily"]

# Clip bounds now stored in meta (set during training)
# Falls back to safe defaults if loading an older model pkl
REALISTIC_MIN = META.get("realistic_min", 0.55)
REALISTIC_MAX = META.get("realistic_max", 2.20)

# Rebuild city lookup as dict for O(1) access
CITY_LOOKUP = {
    row["city"]: row
    for row in META["city_lookup"]
}

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def get_season(month: int) -> str:
    if month in [10, 11, 12, 1, 2, 3]: return "peak"
    if month in [4, 5, 6]:             return "summer"
    return "monsoon"

def safe_index(lst, val, default=0):
    try:
        return lst.index(val)
    except ValueError:
        return default


# ─── MAIN PREDICTION FUNCTION ─────────────────────────────────────────────────
def predict_tourism_cost(
    destination:    str,
    trip_days:      int,
    num_travellers: int  = 2,
    hotel_tier:     str  = "midrange",   # for fallback pricing
    traveller_type: str  = "couple",
    trip_purpose:   str  = "leisure",
    travel_month:   int  = 1,
) -> dict:
    """
    Returns total tourism/activities cost for the trip (all travellers).

    Breakdown:
      entry_fees   = avg entry fee × spots_per_day × days × travellers
      misc         = guide/transport/misc buffer × days × travellers
      activities   = entry_fees + misc
    """

    # ── Look up city ──────────────────────────────────────────────────────────
    city_key  = destination.strip().title()
    city_data = CITY_LOOKUP.get(city_key)

    if city_data:
        base_daily = float(city_data["daily_activity_cost_inr"])
        dest_type  = city_data.get("top_dest_type", "other")
        num_spots  = float(city_data.get("num_spots", 10))
        pop_score  = float(city_data.get("popularity_score", 0.5))
    else:
        # City not in training data — use hotel_tier fallback
        base_daily = FALLBACK_DAILY.get(hotel_tier, 1200)
        dest_type  = "other"
        num_spots  = 10.0
        pop_score  = 0.5

    season = get_season(travel_month)

    # ── Build feature row ─────────────────────────────────────────────────────
    row = {
        "dest_type_enc":       safe_index(DEST_TYPES, dest_type, 7),
        "traveller_type_enc":  safe_index(TRAVELLER_TYPES, traveller_type.lower(), 1),
        "season_enc":          safe_index(SEASONS, season, 0),
        "purpose_enc":         safe_index(PURPOSES, trip_purpose.lower(), 0),
        "trip_days":           trip_days,
        "num_spots":           num_spots,
        "popularity_score":    pop_score,
    }

    X = pd.DataFrame([row])[FEATURE_COLS]
    multiplier = float(MODEL.predict(X)[0])

    # Clip to the same realistic range used during training
    multiplier = float(np.clip(multiplier, REALISTIC_MIN, REALISTIC_MAX))

    # ── Compute cost ──────────────────────────────────────────────────────────
    daily_per_person = base_daily * multiplier
    total_mid        = round(daily_per_person * num_travellers * trip_days)

    return {
        "total_low":        round(total_mid * 0.80),
        "total_mid":        total_mid,
        "total_high":       round(total_mid * 1.30),
        "daily_per_person": round(daily_per_person),
        "multiplier_used":  round(multiplier, 3),
        "base_daily_cost":  round(base_daily),
        "inputs": {
            "destination":    destination,
            "trip_days":      trip_days,
            "num_travellers": num_travellers,
            "traveller_type": traveller_type,
            "trip_purpose":   trip_purpose,
            "season":         season,
            "dest_type":      dest_type,
            "city_found":     city_data is not None,
        }
    }


# ─── CLI TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        dict(destination="Goa",      trip_days=7,  num_travellers=2,
             traveller_type="couple",  trip_purpose="leisure",    travel_month=12),
        dict(destination="Manali",   trip_days=5,  num_travellers=2,
             traveller_type="couple",  trip_purpose="adventure",  travel_month=1),
        dict(destination="Varanasi", trip_days=3,  num_travellers=4,
             traveller_type="family",  trip_purpose="pilgrimage", travel_month=10),
        dict(destination="Jaipur",   trip_days=4,  num_travellers=1,
             traveller_type="solo",    trip_purpose="leisure",    travel_month=3),
    ]

    print("=== TOURISM COST PREDICTIONS ===\n")
    for t in tests:
        r   = predict_tourism_cost(**t)
        inp = r["inputs"]
        print(f"{inp['destination']} | {t['trip_days']}d | "
              f"{inp['traveller_type']} | {inp['trip_purpose']} | {inp['season']}")
        print(f"  Dest type       : {inp['dest_type']}")
        print(f"  Base daily cost : ₹{r['base_daily_cost']:,}/person")
        print(f"  Multiplier      : {r['multiplier_used']}×")
        print(f"  Daily per person: ₹{r['daily_per_person']:,}")
        print(f"  Total (mid)     : ₹{r['total_mid']:,}  "
              f"[₹{r['total_low']:,} – ₹{r['total_high']:,}]")
        print(f"  City in lookup  : {inp['city_found']}")
        print()