"""
flight_model/predict_05.py
---------------------------
Loads the trained model and makes predictions for new trip inputs.
This is what the Streamlit app will call.

Usage (standalone):
    python predict_05.py

Or import in your app:
    from flight_model.predict_05 import predict_flight_cost
"""

import numpy as np
import pandas as pd
import joblib
from datetime import date, datetime
from pathlib import Path


# ─── LOAD ARTIFACTS (done once at import) ─────────────────────────────────────
import __main__

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
        smooth = (count * mean + self.smoothing * self.global_mean) / (count + self.smoothing)
        self.mapping = smooth.to_dict()
        return self

    def transform(self, X: pd.Series) -> pd.Series:
        return X.map(self.mapping).fillna(self.global_mean)

    def fit_transform(self, X: pd.Series, y: pd.Series) -> pd.Series:
        self.fit(X, y)
        return self.transform(X)

setattr(__main__, "TargetEncoder", TargetEncoder)

_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL     = joblib.load(_MODEL_DIR / "flight_model.pkl")
ENCODERS  = joblib.load(_MODEL_DIR / "encoders.pkl")
FEAT_COLS = joblib.load(_MODEL_DIR / "feature_cols.pkl")


# ─── HELPERS ──────────────────────────────────────────────────────────────────
CITY_MAP = {
    "bombay": "Mumbai", "mumbai": "Mumbai",
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru",
    "delhi": "Delhi", "new delhi": "Delhi",
    "madras": "Chennai", "chennai": "Chennai",
    "calcutta": "Kolkata", "kolkata": "Kolkata",
    "hyderabad": "Hyderabad",
    "cochin": "Kochi", "kochi": "Kochi",
    "goa": "Goa",
}

def normalise_city(name: str) -> str:
    return CITY_MAP.get(name.strip().lower(), name.strip().title())

def get_season(month: int) -> int:
    if month in [10, 11, 12, 1, 2, 3]: return 2   # peak
    if month in [4, 5, 6]:             return 1   # summer
    return 0                                       # monsoon

def get_advance_bucket(days: int) -> int:
    if days < 0:   return -1   # unknown
    if days <= 3:  return 0    # last minute
    if days <= 14: return 1    # short
    if days <= 45: return 2    # medium
    return 3                   # long

HOLIDAY_MONTHS = {10, 11, 12, 1}

# ─── ROUTE DURATION LOOKUP (hours) ────────────────────────────────────────────
ROUTE_DURATION = {
    "Mumbai__Delhi":        2.0,  "Delhi__Mumbai":        2.0,
    "Bengaluru__Delhi":     2.5,  "Delhi__Bengaluru":     2.5,
    "Mumbai__Kolkata":      2.5,  "Kolkata__Mumbai":      2.5,
    "Chennai__Delhi":       2.8,  "Delhi__Chennai":       2.8,
    "Bengaluru__Hyderabad": 1.2,  "Hyderabad__Bengaluru": 1.2,
    "Mumbai__Goa":          1.2,  "Goa__Mumbai":          1.2,
    "Delhi__Kolkata":       2.0,  "Kolkata__Delhi":       2.0,  # fixed
    "Mumbai__Chennai":      1.8,  "Chennai__Mumbai":      1.8,
    "Mumbai__Hyderabad":    1.5,  "Hyderabad__Mumbai":    1.5,
    "Delhi__Goa":           2.5,  "Goa__Delhi":           2.5,
    "Bengaluru__Mumbai":    1.8,  "Mumbai__Bengaluru":    1.8,
    "Bengaluru__Chennai":   1.0,  "Chennai__Bengaluru":   1.0,
    "Delhi__Hyderabad":     2.2,  "Hyderabad__Delhi":     2.2,
    "Kolkata__Chennai":     2.5,  "Chennai__Kolkata":     2.5,
    "Kochi__Delhi":         3.0,  "Delhi__Kochi":         3.0,
    "Goa__Bengaluru":       1.2,  "Bengaluru__Goa":       1.2,
}

def apply_encoders(row: dict) -> pd.DataFrame:
    """Apply saved target encoders and return a model-ready DataFrame."""
    for col in ["airline", "source_city", "destination_city", "route"]:
        enc = ENCODERS[col]
        val = row.get(col, "Unknown")
        row[f"{col}_enc"] = enc.mapping.get(val, enc.global_mean)
    return pd.DataFrame([row])[FEAT_COLS]


# ─── MAIN PREDICTION FUNCTION ─────────────────────────────────────────────────
def predict_flight_cost(
    source_city:      str,
    destination_city: str,
    departure_date:   date,
    num_travellers:   int   = 1,
    seat_class:       str   = "economy",
    airline:          str   = "IndiGo",
    total_stops:      int   = 1,
    duration_hrs:     float = None,
) -> dict:
    """
    Returns a dict with:
        price_low, price_mid, price_high  (₹, for ALL travellers return trip)
        per_person_oneway                 (₹ one-way per person)
        per_person_return                 (₹ return per person)
    """
    src   = normalise_city(source_city)
    dst   = normalise_city(destination_city)
    route = f"{src}__{dst}"

    today        = date.today()
    days_advance = (departure_date - today).days if isinstance(departure_date, date) else -1
    month        = departure_date.month if hasattr(departure_date, "month") else -1
    day          = departure_date.day   if hasattr(departure_date, "day")   else -1

    if duration_hrs is None:
        duration_hrs = ROUTE_DURATION.get(route, 2.0 + total_stops * 0.8)

    row = {
        "airline":          airline if airline != "Any" else "IndiGo",
        "source_city":      src,
        "destination_city": dst,
        "route":            route,
        "total_stops":      total_stops,
        "duration_hrs":     duration_hrs,
        "dep_hour":         8,
        "is_business":      1 if seat_class.lower() == "business" else 0,
        "days_advance":     days_advance,
        "journey_month":    month,
        "journey_day":      day,
        "season_enc":       get_season(month) if month > 0 else -1,
        "advance_enc":      get_advance_bucket(days_advance),
        "is_holiday_month": 1 if month in HOLIDAY_MONTHS else 0,
        # encoded cols filled by apply_encoders
        "airline_enc": 0, "source_city_enc": 0,
        "destination_city_enc": 0, "route_enc": 0,
    }

    X = apply_encoders(row)

    log_pred      = MODEL.predict(X)[0]
    price_oneway  = np.expm1(log_pred)
    price_return  = price_oneway * 1.85       # return = slight discount on 2×
    total         = price_return * num_travellers

    return {
        "per_person_oneway": round(price_oneway),
        "per_person_return": round(price_return),
        "total_mid":         round(total),
        "total_low":         round(total * 0.85),
        "total_high":        round(total * 1.25),
        "inputs": {
            "source":       src,
            "destination":  dst,
            "date":         str(departure_date),
            "days_advance": days_advance,
            "travellers":   num_travellers,
            "class":        seat_class,
            "airline":      airline,
            "stops":        total_stops,
        }
    }


# ─── CLI TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_cases = [
        dict(source_city="Mumbai",    destination_city="Delhi",
             departure_date=date(2025, 12, 20), num_travellers=2,
             seat_class="economy"),
        dict(source_city="Bengaluru", destination_city="Goa",
             departure_date=date(2025, 11, 5),  num_travellers=1,
             seat_class="economy", total_stops=0),
        dict(source_city="Delhi",     destination_city="Kolkata",
             departure_date=date(2026, 1, 15),  num_travellers=4,
             seat_class="economy"),                                  # fixed: was business
        dict(source_city="Mumbai",    destination_city="Chennai",
             departure_date=date(2025, 12, 25), num_travellers=2,
             seat_class="economy"),
        dict(source_city="Delhi",     destination_city="Goa",
             departure_date=date(2026, 3, 10),  num_travellers=3,
             seat_class="economy"),
    ]

    print("=== SAMPLE PREDICTIONS ===\n")
    for tc in test_cases:
        result = predict_flight_cost(**tc)
        src = result["inputs"]["source"]
        dst = result["inputs"]["destination"]
        n   = result["inputs"]["travellers"]
        cls = result["inputs"]["class"]
        print(f"{src} → {dst}  ({n} traveller{'s' if n>1 else ''}, {cls})")
        print(f"  Per person (return): ₹{result['per_person_return']:,}")
        print(f"  Total range:         ₹{result['total_low']:,} – ₹{result['total_high']:,}")
        print(f"  Best estimate:       ₹{result['total_mid']:,}")
        print()