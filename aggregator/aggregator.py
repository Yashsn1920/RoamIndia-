"""
aggregator.py
-------------
Master function that calls all 3 sub-models and returns the
complete trip cost estimate.

Place this in the project ROOT (india_trip_predictor/).

Usage:
    from aggregator import predict_trip_cost

    result = predict_trip_cost(
        origin           = "Mumbai",
        destination      = "Goa",
        departure_date   = date(2025, 12, 20),
        return_date      = date(2025, 12, 27),
        num_travellers   = 2,
        hotel_tier       = "midrange",
    )
    print(result)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from datetime import date

# ── Import sub-model predictors ───────────────────────────────────────────────
from flight_model.predict_05  import predict_flight_cost
from hotel_model.predict_03   import predict_hotel_cost
from tourism_model.predict_03 import predict_tourism_cost


# ── Food + local transport daily rates (₹ per person per day) ─────────────────
FOOD_LOCAL_RATES = {
    "budget":   350,
    "midrange": 800,
    "luxury":   2000,
}


def predict_trip_cost(
    origin:          str,
    destination:     str,
    departure_date:  date,
    return_date:     date,
    num_travellers:  int  = 2,
    hotel_tier:      str  = "midrange",   # "budget" | "midrange" | "luxury"
    traveller_type:  str  = "couple",     # "solo" | "couple" | "family" | "group"
    seat_class:      str  = "economy",
    airline:         str  = "Any",
    total_stops:     int  = 1,
    children:        int  = 0,
    meal_plan:       str  = "BB",
    trip_purpose:    str  = "leisure",    # "leisure" | "adventure" | "pilgrimage" | "business"
) -> dict:
    """
    Returns full trip cost breakdown in ₹.
    All costs are totals (not per person) unless stated.
    """

    trip_days = (return_date - departure_date).days
    if trip_days <= 0:
        raise ValueError("return_date must be after departure_date")

    adults = max(1, num_travellers - children)

    # ── 1. FLIGHT ─────────────────────────────────────────────────────────────
    flight = predict_flight_cost(
        source_city      = origin,
        destination_city = destination,
        departure_date   = departure_date,
        num_travellers   = num_travellers,
        seat_class       = seat_class,
        airline          = airline,
        total_stops      = total_stops,
    )

    # ── 2. HOTEL ──────────────────────────────────────────────────────────────
    hotel = predict_hotel_cost(
        destination_city = destination,
        hotel_tier       = hotel_tier,
        checkin_date     = departure_date,
        checkout_date    = return_date,
        adults           = adults,
        children         = children,
    )

    # ── 3. TOURISM / ACTIVITIES ───────────────────────────────────────────────
    tourism = predict_tourism_cost(
        destination      = destination,
        trip_days        = trip_days,
        num_travellers   = num_travellers,
        hotel_tier       = hotel_tier,
        traveller_type   = traveller_type,
        trip_purpose     = trip_purpose,
        travel_month     = departure_date.month,
    )

    # ── 4. FOOD + LOCAL TRANSPORT ─────────────────────────────────────────────
    daily_rate     = FOOD_LOCAL_RATES.get(hotel_tier, 800)
    food_transport = daily_rate * num_travellers * trip_days

    # ── 5. AGGREGATE ──────────────────────────────────────────────────────────
    components = {
        "flights":       flight["total_mid"],
        "hotel":         hotel["total_mid"],
        "activities":    tourism["total_mid"],
        "food_local":    food_transport,
    }
    base_total = sum(components.values())

    return {
        # Three-number range
        "total_low":  round(base_total * 0.85),
        "total_mid":  round(base_total),
        "total_high": round(base_total * 1.22),

        # Component breakdown
        "breakdown": {k: round(v) for k, v in components.items()},

        # Sub-model details (pass through for UI)
        "flight_detail":  flight,
        "hotel_detail":   hotel,
        "tourism_detail": tourism,

        # Trip meta
        "meta": {
            "origin":         origin,
            "destination":    destination,
            "departure":      str(departure_date),
            "return":         str(return_date),
            "trip_days":      trip_days,
            "num_travellers": num_travellers,
            "hotel_tier":     hotel_tier,
            "seat_class":     seat_class,
        }
    }


# ─── CLI TEST ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = predict_trip_cost(
        origin          = "Mumbai",
        destination     = "Goa",
        departure_date  = date(2025, 12, 20),
        return_date     = date(2025, 12, 27),
        num_travellers  = 2,
        hotel_tier      = "midrange",
        traveller_type  = "couple",
    )

    print("=" * 50)
    print(f"  Mumbai → Goa | 7 days | 2 travellers")
    print("=" * 50)
    m = result["meta"]
    print(f"  {m['departure']}  →  {m['return']}  ({m['trip_days']} days)")
    print()
    print(f"  {'Flights':<20} ₹{result['breakdown']['flights']:>10,}")
    print(f"  {'Hotel':<20} ₹{result['breakdown']['hotel']:>10,}")
    print(f"  {'Activities':<20} ₹{result['breakdown']['activities']:>10,}")
    print(f"  {'Food & local':<20} ₹{result['breakdown']['food_local']:>10,}")
    print(f"  {'-'*34}")
    print(f"  {'TOTAL (mid)':<20} ₹{result['total_mid']:>10,}")
    print(f"  {'Range':<20} ₹{result['total_low']:,} – ₹{result['total_high']:,}")