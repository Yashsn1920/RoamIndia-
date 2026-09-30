"""
Shared Gemini travel guide generation for Streamlit and Flask apps.
"""

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

_BASE_DIR = Path(__file__).resolve().parent
for _env_file in (_BASE_DIR / "trip planner" / ".env", _BASE_DIR / ".env"):
    if _env_file.exists():
        load_dotenv(_env_file)

# Primary model; fallbacks used when quota/rate limits hit (free tier is per-model)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL_FALLBACK_CHAIN = [
    m.strip()
    for m in os.getenv(
        "GEMINI_MODEL_FALLBACKS",
        "gemini-3.5-flash-lite,gemini-3.5-flash,gemini-2.5-flash-lite,gemini-2.5-flash",
    ).split(",")
    if m.strip()
]


def _get_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _get_client():
    api_key = _get_api_key()
    if not api_key:
        return None
    from google import genai

    return genai.Client(api_key=api_key)


def _parse_json_response(text):
    """Extract and parse JSON from model output."""
    if not text:
        raise ValueError("Empty response from Gemini")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    return json.loads(cleaned)


def _travel_guide_prompt(destination, weather_summary):
    return f"""Create a detailed travel guide for {destination}, India.
Current weather: {weather_summary}.
All prices and cost estimates must be in Indian Rupees (INR), written with the ₹ symbol. Never use dollars, USD, or any other currency. Hotel prices must be realistic per-night INR ranges.

Return ONLY valid JSON with these keys:
- title (string)
- mapLabel (string)
- overview (string, 2-3 sentences)
- weather (object: temp, desc, wind, humid, sub, planning_tip)
- tip (string, local pro-tip)
- best_time (string)
- itineraries (object with keys "1","2","3"; each value is a list of objects with time, place, desc, food, cost)
- hotels (list of objects: name, desc, price as a per-night INR range using ₹, rating, booking_tips)
- restaurants (list: name, desc, type, price_range, must_try_dishes)
- attractions (list: name, desc, timing, entry_fee, tips)
- gems (list: name, desc, why_special, how_to_reach, best_time)
- travel_tips (object: safety, transportation, budgeting, packing, local_customs, best_months)
"""


def _models_to_try() -> list[str]:
    """Ordered unique model list: env default first, then fallbacks."""
    ordered = [GEMINI_MODEL, *MODEL_FALLBACK_CHAIN, "gemini-3.5-flash-lite"]
    seen: set[str] = set()
    unique: list[str] = []
    for name in ordered:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _is_retryable_api_error(exc: Exception) -> bool:
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg


def _is_retired_model_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg and any(term in msg for term in ("no longer available", "not found", "not supported"))


def _generate_with_model(client, model: str, prompt: str):
    from google.genai import types

    return client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )


def generate_travel_guide(destination, live_weather=None):
    """
    Generate an AI travel guide via Gemini. Returns a dict.
    Raises on API/parse errors when the client is configured.
    """
    destination = (destination or "").strip()
    if not destination:
        raise ValueError("No destination provided")

    client = _get_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to india_trip_predictor/trip planner/.env"
        )

    if live_weather is None:
        live_weather = {
            "temp": "24°C",
            "desc": "Pleasant",
            "wind": "10 km/h",
            "humid": "60%",
            "sub": "Weather data unavailable",
        }

    weather_summary = f"{live_weather.get('temp', 'N/A')}, {live_weather.get('desc', 'N/A')}"
    prompt = _travel_guide_prompt(destination, weather_summary)

    models = _models_to_try()
    last_error: Exception | None = None

    for model in models:
        for attempt in range(2):
            try:
                response = _generate_with_model(client, model, prompt)
                guide = _parse_json_response(response.text)
                guide["source"] = "gemini"
                guide["model"] = model
                guide["weather"] = {**live_weather, **guide.get("weather", {})}
                return guide
            except json.JSONDecodeError as e:
                last_error = e
                break  # bad JSON — try next model
            except Exception as e:
                last_error = e
                if _is_retired_model_error(e):
                    break  # try the next configured model
                if _is_retryable_api_error(e) and attempt == 0:
                    time.sleep(4)
                    continue
                if _is_retryable_api_error(e):
                    break  # try next model
                raise

    if last_error and _is_retryable_api_error(last_error):
        raise RuntimeError(
            "Gemini API quota or rate limit reached for all configured models. "
            "Wait a few minutes, use a different API key, or set GEMINI_MODEL=gemini-3.5-flash-lite "
            "in trip planner/.env"
        ) from last_error
    if last_error:
        raise last_error
    raise RuntimeError("Gemini API failed with no response")


def build_fallback_guide(destination, live_weather=None):
    """Static fallback when Gemini is unavailable."""
    destination = (destination or "Unknown Destination").strip()
    if live_weather is None:
        live_weather = {
            "temp": "24°C",
            "desc": "Pleasant",
            "wind": "10 km/h",
            "humid": "60%",
            "sub": "Typical conditions",
        }

    return {
        "title": f"Travel Guide: {destination}",
        "mapLabel": f"{destination} Region",
        "source": "fallback",
        "overview": (
            f"{destination} is a wonderful Indian destination offering unique experiences. "
            "From historical monuments to vibrant markets and natural landscapes, "
            f"{destination} has something for every traveler."
        ),
        "weather": {
            **live_weather,
            "planning_tip": "Pack light clothing for daytime and layers for evening.",
        },
        "tip": "Visit early morning to avoid crowds. Try local street food. Hire local guides for hidden gems.",
        "best_time": "October to March for the best weather.",
        "itineraries": {
            "1": [
                {
                    "time": "Morning (8-12)",
                    "place": f"{destination} Main Landmark",
                    "desc": "Visit early to beat crowds",
                    "food": "Local breakfast",
                    "cost": "₹300-500",
                },
                {
                    "time": "Afternoon (1-5)",
                    "place": "Historical Site",
                    "desc": "Explore heritage areas",
                    "food": "Traditional lunch",
                    "cost": "₹400-600",
                },
                {
                    "time": "Evening (5-10)",
                    "place": "Scenic Viewpoint",
                    "desc": "Enjoy sunset views",
                    "food": "Dinner",
                    "cost": "₹500-800",
                },
            ],
        },
        "hotels": [
            {"name": "Budget Hotel", "desc": "Affordable accommodation", "price": "₹1000-1500", "rating": "4.0"},
            {"name": "Mid-range Hotel", "desc": "Good comfort", "price": "₹2000-3000", "rating": "4.5"},
            {"name": "Luxury Hotel", "desc": "Premium experience", "price": "₹5000+", "rating": "4.8"},
        ],
        "restaurants": [
            {
                "name": "Local Cuisine",
                "desc": "Authentic food",
                "type": "Indian",
                "price_range": "₹300-500",
                "must_try_dishes": "Regional specialties",
            },
        ],
        "attractions": [
            {
                "name": "Main Landmark",
                "desc": "Primary attraction",
                "timing": "9AM-6PM",
                "entry_fee": "₹200-500",
                "tips": "Visit early",
            },
        ],
        "gems": [
            {
                "name": "Hidden Gem",
                "desc": "Off-beat location",
                "why_special": "Authentic",
                "how_to_reach": "Auto",
                "best_time": "Morning",
            },
        ],
        "travel_tips": {
            "safety": "Stay aware. Use registered taxis.",
            "transportation": "Auto ₹10-50, Taxi ₹100-300",
            "budgeting": "₹2000-3000/day",
            "packing": "Light clothes, sunscreen, shoes",
            "local_customs": "Respectful behavior in temples",
            "best_months": "Oct-Mar ideal",
        },
    }
