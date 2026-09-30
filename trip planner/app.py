import os
import sys
import json
from datetime import date
import requests
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from dotenv import load_dotenv

# Parent package: shared Gemini travel guide logic
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from gemini_guide import generate_travel_guide, build_fallback_guide

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path)

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

def get_live_weather(location):
    """Fetches real-time weather data from OpenWeatherMap API"""
    if not WEATHER_API_KEY:
        return {"temp": "24°C", "desc": "Clear Skies", "wind": "10 km/h", "humid": "55%", "sub": "Weather API Key missing"}

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&units=metric&appid={WEATHER_API_KEY}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            temp = round(data["main"]["temp"])
            humidity = data["main"]["humidity"]
            wind_speed = round(data["wind"]["speed"] * 3.6)  # Convert m/s to km/h
            description = data["weather"][0]["description"].title()

            return {
                "temp": f"{temp}°C",
                "desc": description,
                "wind": f"{wind_speed} km/h",
                "humid": f"{humidity}%",
                "sub": f"Real-time data fetched for {location}"
            }
    except Exception as e:
        print(f"Weather API error: {e}")

    return {"temp": "22°C", "desc": "Misty Clouds", "wind": "8 km/h", "humid": "60%", "sub": "Live data temporarily offline"}


# --- ROUTING FOR APP PAGES ---

@app.route("/planner", methods=["GET"])
def planner_page():
    """Serves the standalone trip planner interface."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "index.html")


@app.route("/planner.css", methods=["GET"])
def planner_stylesheet():
    """Serves the trip planner stylesheet."""
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "planner.css")


@app.route("/calculator", methods=["GET"])
def calculator_page():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "calculator.html")


@app.route("/calculator.css", methods=["GET"])
def calculator_stylesheet():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "calculator.css")


@app.route("/calculator.js", methods=["GET"])
def calculator_script():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "calculator.js")


@app.route("/planner-map", methods=["GET"])
def planner_map():
    from map_utils import build_map_payload

    destination = request.args.get("destination", "India").strip() or "India"
    payload = json.dumps(build_map_payload(destination)).replace("</", "<\\/")
    map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "components", "travel_map.html")
    with open(map_path, encoding="utf-8") as map_file:
        html = map_file.read().replace("__MAP_DATA__", payload)
    return Response(html, mimetype="text/html")

@app.route('/', methods=['GET'])
def home():
    """Serves the main travel guide interface"""
    return jsonify({
        "status": "Flask API running",
        "endpoint": "/api/explore",
        "method": "POST",
        "debug": {
            "api_key_exists": bool(GOOGLE_API_KEY),
        }
    })

@app.route("/debug", methods=['GET'])
def debug():
    """Debug endpoint to check configuration"""
    return jsonify({
        "gemini_configured": bool(GOOGLE_API_KEY),
        "weather_key": "SET" if WEATHER_API_KEY else "MISSING",
        "cwd": os.getcwd(),
        "script_dir": os.path.dirname(os.path.abspath(__file__))
    })

@app.route("/dashboard")
def dashboard_page():
    return jsonify({"status": "Dashboard endpoint active"})

@app.route("/history")
def history_page():
    return jsonify({"status": "History endpoint active"})

@app.route("/<page>")
def static_pages(page):
    return jsonify({"status": f"Page {page} endpoint active"})


# --- TRAVEL EXPLORE API ROUTE ---

@app.route('/api/explore', methods=['POST'])
def explore_destination():
    """API endpoint for travel guide generation"""
    data = request.json or {}
    destination = data.get("destination", "").strip()

    if not destination:
        return jsonify({"error": "No destination provided"}), 400

    live_weather = get_live_weather(destination)

    try:
        guide = generate_travel_guide(destination, live_weather)
        return jsonify(guide)
    except Exception as e:
        print(f"Gemini error for {destination}: {e}")
        fallback = build_fallback_guide(destination, live_weather)
        fallback["api_error"] = str(e)[:300]
        return jsonify(fallback), 200


@app.route("/api/estimate", methods=["POST"])
def estimate_trip():
    data = request.get_json(silent=True) or {}
    try:
        origin = str(data.get("origin", "")).strip()
        destination = str(data.get("destination", "")).strip()
        departure = date.fromisoformat(data.get("departure_date", ""))
        return_date = date.fromisoformat(data.get("return_date", ""))
        if not origin or not destination:
            raise ValueError("Choose both an origin and destination.")
        if origin == destination:
            raise ValueError("Origin and destination must be different.")
        if return_date <= departure:
            raise ValueError("Return date must be after departure.")
        adults = int(data.get("adults", 2))
        children = int(data.get("children", 0))
        if adults < 1 or adults > 10 or children < 0 or children > 5:
            raise ValueError("Choose 1-10 adults and 0-5 children.")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from aggregator.aggregator import predict_trip_cost

        result = predict_trip_cost(
            origin=origin,
            destination=destination,
            departure_date=departure,
            return_date=return_date,
            num_travellers=adults + children,
            children=children,
            hotel_tier=data.get("hotel_tier", "midrange"),
            traveller_type=data.get("traveller_type", "couple"),
            seat_class=data.get("seat_class", "economy"),
            trip_purpose=data.get("trip_purpose", "leisure"),
        )
        return jsonify(result)
    except (TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        app.logger.exception("Trip estimate failed")
        return jsonify({"error": "The trip estimate could not be generated. Check that trained model files are available."}), 500


if __name__ == "__main__":
    # API-only server: no auto browser, no debug reloader (reloader spawns extra processes)
    print("Travel Guide API: http://127.0.0.1:5000/api/explore  (Ctrl+C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)