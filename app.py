"""
app.py
------
Integrated Streamlit App for India Trip Planning:
- Expense Prediction (ML Models)
- AI Travel Guide Explorer (Gemini API)
- Trip Planner with Itineraries (NLP)
- Travel Assistant Chatbot (Language Support)

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta
import sys, os
import json
import requests
import time

sys.path.insert(0, os.path.dirname(__file__))

from gemini_guide import generate_travel_guide, build_fallback_guide, GEMINI_MODEL
from map_component import render_travel_map, render_map_links
from map_utils import google_maps_directions_url


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_travel_guide(destination: str, weather_json: str):
    """Cache Gemini guides for 1 hour to reduce API quota usage."""
    live_weather = json.loads(weather_json) if weather_json else None
    return generate_travel_guide(destination, live_weather)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "RoamIndia",
    page_icon   = "🧭",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── LOAD MODELS & FUNCTIONS ─────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    try:
        from aggregator.aggregator import predict_trip_cost
        return predict_trip_cost
    except Exception as e:
        st.error(f"Error loading predictor: {e}")
        return None

def _fetch_live_weather(location):
    """Fetch weather from OpenWeatherMap when WEATHER_API_KEY is set."""
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), "trip planner", ".env")
    load_dotenv(env_path)
    weather_key = os.getenv("WEATHER_API_KEY")
    if not weather_key:
        return None
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={location}&units=metric&appid={weather_key}"
        )
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return None
        data = response.json()
        return {
            "temp": f"{round(data['main']['temp'])}°C",
            "desc": data["weather"][0]["description"].title(),
            "wind": f"{round(data['wind']['speed'] * 3.6)} km/h",
            "humid": f"{data['main']['humidity']}%",
            "sub": f"Real-time data for {location}",
        }
    except Exception:
        return None


def _fetch_live_hotels(destination, checkin_date, checkout_date, adults, children):
    """Search Google Hotels through SerpAPI for current properties and rates."""
    from dotenv import load_dotenv

    env_path = os.path.join(os.path.dirname(__file__), "trip planner", ".env")
    load_dotenv(env_path)
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return [], "SERPAPI_API_KEY is not configured."

    try:
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "engine": "google_hotels",
                "q": f"hotels in {destination}, India",
                "check_in_date": checkin_date,
                "check_out_date": checkout_date,
                "adults": adults,
                "children": children,
                "currency": "INR",
                "gl": "in",
                "hl": "en",
                "api_key": api_key,
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("error"):
            return [], data["error"]

        hotels = []
        for property_data in data.get("properties") or []:
            nightly_rate = property_data.get("rate_per_night") or {}
            total_rate = property_data.get("total_rate") or {}
            prices = property_data.get("prices") or []
            hotels.append({
                "name": property_data.get("name", "Hotel"),
                "rating": property_data.get("overall_rating"),
                "reviews": property_data.get("reviews"),
                "nightly_price": nightly_rate.get("lowest"),
                "total_price": total_rate.get("lowest"),
                "link": property_data.get("link") or next(
                    (price.get("link") for price in prices if price.get("link")),
                    None,
                ),
            })
        return hotels, None
    except (requests.RequestException, ValueError):
        return [], "The hotel provider request failed. Check the API key, quota, and connection."


def get_travel_guide(destination):
    """
    Fetch AI-generated travel guide from Gemini API.
    Falls back to static data if the API is unavailable.
    """
    live_weather = _fetch_live_weather(destination)
    weather_json = json.dumps(live_weather) if live_weather else ""

    try:
        return _cached_travel_guide(destination, weather_json)
    except Exception as e:
        err = str(e)
        print(f"Gemini error: {type(e).__name__}: {e}")
        if "429" in err or "quota" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            st.warning(
                "Gemini free-tier quota reached. Using offline guide. "
                f"Try again later or set `GEMINI_MODEL=gemini-2.5-flash-lite` in "
                "`trip planner/.env` (current default: "
                f"`{GEMINI_MODEL}`)."
            )
        else:
            st.warning(
                f"Could not reach Gemini AI ({e}). Showing a basic guide instead. "
                "Check GEMINI_API_KEY in india_trip_predictor/trip planner/.env"
            )
        return build_fallback_guide(destination, live_weather)

def generate_travel_response(query, language):
    """Generate responses for travel-related questions."""
    query_lower = query.lower()

    # Keyword-based responses
    if any(word in query_lower for word in ["bus", "transport", "road"]):
        return ("🚌 **Bus Travel:** You can find buses at major bus terminals. "
                "Use apps like BusHire or redBus to book. Fares are usually ₹20-500 depending on distance. "
                "Long-distance buses often have sleeper options.")
    elif any(word in query_lower for word in ["restaurant", "food", "eat", "dinner"]):
        return ("🍛 **Dining:** India has diverse cuisines! Try local street food (cost-effective), "
                "dhabas (roadside eateries), or restaurants. Most meals cost ₹200-1000. Always check hygiene. "
                "Vegetarian options are widely available.")
    elif any(word in query_lower for word in ["safe", "danger", "security"]):
        return ("🛡️ **Safety:** Indian cities are generally safe for tourists. "
                "Avoid isolated areas at night. Use registered taxis, don't flash valuables, "
                "and stay aware of surroundings. Women travelers: use women-only transport when available.")
    elif any(word in query_lower for word in ["time", "when", "season", "weather"]):
        return ("🌤️ **Best Time to Visit:** October-March is ideal (cool and dry). "
                "Avoid May-July (very hot) and June-September (monsoon). "
                "Peak season (Nov-Feb) means more tourists and higher prices.")
    elif any(word in query_lower for word in ["money", "price", "cost", "budget"]):
        return ("💰 **Budget Guide:** Daily costs vary: Budget ₹1,000-2,000, "
                "Mid-range ₹3,000-6,000, Luxury ₹7,000+. "
                "Street food: ₹50-200, Restaurants: ₹300-1,000, Hotels: ₹800-5,000+.")
    elif any(word in query_lower for word in ["hotel", "room", "stay", "accommodation"]):
        return ("🏨 **Accommodation:** Book via Booking.com, MakeMyTrip, or OYO. "
                "Budget hotels: ₹800-1,500, Mid-range: ₹2,000-4,000, Luxury: ₹5,000+. "
                "Book 2-4 weeks in advance for better rates.")
    elif any(word in query_lower for word in ["flight", "airplane", "air"]):
        return ("✈️ **Flights:** Major airlines: IndiGo, SpiceJet, Air India, Vistara. "
                "Book 6-8 weeks ahead for best prices. Economy class: ₹2,000-12,000 depending on distance. "
                "Use Skyscanner, MakeMyTrip, or Cleartrip to compare.")
    else:
        return (f"👋 Thanks for your question about {query}! "
                "I can help with info about transportation, food, safety, weather, budgeting, "
                "hotels, and flights. Feel free to ask more specific questions!")

predict_trip_cost = load_predictor()
model_loaded = predict_trip_cost is not None

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
CITIES = sorted([
    "Mumbai", "Delhi", "Bengaluru", "Chennai", "Kolkata",
    "Hyderabad", "Kochi", "Goa", "Jaipur", "Ahmedabad",
    "Pune", "Bhopal", "Lucknow", "Patna", "Bhubaneswar",
    "Guwahati", "Srinagar", "Chandigarh", "Indore", "Nagpur",
    "Varanasi", "Manali", "Shimla", "Darjeeling", "Udaipur",
    "Mysuru", "Amritsar", "Rishikesh", "Haridwar", "Ooty",
    "Munnar", "Port Blair", "Ladakh"
])

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align: center;'>RoamIndia</h1>",
    unsafe_allow_html=True
)

# ─── CREATE TABS FOR DIFFERENT FEATURES ─────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Expense Predictor",
    "🗺️ Travel Guide Explorer",
    "📋 Trip Planner",
    "🗺️ Live Map Navigator",
    "🤖 Travel Assistant",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: EXPENSE PREDICTOR (Original functionality)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "<p style='text-align: center; color: gray;'>"
        "An end-to-end AI cost estimator for your next Indian vacation.<br>"
        "Predicts flights, hotels, activities, and daily expenses.</p>",
        unsafe_allow_html=True
    )

    if not model_loaded:
        st.warning(
            "⚠️ Model not found. Please ensure all training scripts have been run.",
            icon="🔧"
        )


    # ─── INPUT FORM ───────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("📍 Route & Dates")
        col1, col2 = st.columns(2)
        with col1:
            source_city = st.selectbox("🛫 Flying from", CITIES, index=CITIES.index("Delhi"), key="exp_src")
            departure_date = st.date_input(
                "📅 Departure date",
                value=date.today() + timedelta(days=30),
                min_value=date.today(),
                key="exp_dep"
            )
        with col2:
            dest_city = st.selectbox("🛬 Flying to (Destination)", CITIES, index=CITIES.index("Goa"), key="exp_dst")
            return_date = st.date_input(
                "📅 Return date",
                value=date.today() + timedelta(days=37),
                min_value=date.today() + timedelta(days=1),
                key="exp_ret"
            )

    with st.container(border=True):
        st.subheader("👥 Travellers & Preferences")
        col3, col4, col5 = st.columns(3)
        with col3:
            num_travellers = st.number_input("Adults", min_value=1, max_value=10, value=2, key="exp_adults")
            children = st.number_input("Children", min_value=0, max_value=5, value=0, key="exp_children")
        with col4:
            traveller_type = st.selectbox("Group Type", ["solo", "couple", "family", "group"], index=1, key="exp_type")
            hotel_tier = st.selectbox("Hotel Standard", ["budget", "midrange", "luxury"], index=1, key="exp_hotel")
        with col5:
            trip_purpose = st.selectbox("Trip Purpose", ["leisure", "adventure", "pilgrimage", "business"], key="exp_purpose")
            seat_class = st.selectbox("Flight Class", ["economy", "business"], key="exp_class")

    # ─── VALIDATION ───────────────────────────────────────────────────────────
    errors = []
    if source_city == dest_city:
        errors.append("Origin and destination cannot be the same city.")
    if return_date <= departure_date:
        errors.append("Return date must be after departure date.")

    for err in errors:
        st.error(err)

    # ─── PREDICT ──────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button(
        "✨ Generate Full Trip Estimate",
        type="primary",
        disabled=(not model_loaded or len(errors) > 0),
        use_container_width=True,
        key="exp_predict"
    )

    if predict_btn:
        total_people = num_travellers + children
        trip_days = (return_date - departure_date).days

        with st.spinner("Querying AI models for Flights, Hotels, and Tourism costs..."):
            result = predict_trip_cost(
                origin          = source_city,
                destination     = dest_city,
                departure_date  = departure_date,
                return_date     = return_date,
                num_travellers  = total_people,
                hotel_tier      = hotel_tier,
                traveller_type  = traveller_type,
                seat_class      = seat_class,
                children        = children,
                trip_purpose    = trip_purpose,
            )
            live_hotels, hotel_search_error = _fetch_live_hotels(
                dest_city,
                departure_date.isoformat(),
                return_date.isoformat(),
                num_travellers,
                children,
            )

        st.divider()

        # ── HEADLINE NUMBERS ────────────────────────────────────────────────────
        st.markdown(f"### 📊 Total Trip Cost: {source_city} → {dest_city}")
        st.caption(
            f"{departure_date.strftime('%d %b %Y')} – {return_date.strftime('%d %b %Y')}  "
            f"· {trip_days} days  ·  {total_people} traveller{'s' if total_people > 1 else ''}  "
            f"·  {hotel_tier.title()} standard"
        )

        st.markdown(
            f"<h1 style='text-align: center; color: #1D9E75;'>₹{result['total_mid']:,}</h1>"
            f"<p style='text-align: center; color: gray;'>Expected Range: ₹{result['total_low']:,} – ₹{result['total_high']:,}</p>",
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── BREAKDOWN ───────────────────────────────────────────────────────────
        b = result["breakdown"]

        colA, colB = st.columns([1, 1.2])

        with colA:
            st.markdown("#### Cost Breakdown")
            st.markdown(f"**✈️ Flights:** ₹{b['flights']:,}")
            st.markdown(f"**🏨 Accommodation:** ₹{b['hotel']:,}")
            st.markdown(f"**📸 Activities/Tourism:** ₹{b['activities']:,}")
            st.markdown(f"**🍛 Food & Transport:** ₹{b['food_local']:,}")

        with colB:
            labels = ['Flights', 'Accommodation', 'Activities', 'Food & Local']
            values = [b['flights'], b['hotel'], b['activities'], b['food_local']]
            colors = ['#7F77DD', '#1D9E75', '#D85A30', '#F2C94C']

            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.5, marker=dict(colors=colors))])
            fig.update_layout(
                margin=dict(t=0, b=0, l=0, r=0),
                showlegend=True,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=200
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🏨 Live Hotel Search")
        st.caption("Current Google Hotels results via SerpAPI. Prices and availability can change.")
        if hotel_search_error == "SERPAPI_API_KEY is not configured.":
            st.info(
                "Add `SERPAPI_API_KEY` to `india_trip_predictor/trip planner/.env` "
                "to load live hotel listings."
            )
        elif hotel_search_error:
            st.warning(f"Live hotel search is unavailable: {hotel_search_error}")
        elif not live_hotels:
            st.info(f"No hotel listings were returned for {dest_city} and these dates.")
        else:
            for hotel in live_hotels[:8]:
                with st.container(border=True):
                    st.markdown(f"**{hotel['name']}**")
                    details = []
                    if hotel["rating"] is not None:
                        rating = f"{hotel['rating']} / 5"
                        if hotel["reviews"] is not None:
                            rating += f" ({hotel['reviews']} reviews)"
                        details.append(f"Rating: {rating}")
                    if hotel["nightly_price"]:
                        details.append(f"Per night: {hotel['nightly_price']}")
                    if hotel["total_price"]:
                        details.append(f"Stay total: {hotel['total_price']}")
                    if details:
                        st.write(" · ".join(details))
                    if hotel["link"]:
                        st.link_button("View hotel", hotel["link"])

        # ── AI INSIGHTS & TRIP DETAILS ──────────────────────────────────────────
        st.markdown("#### 💡 AI Insights & Trip Details")

        f_detail = result["flight_detail"]
        h_detail = result["hotel_detail"]
        t_detail = result["tourism_detail"]

        season = h_detail["inputs"]["season"].title()
        days_adv = f_detail["inputs"]["days_advance"]

        with st.expander("🌦️ Seasonality Impact", expanded=True):
            if season == "Peak":
                st.warning(f"You are traveling during the **Peak Season** (Winter/Holidays). Expect up to a 30% premium on hotels and crowded tourist spots. Book well in advance!")
            elif season == "Monsoon":
                st.success(f"You are traveling during the **Monsoon Season** (Off-peak). Our models have applied a significant discount (approx 20-28%) on your accommodation and activities!")
            else:
                st.info(f"You are traveling during the **Summer Season**. Prices are generally average, though hill stations will command a premium while plains remain cheaper.")

        with st.expander("✈️ Flight Pricing Details", expanded=False):
            st.markdown(f"**Estimated Fare:** ₹{f_detail['per_person_return']:,} per person (Return trip)")
            if days_adv <= 14:
                st.warning(f"**Scarcity Premium:** Booking only {days_adv} days ahead incurs a last-minute premium. Booking 30+ days in advance could save you ~20% on flights.")
            elif days_adv <= 30:
                st.info(f"**Booking Window:** You are booking {days_adv} days in advance. This is an average booking window.")
            else:
                st.success(f"**Early Bird:** Good job! Booking {days_adv} days ahead secures near-optimal base fares.")

        with st.expander("🏨 Hotel Breakdown", expanded=False):
            st.markdown(f"**Base Rate:** ₹{h_detail['per_night_mid']:,} per night for a **{hotel_tier.title()}** room.")
            st.markdown(f"**Model Assumption:** For {dest_city}, a {hotel_tier} room typically assumes basic amenities (AC, WiFi, Restaurant).")

        with st.expander("📸 Tourism & Activities", expanded=False):
            if t_detail["inputs"]["city_found"] == "False":
                st.info(f"{dest_city} wasn't in our primary tourism dataset, so we used a fallback daily cost of ₹{t_detail['base_daily_cost']} based on your {hotel_tier} preference.")
            else:
                st.markdown(f"**Destination Profile:** {dest_city} is typically a **{t_detail['inputs']['dest_type'].title()}** destination.")

            st.markdown(f"**Group Multiplier:** Because you are travelling as a **{traveller_type.title()}** for **{trip_purpose.title()}** over **{trip_days} days**, the model applied a demand multiplier of **{t_detail['multiplier_used']}x** to the base daily city rate.")
            st.markdown(f"**Daily Spend:** Expect to spend around **₹{t_detail['daily_per_person']:,}** per person per day on entry fees, guides, and miscellaneous activities.")

        st.divider()
        st.caption(
            "Model trained on Kaggle and MakeMyTrip Indian travel datasets. "
            "Predictions are estimates based on historical AI pricing models. "
        )

        if st.button("🗺️ View this route on map", key="exp_route_map"):
            st.session_state["map_route_from"] = source_city
            st.session_state["map_route_to"] = dest_city
            st.session_state["map_focus_dest"] = dest_city
            st.success(f"Route saved: {source_city} → {dest_city}. Open the **Live Map Navigator** tab.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRAVEL GUIDE EXPLORER (AI-powered destination guide)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("🗺️ AI Travel Guide Explorer")
    st.markdown(
        "Select a destination to get a comprehensive AI-generated travel guide "
        "with weather, itineraries, hotels, restaurants, and hidden gems."
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        guide_destination = st.selectbox(
            "Choose a destination",
            CITIES,
            index=CITIES.index("Goa"),
            key="guide_dest"
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_guide = st.button("🔍 Explore", key="gen_guide")

    if generate_guide:
        with st.spinner(f"Generating comprehensive travel guide for {guide_destination}..."):
            guide = get_travel_guide(guide_destination)

        st.session_state["last_guide"] = guide
        st.session_state["last_guide_dest"] = guide_destination

        source = guide.get("source", "gemini")
        if source == "fallback":
            st.success("Travel guide ready (offline fallback)")
        else:
            st.success("AI travel guide generated with Gemini")
        st.markdown("<br>", unsafe_allow_html=True)

        # Display overview
        if "overview" in guide:
            st.markdown("## 📖 Overview")
            st.markdown(guide["overview"])

        # Display weather section
        if "weather" in guide:
            st.markdown("## 🌤️ Current Weather & Planning")
            col1, col2, col3, col4 = st.columns(4)
            w = guide["weather"]
            with col1:
                st.metric("🌡️ Temperature", w.get("temp", "N/A"))
            with col2:
                st.metric("☁️ Weather", w.get("desc", "N/A"))
            with col3:
                st.metric("💨 Wind Speed", w.get("wind", "N/A"))
            with col4:
                st.metric("💧 Humidity", w.get("humid", "N/A"))

            if "planning_tip" in w:
                st.info(w["planning_tip"])

        # Display best time section
        if "best_time" in guide:
            with st.expander("📅 Best Time to Visit", expanded=True):
                st.markdown(guide["best_time"])

        # Display pro tip
        if "tip" in guide:
            st.markdown("## 💡 Local Pro Tips")
            st.markdown(guide["tip"])

        # Display itineraries
        if "itineraries" in guide:
            st.markdown("## 📅 Suggested Itineraries")
            for day, activities in guide["itineraries"].items():
                with st.expander(f"**Day {day}** - Complete Itinerary", expanded=(day == "1")):
                    for idx, activity in enumerate(activities):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"### ⏰ {activity.get('time', 'N/A')}")
                        with col2:
                            st.markdown(f"### 📍 {activity.get('place', 'N/A')}")

                        # Main description as full paragraphs
                        st.markdown(activity.get('desc', ''))

                        col_food, col_cost = st.columns(2)
                        with col_food:
                            st.markdown(f"**🍽️ Eat:** {activity.get('food', 'Local cuisine')}")
                        with col_cost:
                            st.markdown(f"**💰 Cost:** {activity.get('cost', 'Variable')}")

                        if idx < len(activities) - 1:
                            st.divider()

        # Display accommodations
        if "hotels" in guide:
            st.markdown("## 🏨 Accommodation Recommendations")
            for hotel in guide["hotels"]:
                with st.expander(f"**{hotel.get('name', 'Hotel')}** - {hotel.get('price', 'Pricing')}"):
                    st.markdown(hotel.get('desc', ''))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Rating:** {hotel.get('rating', 'N/A')}")
                    with col2:
                        st.markdown(f"**Price Range:** {hotel.get('price', 'Contact for price')}")

                    if "booking_tips" in hotel:
                        st.info(hotel["booking_tips"])

        # Display restaurants
        if "restaurants" in guide:
            st.markdown("## 🍛 Dining Recommendations")
            for restaurant in guide["restaurants"]:
                with st.expander(f"**{restaurant.get('name', 'Restaurant')}** - {restaurant.get('type', 'Cuisine')}"):
                    st.markdown(restaurant.get('desc', ''))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Type:** {restaurant.get('type', 'Cuisine')}")
                    with col2:
                        st.markdown(f"**Price Range:** {restaurant.get('price_range', 'Variable')}")

                    if "must_try_dishes" in restaurant:
                        st.markdown(f"**Must Try:** {restaurant['must_try_dishes']}")

        # Display attractions
        if "attractions" in guide:
            st.markdown("## 🎯 Top Attractions")
            for attraction in guide["attractions"]:
                with st.expander(f"**{attraction.get('name', 'Attraction')}**", expanded=False):
                    st.markdown(attraction.get('desc', ''))

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**⏱️ Timing:** {attraction.get('timing', 'N/A')}")
                    with col2:
                        st.markdown(f"**💵 Entry:** {attraction.get('entry_fee', 'N/A')}")
                    with col3:
                        st.markdown(f"**Duration:** 2-3 hours")

                    if "tips" in attraction:
                        st.info(f"**💡 Insider Tips:** {attraction['tips']}")

        # Display hidden gems
        if "gems" in guide:
            st.markdown("## 💎 Hidden Gems & Off-Beat Locations")
            for gem in guide["gems"]:
                with st.expander(f"**{gem.get('name', 'Gem')}** - {gem.get('why_special', 'Special spot')}"):
                    st.markdown(gem.get('desc', ''))

                    st.markdown(f"**Why It's Special:** {gem.get('why_special', '')}")
                    st.markdown(f"**How to Reach:** {gem.get('how_to_reach', '')}")
                    st.markdown(f"**Best Time:** {gem.get('best_time', '')}")

        # Display travel tips
        if "travel_tips" in guide:
            st.markdown("## 🛡️ Travel Tips & Practical Information")
            tips = guide["travel_tips"]

            col1, col2 = st.columns(2)
            with col1:
                with st.expander("🛡️ Safety Information"):
                    st.markdown(tips.get('safety', ''))

                with st.expander("🚌 Transportation Guide"):
                    st.markdown(tips.get('transportation', ''))

                with st.expander("🎒 Packing Checklist"):
                    st.markdown(tips.get('packing', ''))

            with col2:
                with st.expander("💰 Budgeting Guide"):
                    st.markdown(tips.get('budgeting', ''))

                with st.expander("🤝 Local Customs & Culture"):
                    st.markdown(tips.get('local_customs', ''))

                with st.expander("📅 Best Months to Visit"):
                    st.markdown(tips.get('best_months', ''))

        st.markdown("---")
        st.markdown("## 🗺️ Destination Map")
        st.caption(
            "Interactive map from AI-TRAVEL-GUIDE — attractions, hidden gems, "
            "and itinerary stops. Click markers for details."
        )
        with st.spinner("Plotting places on map…"):
            render_travel_map(
                guide_destination,
                guide=guide,
                route_from=st.session_state.get("map_route_from"),
                height=460,
            )
        render_map_links(guide_destination, guide)

    elif st.session_state.get("last_guide") and st.session_state.get("last_guide_dest"):
        st.info(
            f"Showing last guide map for **{st.session_state['last_guide_dest']}**. "
            "Click Explore to refresh."
        )
        render_travel_map(
            st.session_state["last_guide_dest"],
            guide=st.session_state["last_guide"],
            route_from=st.session_state.get("map_route_from"),
            height=400,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: TRIP PLANNER (Complete itinerary planning)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📋 Complete Trip Planner")
    st.markdown(
        "Plan your entire trip with cost breakdown, day-wise itineraries, "
        "accommodation, and local recommendations."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        trip_dest = st.selectbox("Destination", CITIES, key="trip_dest", index=CITIES.index("Udaipur"))
    with col2:
        trip_duration = st.number_input("Trip Duration (days)", min_value=2, max_value=30, value=5, key="trip_duration")
    with col3:
        trip_budget = st.number_input("Total Budget (₹)", min_value=10000, max_value=500000, value=50000, key="trip_budget", step=5000)

    create_plan = st.button("📋 Create Detailed Plan", key="create_plan")

    if create_plan:
        st.divider()
        st.subheader(f"🎯 Trip Plan: {trip_dest} for {trip_duration} days")

        # Budget breakdown
        budget_per_day = trip_budget // trip_duration
        st.metric("Budget per day", f"₹{budget_per_day:,}", f"Total: ₹{trip_budget:,}")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏨 Accommodation", f"₹{budget_per_day * 0.4:,.0f}")
        with col2:
            st.metric("🍛 Food", f"₹{budget_per_day * 0.3:,.0f}")
        with col3:
            st.metric("🚌 Transport", f"₹{budget_per_day * 0.15:,.0f}")
        with col4:
            st.metric("📸 Activities", f"₹{budget_per_day * 0.15:,.0f}")

        st.divider()

        # Day-wise itinerary
        st.subheader("📅 Day-wise Itinerary")
        for day in range(1, trip_duration + 1):
            with st.expander(f"**Day {day}** - Explore {trip_dest}", expanded=(day == 1)):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("**Morning Activity**")
                    st.markdown(f"- Visit main attraction or market in {trip_dest}")
                    st.markdown(f"- Budget: ₹{budget_per_day * 0.1:.0f}")

                    st.markdown("**Afternoon Activity**")
                    st.markdown(f"- Explore heritage sites or natural landmarks")
                    st.markdown(f"- Budget: ₹{budget_per_day * 0.08:.0f}")

                    st.markdown("**Evening Activity**")
                    st.markdown(f"- Local dining and street food experience")
                    st.markdown(f"- Budget: ₹{budget_per_day * 0.07:.0f}")

                with col2:
                    st.markdown("**Meals**")
                    st.markdown(f"- Breakfast: ₹200")
                    st.markdown(f"- Lunch: ₹400")
                    st.markdown(f"- Dinner: ₹500")
                    st.markdown(f"**Total: ₹1,100**")

        st.divider()
        st.subheader("📍 Best Time to Visit")
        st.markdown(
            f"- **Peak Season:** November - February (expect higher prices)\n"
            f"- **Off-Season:** June - August (budget-friendly)\n"
            f"- **Best Weather:** October - November"
        )

        st.subheader("✈️ Getting There")
        st.markdown(f"- Nearest major airport to {trip_dest}\n- Direct flights available from major Indian cities\n- Estimated flight cost: ₹2,000 - ₹8,000 per person")

        st.subheader("🏨 Accommodation Types")
        acc_cols = st.columns(3)
        with acc_cols[0]:
            st.markdown("**Budget Hotels**")
            st.markdown("₹800-1,500/night\n- Basic amenities\n- Shared spaces")
        with acc_cols[1]:
            st.markdown("**Mid-range Hotels**")
            st.markdown("₹2,000-4,000/night\n- Private rooms\n- Good facilities")
        with acc_cols[2]:
            st.markdown("**Luxury Hotels**")
            st.markdown("₹5,000+/night\n- Premium amenities\n- Excellent service")

        st.info("💾 **Tip:** Book accommodations in advance for better rates, especially during peak season!")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: LIVE MAP NAVIGATOR (AI-TRAVEL-GUIDE Leaflet integration)
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🗺️ Live Map Navigator")
    st.markdown(
        "Explore destinations on an interactive map — the same Leaflet engine as "
        "**AI-TRAVEL-GUIDE**. Pan, zoom, and click markers. Generate a travel guide "
        "first to plot attractions and hidden gems."
    )

    def _city_index(city_list, name, default="Goa"):
        name = name if name in city_list else default
        return city_list.index(name)

    mcol1, mcol2, mcol3 = st.columns([2, 2, 1])
    with mcol1:
        map_dest = st.selectbox(
            "Destination on map",
            CITIES,
            index=_city_index(CITIES, st.session_state.get("map_focus_dest", "Goa")),
            key="map_dest_select",
        )
    with mcol2:
        route_options = ["— None —"] + CITIES
        origin_default = st.session_state.get("map_route_from", "— None —")
        if origin_default not in route_options:
            origin_default = "— None —"
        map_origin = st.selectbox(
            "Route from (optional)",
            route_options,
            index=route_options.index(origin_default),
            key="map_origin_select",
        )
    with mcol3:
        st.markdown("<br>", unsafe_allow_html=True)
        show_map_btn = st.button("Update map", key="update_map", type="primary")

    use_guide = st.checkbox(
        "Overlay places from last AI travel guide",
        value=bool(st.session_state.get("last_guide")),
        key="map_use_guide",
    )

    guide_for_map = None
    if use_guide and st.session_state.get("last_guide"):
        if st.session_state.get("last_guide_dest") == map_dest:
            guide_for_map = st.session_state["last_guide"]
        else:
            st.warning(
                f"Last guide was for **{st.session_state.get('last_guide_dest')}**. "
                f"Generate a guide for **{map_dest}** in Travel Guide Explorer."
            )

    route_from = None if map_origin == "— None —" else map_origin
    if route_from:
        st.session_state["map_route_from"] = route_from
    st.session_state["map_focus_dest"] = map_dest

    with st.spinner("Loading map…"):
        render_travel_map(
            map_dest,
            guide=guide_for_map,
            route_from=route_from,
            height=520,
        )

    if show_map_btn:
        st.toast(f"Map updated for {map_dest}")

    link_cols = st.columns([1, 1, 1])
    with link_cols[0]:
        st.link_button(
            "Google Maps — destination",
            f"https://www.google.com/maps/search/?api=1&query={map_dest}%2C+India",
            use_container_width=True,
        )
    with link_cols[1]:
        if route_from:
            st.link_button(
                f"Directions: {route_from} → {map_dest}",
                google_maps_directions_url(route_from, map_dest),
                use_container_width=True,
            )
    with link_cols[2]:
        st.link_button(
            "Full tourist planner (AI-TRAVEL-GUIDE)",
            "http://localhost:3000/dashboard.html",
            use_container_width=True,
            help="Run: cd AI-TRAVEL-GUIDE && npm start",
        )

    with st.expander("About maps in this app"):
        st.markdown(
            """
            - **This tab** uses OpenStreetMap + Leaflet (embedded from `AI-TRAVEL-GUIDE` style maps).
            - **Travel Guide Explorer** adds markers for attractions, gems, hotels, and day plans.
            - **Full tourist site** (`dashboard.html`) adds live GPS, nearby POIs, fuel calculator,
              and Geoapify routing — start it with Node.js on port 3000 when you need those extras.
            """
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: TRAVEL ASSISTANT (NLP-based chatbot)
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🤖 Travel Assistant Chat")
    st.markdown(
        "Ask questions about travel in India! Includes multi-language support "
        "with phrase suggestions in English, Hindi, and Kannada."
    )

    # Language selection
    col1, col2 = st.columns([1, 3])
    with col1:
        chat_lang = st.selectbox(
            "Language",
            ["English", "Hindi (हिंदी)", "Kannada (ಕನ್ನಡ)"],
            key="chat_lang"
        )

    st.divider()

    # Display common phrases based on selected language
    st.subheader("📝 Common Phrases")

    phrases = {
        "English": [
            "Where is the nearest bus stop?",
            "How much is the fare?",
            "Can you recommend a good restaurant?",
            "Where are the tourist attractions?",
            "What's the best time to visit?",
            "Is it safe to travel at night?"
        ],
        "Hindi (हिंदी)": [
            "बस स्टॉप कहाँ है? (Bus stop kahan hai?)",
            "किराया कितना है? (Kiraay kitna hai?)",
            "कोई अच्छा रेस्टोरेंट सुझाएं (Koi acha restaurant)",
            "पर्यटन स्थल कहाँ हैं? (Paryatan sthal kahan hain?)",
            "घूमने के लिए सबसे अच्छा समय कब है?",
            "क्या रात में यात्रा करना सुरक्षित है?"
        ],
        "Kannada (ಕನ್ನಡ)": [
            "ಬಸ್ ನಿಲ್ದಾಣ ಎಲ್ಲಿದೆ? (Bus nildan ellide?)",
            "ದರ ಎಷ್ಟು? (Dar eshtu?)",
            "ಒಂದು ಉತ್ತಮ ರೆಸ್ತೋರೆಂಟ್ ಅನುಮೋದಿಸಿ (Restaurant suggest)",
            "ಪರ್ಯಟನ ಸ್ಥಳಗಳು ಎಲ್ಲಿ? (Tourist sthal ellide?)",
            "ಭೇಟಿ ನೀಡಲು ಉತ್ತಮ ಸಮಯ ಯಾವಾಗ? (Best time?)",
            "ರಾತ್ರಿಯಲ್ಲಿ ಪ್ರಯಾಣ ಸುರಕ್ಷಿತವೇ? (Night travel safe?)"
        ]
    }

    lang_key = chat_lang.split("(")[0].strip() if "(" in chat_lang else chat_lang

    cols = st.columns(2)
    for idx, phrase in enumerate(phrases.get(chat_lang, phrases["English"])):
        col = cols[idx % 2]
        with col:
            if col.button(f"💬 {phrase}", key=f"phrase_{idx}"):
                st.session_state.user_query = phrase

    st.divider()

    # Chat interface
    st.subheader("💬 Ask Your Question")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input(
        "Type your travel question here...",
        key="user_input",
        placeholder=f"E.g., {phrases.get(chat_lang, phrases['English'])[0]}"
    )

    send_btn = st.button("📤 Send", key="send_message")

    if send_btn and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "message": user_input})

        # Generate bot response
        bot_response = generate_travel_response(user_input, chat_lang)
        st.session_state.chat_history.append({"role": "assistant", "message": bot_response})

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**You:** {msg['message']}")
            else:
                st.markdown(f"**Assistant:** {msg['message']}")

    # Travel tips
    st.divider()
    with st.expander("💡 General Travel Tips for India"):
        st.markdown("""
        **Before You Go:**
        - Get travel insurance
        - Check visa requirements
        - Notify your bank of travel dates
        - Book flights 6-8 weeks in advance

        **During Your Trip:**
        - Keep copies of important documents
        - Avoid traveling alone at night
        - Use registered taxis/Uber
        - Try local food but be cautious
        - Respect local customs and traditions

        **Health & Safety:**
        - Carry basic medications
        - Drink bottled water
        - Use sunscreen and insect repellent
        - Get vaccinated (Typhoid, Hepatitis A)
        - Travel insurance is essential
        """)


# ─── GLOBAL FOOTER ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.8em;'>"
    "RoamIndia | Machine learning, Gemini AI, and interactive trip planning | "
    "Always check latest travel advisories before booking</p>",
    unsafe_allow_html=True
)
