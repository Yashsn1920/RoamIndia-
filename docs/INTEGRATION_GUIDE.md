# 🇮🇳 Integrated India Trip Planner - Integration Guide

## Overview

The India Trip Planner is now an **integrated multi-feature application** combining:
- 💰 **ML-based Expense Prediction** (Flights, Hotels, Activities)
- 🗺️ **AI Travel Guide Explorer** (Gemini API powered)
- 📋 **Trip Planner with Itineraries** (NLP-based planning)
- 🤖 **Travel Assistant Chatbot** (Multi-language support)

---

## 🚀 Features

### 1. **💰 Expense Predictor (Tab 1)**
The original core feature - predicts total trip costs using three ML models:
- ✈️ **Flight Costs:** XGBoost predicts fares based on dates, cities, class
- 🏨 **Hotel Costs:** XGBoost with seasonal adjustments
- 📸 **Activity Costs:** Gradient Boosting with dynamic multipliers

**How to use:**
1. Select source and destination cities
2. Choose travel dates and traveler preferences
3. Click "Generate Full Trip Estimate"
4. View detailed cost breakdown and AI insights

---

### 2. **🗺️ Travel Guide Explorer (Tab 2)**
AI-generated comprehensive travel guides using Gemini API:
- 🌤️ **Live Weather Data** - Current conditions for destination
- 📅 **Multi-day Itineraries** - Activities, timing, and food pairing
- 🏨 **Hotel Recommendations** - Curated accommodations
- 🍛 **Restaurant Suggestions** - Authentic local dining
- 🎯 **Top Attractions** - Major landmarks and sites
- 💎 **Hidden Gems** - Off-beat locations

**How to use:**
1. Select a destination from the dropdown
2. Click "🔍 Explore"
3. Browse weather, itineraries, hotels, restaurants, attractions, and hidden gems

**Note:** Requires Gemini API (falls back to static data if unavailable)

---

### 3. **📋 Trip Planner (Tab 3)**
Complete trip planning with:
- 💵 **Budget Breakdown** - Daily allocation across categories
- 📅 **Day-wise Itineraries** - Suggested activities for each day
- 🏨 **Accommodation Guide** - Budget, mid-range, and luxury options
- 🌤️ **Best Time to Visit** - Seasonal recommendations
- ✈️ **Transportation Info** - Flight and ground transport guidance
- 💡 **Practical Tips** - Booking and travel advice

**How to use:**
1. Select destination and trip duration
2. Set total budget
3. Click "📋 Create Detailed Plan"
4. Review day-wise breakdown and recommendations

---

### 4. **🤖 Travel Assistant Chat (Tab 4)**
Multi-language travel chatbot with:
- 💬 **NLP-Based Responses** - Smart answers to travel questions
- 🌐 **Multi-language Support** - English, Hindi, Kannada
- 📝 **Quick Phrase Buttons** - Common travel phrases
- 💾 **Chat History** - Maintain conversation context
- 💡 **Travel Tips** - Safety, health, and practical advice

**Supported Topics:**
- Transportation (buses, trains, flights)
- Dining and restaurants
- Safety and security
- Weather and seasons
- Budgeting and costs
- Accommodations
- General travel info

**How to use:**
1. Select language (English, Hindi, Kannada)
2. Click a phrase or type your question
3. View AI-generated response
4. Continue conversation with follow-up questions

---

## 📋 Integration Architecture

```
app.py (Streamlit Main App)
├── Tab 1: Expense Predictor
│   ├── aggregator.predict_trip_cost()
│   ├── flight_model/
│   ├── hotel_model/
│   └── tourism_model/
│
├── Tab 2: Travel Guide Explorer
│   ├── get_travel_guide()
│   └── Gemini API (trip_planner/app.py:5000)
│
├── Tab 3: Trip Planner
│   ├── Budget calculator
│   └── Day-wise itinerary generator
│
└── Tab 4: Travel Assistant
    ├── generate_travel_response()
    ├── NLP query processing
    └── Multi-language phrase dictionary
```

---

## 🔧 Setup Instructions

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) Gemini API key for Travel Guide feature

### Installation

1. **Install dependencies:**
```bash
cd c:\expensemodel\india_trip_predictor
pip install -r requirements.txt
```

2. **(Optional) Setup Gemini API:**
If you want Travel Guide Explorer to use live AI generation:
```bash
# Create a .env file in the trip_planner directory
echo GEMINI_API_KEY=your_api_key_here > trip_planner\.env
echo WEATHER_API_KEY=your_weather_key_here >> trip_planner\.env
```

3. **Start the Travel Guide API (in separate terminal):**
```bash
cd trip_planner
python app.py
# This starts Flask server on http://localhost:5000
```

4. **Run the main Streamlit app:**
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📦 Dependencies Added

New dependencies for integrated features:
- `requests>=2.31.0` - For API calls to Gemini
- `python-dotenv>=1.0.0` - Environment variable management
- `google-genai>=0.3.0` - Gemini API client (for Travel Guide)

---

## 🔌 API Endpoints

### Gemini Travel Guide API
- **URL:** `POST http://localhost:5000/api/explore`
- **Request Body:**
```json
{
  "destination": "Goa"
}
```
- **Response:** Comprehensive travel guide (JSON)

---

## 💾 File Structure

```
india_trip_predictor/
├── app.py                          # Main Streamlit app (INTEGRATED)
├── requirements.txt                 # Updated dependencies
├── INTEGRATION_GUIDE.md            # This file
│
├── aggregator/
│   └── aggregator.py              # ML prediction engine
│
├── flight_model/                   # Flight cost predictor
├── hotel_model/                    # Hotel cost predictor
├── tourism_model/                  # Activity cost predictor
│
├── trip_planner/                   # Flask app for AI Travel Guide
│   ├── app.py                     # Gemini API integration
│   ├── server.js                  # User authentication
│   ├── package.json               # Node.js dependencies
│   ├── public/
│   │   ├── index.html            # Travel guide interface
│   │   ├── dashboard.html        # Dashboard
│   │   └── login.html            # Authentication
│   └── users.json                # User data
│
├── AI-TRAVEL-GUIDE/               # Web interface assets
│   ├── server.js                 # Express server
│   ├── public/
│   │   ├── index.html           # Main interface
│   │   ├── dashboard.html       # Dashboard view
│   │   └── login.html           # Login page
│   └── package.json
│
├── nlps.txt                       # NLP phrase definitions
├── data/                          # Training datasets
│   ├── raw/                      # Original datasets
│   └── processed/                # Cleaned data
│
└── models/                        # Trained ML models
```

---

## 🎯 Usage Examples

### Example 1: Plan Delhi → Goa Trip
1. **Tab 1 (Expense Predictor):**
   - Source: Delhi
   - Destination: Goa
   - Dates: 30 days from today
   - Travelers: 2 adults
   - Result: ₹XX,XXX total cost

2. **Tab 2 (Travel Guide):**
   - Select "Goa"
   - Get itineraries, restaurants, beaches

3. **Tab 3 (Trip Planner):**
   - Budget: ₹50,000
   - Duration: 7 days
   - Get day-wise plan

4. **Tab 4 (Chat):**
   - Ask "How to reach Goa from Delhi?"
   - Get transportation tips

---

## 🐛 Troubleshooting

### Issue: Travel Guide returning fallback data
**Solution:**
- Ensure Flask server is running on port 5000
- Check Gemini API key is set correctly
- Review trip_planner/.env file

### Issue: Chat assistant not responding
**Solution:**
- Check internet connection
- Verify no console errors in terminal
- Try simpler, more direct questions

### Issue: Models loading slowly
**Solution:**
- First run may download/cache models
- Subsequent runs will be faster
- Close and reopen app if stuck

---

## 📝 Customization

### Add New Cities
Edit `CITIES` list in app.py (around line 80):
```python
CITIES = sorted([
    "Mumbai", "Delhi", "Bangalore", ..., "YourCity"
])
```

### Modify Budget Categories
Edit Tab 3 budget allocation:
```python
col1.st.metric("🏨 Accommodation", f"₹{budget_per_day * 0.4:,.0f}")
```

### Add More Chat Responses
Extend `generate_travel_response()` function:
```python
elif "keyword" in query_lower:
    return "Your custom response here"
```

---

## 🔐 Security Notes

- Never commit .env files with API keys
- Use environment variables for sensitive data
- Validate user inputs (implemented)
- Keep API keys rotated regularly

---

## 📞 Support

For issues or questions:
1. Check this integration guide
2. Review individual component documentation
3. Check console for error messages
4. Verify all API keys and configurations

---

## 🎓 Technical Details

### ML Models Used
- **XGBoost:** Flights & Hotels (non-linear, tree-based)
- **GradientBoosting:** Activities (ensemble method)
- **Target Encoding:** Categorical variable handling

### NLP Approach
- **Keyword matching:** Quick pattern recognition
- **Intent detection:** Understand user intent
- **Fallback responses:** Handle unknown queries gracefully

### API Integration
- **Gemini API:** For intelligent travel guides
- **OpenWeatherMap:** (optional) For live weather
- **Streamlit:** Frontend framework
- **Flask:** Backend API server

---

## 📊 Performance Metrics

- **Expense Predictor:** Responds in <2 seconds
- **Travel Guide:** Responds in 3-5 seconds (with API)
- **Chat Assistant:** Responds instantly (keyword-based)
- **Concurrent Users:** Streamlit handles multiple sessions

---

## 🚀 Future Enhancements

Possible additions:
- Real-time flight booking integration
- Hotel booking API
- Advanced NLP with transformer models
- Multi-language support for all tabs
- User accounts and saved itineraries
- Mobile app version
- Integration with travel booking sites

---

**Last Updated:** June 2026
**Version:** 2.0 (Integrated)
