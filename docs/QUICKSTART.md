# 🚀 Quick Start Guide

## Get Running in 2 Minutes

### Option 1: Basic Setup (No Gemini API)
```bash
# 1. Navigate to project directory
cd c:\expensemodel\india_trip_predictor

# 2. Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App opens automatically at http://localhost:8501

---

### Option 2: Full Setup (With Gemini AI Travel Guide)
```bash
# 1. Navigate to project directory
cd c:\expensemodel\india_trip_predictor

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Get Gemini API Key
# Visit: https://ai.google.dev/
# Copy your API key

# 4. Create .env file in trip_planner directory
cd trip_planner
type nul > .env
# Add to .env:
# GEMINI_API_KEY=your_key_here
# WEATHER_API_KEY=optional_key_here

# 5. Install Node.js dependencies (for auth server)
cd AI-TRAVEL-GUIDE
npm install

# 6. Start Flask Travel Guide API (Terminal 1)
cd ..\trip_planner
python app.py
# Flask runs on http://localhost:5000

# 7. Start Streamlit app (Terminal 2)
cd ..
streamlit run app.py
# Streamlit runs on http://localhost:8501
```

---

## ✅ What to Expect

### If everything works:
- ✅ Streamlit dashboard opens with 4 tabs
- ✅ Tab 1 shows expense predictor form
- ✅ Tab 2 can explore AI travel guides (with API) or fallback data
- ✅ Tab 3 shows trip planning interface
- ✅ Tab 4 shows travel assistant chatbot

### Common Issues & Fixes

**"ModuleNotFoundError: No module named..."**
```bash
pip install -r requirements.txt
# or individually: pip install streamlit plotly requests
```

**"Connection refused" on port 5000**
- Flask server not running? Start it first in separate terminal
- Port already in use? Change in trip_planner/app.py, line ~250

**Travel Guide showing fallback data**
- Gemini API key missing or incorrect
- Check trip_planner/.env file
- Restart Flask server after setting keys

**Slow first load**
- Normal! Models cache on first run
- Subsequent loads are instant

---

## 🎮 Try These Features

### 1. Expense Predictor
- Set source: Delhi, destination: Goa
- Dates: 30 days from today, 7 days duration
- Click "Generate Full Trip Estimate"
- View ₹ breakdown and insights

### 2. Travel Guide
- Select: Goa
- Click "🔍 Explore"
- See itineraries, hotels, restaurants

### 3. Trip Planner
- Destination: Udaipur
- Duration: 5 days
- Budget: ₹50,000
- Click "📋 Create Detailed Plan"

### 4. Travel Assistant
- Try: "How to reach Goa from Delhi?"
- Or: "What's the best time to visit?"
- Or: "Tell me about hotels in Mumbai"

---

## 📊 App Structure

```
┌─────────────────────────────────────┐
│  STREAMLIT FRONTEND (app.py)        │
├─────────────────────────────────────┤
│ Tab 1: Expense Predictor            │
│ ├─ ML Models (XGBoost, etc)         │
│ ├─ Flight/Hotel/Activity costs      │
│ └─ Cost breakdown charts            │
│                                     │
│ Tab 2: Travel Guide                 │
│ ├─ Gemini API Backend               │
│ ├─ Weather data                     │
│ ├─ Itineraries                      │
│ └─ Hotels/Restaurants               │
│                                     │
│ Tab 3: Trip Planner                 │
│ ├─ Budget calculator                │
│ ├─ Day-wise planning                │
│ └─ Travel tips                      │
│                                     │
│ Tab 4: Travel Assistant             │
│ ├─ NLP responses                    │
│ ├─ Multi-language support           │
│ ├─ Phrase suggestions               │
│ └─ Chat history                     │
└─────────────────────────────────────┘
         ↓↑ REST API
┌─────────────────────────────────────┐
│  FLASK BACKEND (trip_planner/)      │
├─────────────────────────────────────┤
│ Gemini API Integration              │
│ ├─ Travel guide generation          │
│ ├─ Weather API calls                │
│ └─ Itinerary creation               │
└─────────────────────────────────────┘
```

---

## 💡 Pro Tips

1. **Faster Setup:** Skip Gemini API setup initially, just use fallback data
2. **Testing:** Use predefined cities (Mumbai, Delhi, Goa, etc.)
3. **Performance:** First run is slow, subsequent runs are fast (caching)
4. **Development:** Streamlit auto-reloads on code changes
5. **Debugging:** Check terminal for error messages

---

## 🔧 Troubleshooting Checklist

- [ ] Python 3.8+ installed? (`python --version`)
- [ ] Requirements installed? (`pip list | grep streamlit`)
- [ ] Flask running on port 5000? (Check terminal 1)
- [ ] Streamlit running on port 8501? (Check terminal 2)
- [ ] .env file in trip_planner/? (If using Gemini API)
- [ ] No port conflicts? (`netstat -ano | findstr 8501`)

---

## 📞 Need Help?

1. Check INTEGRATION_GUIDE.md for detailed documentation
2. Check Flask/Streamlit logs in terminal
3. Verify all API keys are correct
4. Try restarting both servers

---

**Happy Trip Planning! 🌍✈️🏖️**
