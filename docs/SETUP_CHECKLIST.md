# ✅ Complete Setup & Verification Checklist

## Pre-Setup Requirements
- [ ] Windows/Linux/Mac with Python 3.8+
- [ ] Internet connection (for API calls and package downloads)
- [ ] 500MB free disk space
- [ ] Administrator access (for installing packages)

---

## Step 1: Environment Setup

### 1.1 Navigate to Project
```bash
cd c:\expensemodel\india_trip_predictor
```
✅ Verify you see: app.py, requirements.txt, INTEGRATION_GUIDE.md, etc.

### 1.2 Create Virtual Environment (Recommended)
```bash
python -m venv venv
.\venv\Scripts\activate
```
✅ Command prompt should now show: (venv)

### 1.3 Upgrade pip
```bash
python -m pip install --upgrade pip
```
✅ Should complete without errors

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- joblib
- streamlit
- plotly
- requests
- python-dotenv
- google-genai
```

✅ No ERROR messages should appear
✅ All packages should be "Successfully installed" or already present

---

## Step 3: Verify Installation

### 3.1 Check Streamlit
```bash
streamlit --version
```
✅ Should show version >= 1.33.0

### 3.2 Check Python Packages
```bash
python -c "import streamlit, plotly, requests; print('All imports OK!')"
```
✅ Should output: "All imports OK!"

### 3.3 Verify Models Exist
```bash
# In PowerShell:
Test-Path ".\flight_model"
Test-Path ".\hotel_model"
Test-Path ".\tourism_model"
Test-Path ".\aggregator"
```
✅ All should return: True

---

## Step 4: Optional - Setup Gemini API (For AI Travel Guides)

### 4.1 Get API Key
1. Go to: https://ai.google.dev/
2. Click "Get API Key"
3. Create new API key in Google Cloud
4. Copy the key

### 4.2 Create Environment File
```bash
# In trip_planner folder:
cd trip_planner
type nul > .env
```

### 4.3 Add Keys to .env
Open trip_planner/.env in text editor:
```
GEMINI_API_KEY=your_key_here_xxxxxxxxxx
WEATHER_API_KEY=optional_weather_key
```
✅ Save and close

### 4.4 Test Flask Backend
```bash
# In trip_planner folder:
python app.py
```
✅ Should see:
```
 * Running on http://127.0.0.1:5000
```
Keep this running! (Open new terminal for step 5)

---

## Step 5: Run Main Streamlit App

### 5.1 Open New Terminal/PowerShell
```bash
# Activate venv if needed:
.\venv\Scripts\activate

# Navigate to project:
cd c:\expensemodel\india_trip_predictor

# Run Streamlit:
streamlit run app.py
```

✅ Should see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://x.x.x.x:8501
```

### 5.2 App Opens Automatically
✅ Browser should open http://localhost:8501
✅ Should see "🇮🇳 India Trip Planner" header
✅ Should see 4 tabs: 💰 📍 📋 🤖

---

## Step 6: Test Each Tab

### Tab 1: 💰 Expense Predictor
1. [ ] Select source city: Delhi
2. [ ] Select destination: Goa
3. [ ] Dates are pre-filled
4. [ ] Select travelers: 2 adults
5. [ ] Click "Generate Full Trip Estimate"
6. [ ] Should see ₹ cost breakdown
✅ Cost should display in chart

### Tab 2: 🗺️ Travel Guide Explorer
1. [ ] Select destination: Goa
2. [ ] Click "🔍 Explore"
3. [ ] Should show weather, itineraries, hotels, restaurants
✅ Works with Gemini API (live) or fallback data

### Tab 3: 📋 Trip Planner
1. [ ] Select destination: Udaipur
2. [ ] Set duration: 5 days
3. [ ] Set budget: ₹50,000
4. [ ] Click "📋 Create Detailed Plan"
5. [ ] Should show budget breakdown
✅ Shows day-wise itinerary

### Tab 4: 🤖 Travel Assistant
1. [ ] Select language: English
2. [ ] Click "Where is the nearest bus stop?"
3. [ ] Should show response
4. [ ] Type "Tell me about hotels"
5. [ ] Click "📤 Send"
✅ Response appears in chat

---

## Step 7: Troubleshooting

### Issue: "ModuleNotFoundError"
```bash
# Solution:
pip install -r requirements.txt
```
Then restart Streamlit.

### Issue: Port 8501 already in use
```bash
# Solution - Run on different port:
streamlit run app.py --server.port 8502
```

### Issue: Port 5000 already in use
```bash
# Solution - Check what's using it:
netstat -ano | findstr 5000

# Kill process:
taskkill /PID <PID> /F
```

### Issue: "Connection refused" for Travel Guide
**Possible causes:**
1. [ ] Flask server not running on port 5000
2. [ ] Flask crashed - check trip_planner terminal
3. [ ] Gemini API key missing or invalid

**Solution:**
```bash
# Start Flask in separate terminal:
cd trip_planner
python app.py
```

### Issue: Slow first load
✅ **This is normal!** Models cache on first run.
- First load: 30-60 seconds
- Subsequent loads: <2 seconds

### Issue: "Gemini API Error"
```bash
# Check .env file:
cat trip_planner\.env

# Verify key is valid at:
https://ai.google.dev/
```

---

## Final Verification Checklist

### 🔧 Technical Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment activated
- [ ] All dependencies installed
- [ ] No import errors
- [ ] Streamlit version >= 1.33.0

### 🌐 Network & APIs
- [ ] Port 8501 accessible (Streamlit)
- [ ] Port 5000 accessible (Flask - if using API)
- [ ] Internet connection working
- [ ] Firewall allows localhost connections

### 📊 ML Models
- [ ] flight_model/ exists
- [ ] hotel_model/ exists
- [ ] tourism_model/ exists
- [ ] aggregator.py works

### 🎮 App Features
- [ ] Tab 1 displays expense predictions
- [ ] Tab 2 displays travel guides (with fallback)
- [ ] Tab 3 displays trip plans
- [ ] Tab 4 displays chat responses
- [ ] All tabs have unique widget keys (no conflicts)

### 📝 Documentation
- [ ] INTEGRATION_GUIDE.md exists
- [ ] QUICKSTART.md exists
- [ ] .env.example exists
- [ ] This checklist exists

---

## Performance Benchmarks

**Expected Performance:**
- Page load: 1-2 seconds
- Expense prediction: 2-5 seconds
- Travel guide (API): 3-5 seconds
- Travel guide (fallback): <1 second
- Chat response: <1 second

---

## File Verification

```bash
# Run this to verify all files:
# Windows PowerShell:
@(
    "app.py",
    "requirements.txt",
    "INTEGRATION_GUIDE.md",
    "QUICKSTART.md",
    ".env.example",
    "aggregator\aggregator.py",
    "flight_model\__init__.py",
    "hotel_model\__init__.py",
    "tourism_model\__init__.py",
    "trip_planner\app.py"
) | ForEach-Object {
    if (Test-Path $_) { Write-Host "✓ $_" }
    else { Write-Host "✗ MISSING: $_" }
}
```

---

## Support Resources

1. **Streamlit Docs:** https://docs.streamlit.io/
2. **Gemini API Docs:** https://ai.google.dev/docs
3. **Python Docs:** https://docs.python.org/3/
4. **XGBoost Docs:** https://xgboost.readthedocs.io/

---

## Quick Commands Reference

```bash
# Activate environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run Flask backend
cd trip_planner && python app.py

# Check port 8501
netstat -ano | findstr 8501

# Check port 5000
netstat -ano | findstr 5000

# Kill process on port 5000
taskkill /PID <PID> /F

# Deactivate environment
deactivate
```

---

## Success Indicators ✅

Your setup is **COMPLETE** when:
1. ✅ Streamlit opens without errors
2. ✅ All 4 tabs visible
3. ✅ Tab 1 shows expense predictions
4. ✅ Tab 2 shows travel guides
5. ✅ Tab 3 shows trip plans
6. ✅ Tab 4 shows chat responses
7. ✅ No RED ERROR messages
8. ✅ No WARNING about missing models

---

## 🎉 You're Ready!

Once all checks pass, you have:
- ✅ ML-based expense predictor
- ✅ AI travel guide explorer
- ✅ Trip planning assistant
- ✅ Multi-language chatbot

**Start by visiting:** http://localhost:8501

**Happy Trip Planning!** 🌍✈️🏖️

---

*Last Updated: June 2026*
*Version: 2.0 Integrated*
