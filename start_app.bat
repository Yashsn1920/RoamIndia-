@echo off
cd /d "%~dp0"
echo.
echo Starting RoamIndia...
echo Open in browser: http://localhost:8501
echo Press Ctrl+C to stop.
echo.
streamlit run app.py
