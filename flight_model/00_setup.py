"""
00_setup.py
-----------
Run this FIRST before anything else.
Creates all necessary folders and checks that your datasets are in place.

Instructions:
  1. Download the two datasets from Kaggle (links below)
  2. Place the files in data/raw/ as described
  3. Run:  python 00_setup.py
  4. If all checks pass, proceed to:  python 01_clean_data.py
"""

import os
import sys

# ─── FOLDER STRUCTURE ─────────────────────────────────────────────────────────
FOLDERS = [
    "data/raw",
    "data/processed",
    "data/eda_charts",
    "models",
]

print("Creating folder structure...")
for folder in FOLDERS:
    os.makedirs(folder, exist_ok=True)
    print(f"  ✓  {folder}/")

# ─── DATASET CHECKLIST ────────────────────────────────────────────────────────
print("""
╔══════════════════════════════════════════════════════════════════════════╗
║              DATASETS YOU NEED TO DOWNLOAD FROM KAGGLE                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  DATASET 1  (primary — 300k rows, has days_left feature)               ║
║  URL  : kaggle.com/datasets/shubhambathwal/flight-price-prediction      ║
║  File : Clean_Dataset.csv                                               ║
║  Save to: data/raw/Clean_Dataset.csv                                    ║
║                                                                          ║
║  DATASET 2  (secondary — 10k rows, has journey date)                   ║
║  URL  : kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh     ║
║  File : Data_Train.xlsx                                                 ║
║  Save to: data/raw/Data_Train.xlsx                                      ║
║                                                                          ║
║  WHY NOT THE TWO FROM YOUR ORIGINAL LIST?                              ║
║  • yashdharme36/airfare-ml-predicting-flight-fares   → same source     ║
║    as DS1, smaller subset. DS1 is the full version. Use DS1.           ║
║  • chidinmaokonta/flight-price-prediction-dataset    → non-India data  ║
║    (mostly US routes). Not useful for Indian price prediction.         ║
║    SKIP this one.                                                       ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
""")

# ─── CHECK FILES EXIST ────────────────────────────────────────────────────────
REQUIRED_FILES = {
    "data/raw/Clean_Dataset.csv":  "Kaggle: shubhambathwal/flight-price-prediction",
    "data/raw/Data_Train.xlsx":    "Kaggle: nikhilmittal/flight-fare-prediction-mh",
}

all_ok = True
print("Checking required files...")
for path, source in REQUIRED_FILES.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✓  {path}  ({size_mb:.1f} MB)")
    else:
        print(f"  ✗  MISSING: {path}")
        print(f"     Download from: {source}")
        all_ok = False

if all_ok:
    print("""
All files found. Run the pipeline in order:

  python 01_clean_data.py          ← merge + clean datasets
  python 02_eda.py                 ← generate EDA charts
  python 03_feature_engineering.py ← build feature matrix
  python 04_train_model.py         ← train XGBoost model
  python 05_predict.py             ← test predictions in terminal
  streamlit run app.py             ← launch web UI
""")
else:
    print("\nPlease download the missing files and re-run this script.")
    sys.exit(1)