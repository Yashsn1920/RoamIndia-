import os, sys, shutil, json
import pandas as pd

base = r'c:\expensemodel\india_trip_predictor\data\raw'

# 1. Clean_Dataset
if os.path.exists(os.path.join(base, 'Clean_Dataset.csv', 'Clean_Dataset.csv')):
    shutil.move(os.path.join(base, 'Clean_Dataset.csv', 'Clean_Dataset.csv'), os.path.join(base, 'Clean_Dataset_tmp.csv'))
    shutil.rmtree(os.path.join(base, 'Clean_Dataset.csv'))
    shutil.move(os.path.join(base, 'Clean_Dataset_tmp.csv'), os.path.join(base, 'Clean_Dataset.csv'))

# 2. Data_Train
if os.path.exists(os.path.join(base, 'Data_Train.xlsx', 'Data_Train.xlsx')):
    shutil.move(os.path.join(base, 'Data_Train.xlsx', 'Data_Train.xlsx'), os.path.join(base, 'Data_Train_tmp.xlsx'))
    shutil.rmtree(os.path.join(base, 'Data_Train.xlsx'))
    shutil.move(os.path.join(base, 'Data_Train_tmp.xlsx'), os.path.join(base, 'Data_Train.xlsx'))

# 3. Google Hotels
if os.path.exists(os.path.join(base, 'google_hotels.csv', 'google_hotel_data_clean_v2.csv')):
    shutil.move(os.path.join(base, 'google_hotels.csv', 'google_hotel_data_clean_v2.csv'), os.path.join(base, 'google_hotels_tmp.csv'))
    shutil.rmtree(os.path.join(base, 'google_hotels.csv'))
    shutil.move(os.path.join(base, 'google_hotels_tmp.csv'), os.path.join(base, 'google_hotels.csv'))

# 4. PromptCloud Hotels
if os.path.exists(os.path.join(base, 'promptcloud_hotels.csv', 'makemytrip_com-travel_sample.csv')):
    shutil.move(os.path.join(base, 'promptcloud_hotels.csv', 'makemytrip_com-travel_sample.csv'), os.path.join(base, 'promptcloud_hotels_tmp.csv'))
    shutil.rmtree(os.path.join(base, 'promptcloud_hotels.csv'))
    shutil.move(os.path.join(base, 'promptcloud_hotels_tmp.csv'), os.path.join(base, 'promptcloud_hotels.csv'))

# 5. Tourist spots
if os.path.exists(os.path.join(base, 'india_tourist_spots.csv', 'Data', 'Tourist_Spots.csv')):
    shutil.move(os.path.join(base, 'india_tourist_spots.csv', 'Data', 'Tourist_Spots.csv'), os.path.join(base, 'india_tourist_spots_tmp.csv'))
    shutil.rmtree(os.path.join(base, 'india_tourist_spots.csv'))
    shutil.move(os.path.join(base, 'india_tourist_spots_tmp.csv'), os.path.join(base, 'india_tourist_spots.csv'))

# 6. MMT Hotels (Combine)
mmt_dir = os.path.join(base, 'mmt_hotels.csv')
if os.path.isdir(mmt_dir):
    dfs = []
    for f in os.listdir(mmt_dir):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(mmt_dir, f))
            # Some city files might not have a city column, add it based on filename
            if 'city' not in df.columns and 'City' not in df.columns:
                df['city'] = f.split('.')[0].title()
            dfs.append(df)
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(os.path.join(base, 'mmt_hotels_tmp.csv'), index=False)
        shutil.rmtree(mmt_dir)
        shutil.move(os.path.join(base, 'mmt_hotels_tmp.csv'), mmt_dir) # mmt_dir is a string representing the target file path now

# 7. Indian Tourism JSON to CSV
json_dir = os.path.join(base, 'indian_tourism.csv')
if os.path.isdir(json_dir):
    json_path = os.path.join(json_dir, 'india_tourism_dataset.json')
    if os.path.exists(json_path):
        df = pd.read_json(json_path)
        df.to_csv(os.path.join(base, 'indian_tourism_tmp.csv'), index=False)
        shutil.rmtree(json_dir)
        shutil.move(os.path.join(base, 'indian_tourism_tmp.csv'), json_dir)

print('Cleanup completed.')
