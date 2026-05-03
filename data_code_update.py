import xarray as xr
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Hide warnings to keep the run clean
warnings.filterwarnings('ignore')

# PART 1: Spatial & Data Processing


# 1. Load and prepare rainfall data
nc = xr.open_mfdataset('./rain/*.nc', combine='by_coords', chunks='auto', data_vars='minimal')
rain_df = nc.to_dataframe().dropna().reset_index()

pts = gpd.points_from_xy(rain_df.lon, rain_df.lat)
rain_geo = gpd.GeoDataFrame(rain_df, geometry=pts, crs="EPSG:4326")

# 2. Load and prepare suburb boundaries
suburb_geo = gpd.read_file('SA2_2021_AUST_SHP_GDA2020/SA2_2021_AUST_GDA2020.shp')

# 3. Match coordinates using meter-based distance mapping
rain_geo = rain_geo.to_crs("EPSG:3577")
suburb_geo = suburb_geo.to_crs("EPSG:3577")

merged = gpd.sjoin_nearest(suburb_geo, rain_geo, how="inner", max_distance=10000, distance_col="dist_m")
merged.to_csv('rainfall_2021_2026.csv', index=False)

# 4. Calculate monthly averages and clean up column names
groups = ['STE_NAME21', 'GCC_NAME21', 'SA2_NAME21', 'time']
avg_rain = merged.groupby(groups)['monthly_rain'].mean().reset_index()

avg_rain = avg_rain.rename(columns={
    'STE_NAME21': 'State',
    'GCC_NAME21': 'City_Region',
    'SA2_NAME21': 'Suburb',
    'time': 'Date',
    'monthly_rain': 'Rain_mm'
})

# 5. Format dates and numbers, then save
avg_rain['Rain_mm'] = avg_rain['Rain_mm'].round(2)
avg_rain['Date'] = pd.to_datetime(avg_rain['Date']).dt.date
avg_rain.to_csv('rainfall_avg_2021_2026.csv', index=False)

# PART 2: Machine Learning & Prediction

# 1. Load the data we just created and merge coordinates
rain = pd.read_csv('rainfall_avg_2021_2026.csv')
locs = pd.read_csv('coords.csv')

rain['Date'] = pd.to_datetime(rain['Date'])
rain = rain.merge(locs, on='Suburb', how='left').dropna(subset=['Latitude', 'Longitude'])

# 2. Build time features
rain['Year'] = rain['Date'].dt.year
rain['Month'] = rain['Date'].dt.month
rain['Sin'] = np.sin(2 * np.pi * rain['Month'] / 12)
rain['Cos'] = np.cos(2 * np.pi * rain['Month'] / 12)

rain = rain.sort_values(['Suburb', 'Year', 'Month']).reset_index(drop=True)

rain[['State', 'Suburb', 'Latitude', 'Longitude']] \
    .drop_duplicates(subset=['Suburb']) \
    .dropna() \
    .to_csv('suburb_coords.csv', index=False)

# 3. Build historical features
by_suburb = rain.groupby('Suburb')['Rain_mm']

rain['Lag1'] = by_suburb.shift(1)
rain['Lag3'] = by_suburb.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
rain['Lag6'] = by_suburb.transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
rain['Lag12'] = by_suburb.shift(12)

train_data = rain.dropna(subset=['Lag1', 'Lag12'])

# 4. Train the Random Forest Model
features = ['Latitude', 'Longitude', 'Year', 'Month', 'Sin', 'Cos', 'Lag1', 'Lag3', 'Lag6', 'Lag12']

X = train_data[features]
y = train_data['Rain_mm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

steps = [
    ('scale', StandardScaler()),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42, n_jobs=-1))
]

model = Pipeline(steps)
model.fit(X_train, y_train)

# 5. Set up for future predictions
tgt_yr = 2026
tgt_mos = [5, 6, 7, 8, 9, 10]

suburb_locs = rain[['Suburb', 'State', 'Latitude', 'Longitude']].drop_duplicates().set_index('Suburb')
history = {}

# Gather the recent history for valid suburbs
for sub in suburb_locs.index:
    sub_data = rain[rain['Suburb'] == sub].sort_values(['Year', 'Month'])
    yr = sub_data.iloc[-1]['Year']
    mo = sub_data.iloc[-1]['Month']
    
    if yr == 2026 and mo == 4:
        history[sub] = list(sub_data['Rain_mm'].values)

active = suburb_locs[suburb_locs.index.isin(history.keys())].reset_index()
all_preds = []

# 6. Loop through months and predict sequentially
for m in tgt_mos:
    m_sin = np.sin(2 * np.pi * m / 12)
    m_cos = np.cos(2 * np.pi * m / 12)
    
    m_data = []
    
    # Calculate latest lags for each suburb
    for _, row in active.iterrows():
        sub = row['Suburb']
        seq = history[sub]
        
        m_data.append({
            'Suburb': sub,
            'State': row['State'],
            'Latitude': row['Latitude'],
            'Longitude': row['Longitude'],
            'Year': tgt_yr,
            'Month': m,
            'Sin': m_sin,
            'Cos': m_cos,
            'Lag1': seq[-1],
            'Lag3': sum(seq[-3:]) / len(seq[-3:]),
            'Lag6': sum(seq[-6:]) / len(seq[-6:]),
            'Lag12': seq[-12]
        })
        
    m_df = pd.DataFrame(m_data)
    
    # Run the prediction
    preds = model.predict(m_df[features])
    preds = np.maximum(0, np.round(preds, 2))
    
    m_df['Pred_Rain'] = preds
    
    # Add new prediction back into the sequence
    for i, sub in enumerate(active['Suburb']):
        history[sub].append(preds[i])
        
    cols = ['State', 'Suburb', 'Year', 'Month', 'Pred_Rain']
    all_preds.append(m_df[cols])

# 7. Export final predictions
final_preds = pd.concat(all_preds, ignore_index=True)
final_preds.to_csv('rainfall_predictions_2026_May_Oct.csv', index=False)
