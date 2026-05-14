from sqlalchemy import create_engine, text, VARCHAR, INT, DECIMAL, DATE
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

engine = create_engine('aws_db_url')

# Drop all tables
with engine.connect() as conn:
    conn.execute(text("DROP TABLE IF EXISTS annual_water;"))
    conn.execute(text("DROP TABLE IF EXISTS water_price;"))
    conn.execute(text("DROP TABLE IF EXISTS water_use;"))
    conn.execute(text("DROP TABLE IF EXISTS rainfall;"))
    conn.execute(text("DROP TABLE IF EXISTS storage;"))
    conn.commit()

# 1. Upload annual_water
df_annual = pd.read_csv('Cleaned annual water data.csv')
df_annual.columns = ['State', 'Year', 'Water_Value']

df_annual.to_sql(
    'annual_water',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'State': VARCHAR(50),
        'Year':  VARCHAR(20),
        'Water_Value': DECIMAL(10, 2),
    }
)

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE annual_water ADD PRIMARY KEY (State, Year);"))
    conn.commit()

print('annual_water uploaded')

# 2. Upload water_price
df_price = pd.read_csv('Cleaned water prices.csv')
df_price.columns = ['State', 'Year', 'Price_Value']

df_price.to_sql(
    'water_price',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'State': VARCHAR(50),
        'Year':  VARCHAR(20),
        'Price_Value': DECIMAL(10, 2),
    }
)

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE water_price ADD PRIMARY KEY (State, Year);"))
    conn.commit()

print('water_price uploaded')

# 3. Upload water_use
df_use = pd.read_csv('Distributed water use.csv')
df_use.columns = ['Year', 'Area_Avg_L', 'Household_GL']

df_use.to_sql(
    'water_use',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'Year':         VARCHAR(20),
        'Area_Avg_L':   INT(),
        'Household_GL': INT(),
    }
)

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE water_use ADD PRIMARY KEY (Year);"))
    conn.commit()

print('water_use uploaded')

# 4. Upload rainfall
df_rainfall = pd.read_csv('rainfall_avg_2021_2026.csv')

# Convert date and extract year/month
df_rainfall['Date'] = pd.to_datetime(df_rainfall['Date'])
df_rainfall['Year'] = df_rainfall['Date'].dt.year
df_rainfall['Month'] = df_rainfall['Date'].dt.month

# Aggregate by state, year, month
df_rainfall = df_rainfall.groupby(['State', 'Year', 'Month'])['Avg_Rainfall_mm'].mean().reset_index()
df_rainfall.columns = ['State', 'Year', 'Month', 'Rainfall']

df_rainfall.to_sql(
    'rainfall',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'State':    VARCHAR(50),
        'Year':     INT(),
        'Month':    INT(),
        'Rainfall': DECIMAL(10, 2),
    }
)

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE rainfall ADD PRIMARY KEY (State, Year, Month);"))
    conn.commit()

print('rainfall uploaded')

# 5. Upload storage
df_storage = pd.read_csv('State_Timeseries_percent.csv')

df_storage['Observation Date'] = pd.to_datetime(df_storage['Observation Date'], format='%d/%m/%Y')

cols = ['Group Name', 'Observation Date', 'Accessible volume (GL)', 'Capacity Active (GL)']
df_storage = df_storage[cols]
df_storage.columns = ['State', 'Date', 'Acc_Vol_GL', 'Cap_Vol_GL']

df_storage['Year'] = df_storage['Date'].dt.year
df_storage['Month'] = df_storage['Date'].dt.month

df_storage = df_storage.dropna(subset=['Acc_Vol_GL'])
df_storage['Acc_Vol_GL'] = df_storage['Acc_Vol_GL'].str.replace(',', '').astype(float)
df_storage['Cap_Vol_GL'] = df_storage['Cap_Vol_GL'].str.replace(',', '').astype(float)

df_storage['Percent'] = (df_storage['Acc_Vol_GL'] / df_storage['Cap_Vol_GL'] * 100).round(2)
df_storage.to_sql(
    'storage',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'State':        VARCHAR(50),
        'Date':         DATE(),
        'Acc_Vol_GL':   INT(),
        'Cap_Vol_GL':   INT(),
        'Percent':      DECIMAL(10, 2),
        'Year':         INT(),
        'Month':        INT()
    }
)

print('storage uploaded')

print('\nAll tables imported successfully!')