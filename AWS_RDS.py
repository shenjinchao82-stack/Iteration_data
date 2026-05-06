from sqlalchemy import create_engine, text, VARCHAR, INT, DECIMAL, BOOLEAN, TEXT, DATE
import pandas as pd
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

engine = create_engine(os.environ['DB_URL'])

# Drop all tables in reverse dependency order to avoid FK conflicts
with engine.connect() as conn:
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
    conn.execute(text("DROP TABLE IF EXISTS benchmark_stats;"))
    conn.execute(text("DROP TABLE IF EXISTS bill_monthly_history;"))
    conn.execute(text("DROP TABLE IF EXISTS water_bills;"))
    conn.execute(text("DROP TABLE IF EXISTS users;"))
    conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
    conn.commit()

# 1. Upload users
df_users = pd.read_csv('users.csv')

df_users.to_sql(
    'users',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'UUID':       VARCHAR(36),
        'Google_ID':  VARCHAR(30),
        'state':      VARCHAR(10),
        'population': INT(),
        'suburb':     VARCHAR(100),
    }
)

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE users ADD PRIMARY KEY (UUID);"))
    conn.commit()

print('users uploaded')

#  2. Upload water_bills 
df_bills = pd.read_csv('water_bills.csv')

df_bills.to_sql(
    'water_bills',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'id':                            INT(),
        'user_id':                       VARCHAR(36),
        'state':                         VARCHAR(10),
        'suburb':                        VARCHAR(100),
        'population':                    INT(),
        'year':                          INT(),
        'month':                         INT(),
        'season':                        VARCHAR(10),
        'is_vacant':                     INT(),
        'date':                          DATE(),
        'issue_date':                    DATE(),
        'due_date':                      DATE(),
        'start_date':                    DATE(),
        'end_date':                      DATE(),
        'cost':                          DECIMAL(10, 2),
        'last_bill':                     DECIMAL(10, 2),
        'total_water_consumption':       DECIMAL(10, 2),
        'total_water_consumption_unit':  VARCHAR(5),
        'average_water_usage_l_per_day': INT(),
        'average_daily_cost':            DECIMAL(10, 2),
        'notes':                         TEXT(),
    }
)

with engine.connect() as conn:
    conn.execute(text("ALTER TABLE water_bills ADD PRIMARY KEY (id);"))
    # Link each bill to its owner in users
    conn.execute(text("""
        ALTER TABLE water_bills
        ADD CONSTRAINT fk_bills_user
        FOREIGN KEY (user_id) REFERENCES users(UUID);
    """))
    conn.commit()

print('water_bills uploaded')

#  3. Upload bill_monthly_history 
df_history = pd.read_csv('bill_monthly_history.csv')

# Replace NaN with None so MySQL stores them as NULL
df_history['usage']      = df_history['usage'].where(df_history['usage'].notna(), None)
df_history['usage_unit'] = df_history['usage_unit'].where(df_history['usage_unit'].notna(), None)
df_history['usage_kl']   = df_history['usage_kl'].where(df_history['usage_kl'].notna(), None)

df_history.to_sql(
    'bill_monthly_history',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'bill_id':                INT(),
        'user_id':                VARCHAR(36),
        'period':                 VARCHAR(20),
        'month':                  VARCHAR(15),
        'year':                   INT(),
        'start_date':             DATE(),
        'end_date':               DATE(),
        'usage':                  DECIMAL(10, 2),
        'usage_unit':             VARCHAR(10),
        'usage_kl':               DECIMAL(10, 2),
        'average_litres_per_day': INT(),
        'inferred_from_chart':    BOOLEAN(),
        'confidence':             VARCHAR(10),
    }
)

with engine.connect() as conn:
    conn.execute(text("""
        ALTER TABLE bill_monthly_history
        ADD CONSTRAINT fk_history_bill
        FOREIGN KEY (bill_id) REFERENCES water_bills(id);
    """))
    conn.execute(text("""
        ALTER TABLE bill_monthly_history
        ADD CONSTRAINT fk_history_user
        FOREIGN KEY (user_id) REFERENCES users(UUID);
    """))
    conn.commit()

print('bill_monthly_history uploaded')

#  4. Upload benchmark_stats
df_stats = pd.read_csv('benchmark_stats.csv')

df_stats.to_sql(
    'benchmark_stats',
    con=engine,
    if_exists='replace',
    index=False,
    chunksize=1000,
    method='multi',
    dtype={
        'state':            VARCHAR(10),
        'month':            INT(),
        'season':           VARCHAR(10),
        'population':       INT(),
        'count':            INT(),
        'mean_bill':        DECIMAL(10, 2),
        'std_bill':         DECIMAL(10, 2),
        'p25_bill':         DECIMAL(10, 2),
        'p50_bill':         DECIMAL(10, 2),
        'p75_bill':         DECIMAL(10, 2),
        'p90_bill':         DECIMAL(10, 2),
        'iqr_bill':         DECIMAL(10, 2),
        'cv_bill':          DECIMAL(10, 4),
        'upper_fence':      DECIMAL(10, 2),
        'lower_fence':      DECIMAL(10, 2),
        'saving_to_median': DECIMAL(10, 2),
        'mean_lpd':         DECIMAL(10, 1),
        'p25_lpd':          DECIMAL(10, 1),
        'p50_lpd':          DECIMAL(10, 1),
        'p75_lpd':          DECIMAL(10, 1),
        'p90_lpd':          DECIMAL(10, 1),
    }
)

# Composite PK: each (state, month, population) combination is unique
with engine.connect() as conn:
    conn.execute(text("""
        ALTER TABLE benchmark_stats
        ADD PRIMARY KEY (state, month, population);
    """))
    conn.commit()

print('benchmark_stats uploaded')
print('\nAll tables imported successfully!')