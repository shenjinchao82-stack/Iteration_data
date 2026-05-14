import warnings
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")



# 1. Load data

df = pd.read_csv("rainfall_avg_2021_2026.csv")
coords = pd.read_csv("coords.csv")

df["Date"] = pd.to_datetime(df["Date"])

df = df.merge(coords, on="Suburb", how="left")
df = df.dropna(subset=["Latitude", "Longitude", "Avg_Rainfall_mm"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

# Encode month as cyclic features
df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

df = df.sort_values(["Suburb", "Year", "Month"]).reset_index(drop=True)


# 2. Feature engineering

# Previous month rainfall
df["Rainfall_Lag1"] = df.groupby("Suburb")["Avg_Rainfall_mm"].shift(1)

# Rolling rainfall averages based only on past data
df["Rainfall_Lag3_Mean"] = df.groupby("Suburb")["Avg_Rainfall_mm"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)

df["Rainfall_Lag6_Mean"] = df.groupby("Suburb")["Avg_Rainfall_mm"].transform(
    lambda x: x.shift(1).rolling(6, min_periods=1).mean()
)

# Rainfall from the same month in the previous year
df["Rainfall_YearAgo"] = df.groupby("Suburb")["Avg_Rainfall_mm"].shift(12)

df_train = df.dropna(subset=[
    "Rainfall_Lag1",
    "Rainfall_Lag3_Mean",
    "Rainfall_Lag6_Mean",
    "Rainfall_YearAgo"
]).copy()


# 3. Prepare training data

feature_cols = [
    "Latitude",
    "Longitude",
    "Year",
    "Month",
    "Month_sin",
    "Month_cos",
    "Rainfall_Lag1",
    "Rainfall_Lag3_Mean",
    "Rainfall_Lag6_Mean",
    "Rainfall_YearAgo"
]

X = df_train[feature_cols]
y = df_train["Avg_Rainfall_mm"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 5-fold cross validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# 4. Train models and select the best one

models = {
    "Multiple Linear Regression": {
        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        "params": {}
    },

    "LightGBM": {
        "pipeline": Pipeline([
            ("model", LGBMRegressor(
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ]),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__num_leaves": [31, 63],
            "model__max_depth": [-1, 10]
        }
    },

    "XGBoost": {
        "pipeline": Pipeline([
            ("model", XGBRegressor(
                random_state=42,
                n_jobs=-1,
                objective="reg:squarederror",
                eval_metric="mae"
            ))
        ]),
        "params": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [3, 5],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0]
        }
    },

    "Random Forest": {
        "pipeline": Pipeline([
            ("model", RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            ))
        ]),
        "params": {
            "model__n_estimators": [150, 200],
            "model__max_depth": [10, 15, 20],
            "model__min_samples_split": [2, 5]
        }
    }
}

results = []
best_model = None
best_model_name = None
best_cv_mae = float("inf")

start_time = time.time()

for name, item in models.items():
    search = GridSearchCV(
        estimator=item["pipeline"],
        param_grid=item["params"],
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    cv_mae = -search.best_score_

    y_pred = search.predict(X_test)
    y_pred = np.maximum(0, y_pred)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "CV_MAE": round(cv_mae, 4),
        "Test_MAE": round(test_mae, 4),
        "Test_RMSE": round(test_rmse, 4),
        "Test_R2": round(test_r2, 4),
        "Best_Params": search.best_params_
    })

    # Select the model with the lowest cross-validation MAE
    if cv_mae < best_cv_mae:
        best_cv_mae = cv_mae
        best_model = search.best_estimator_
        best_model_name = name

training_time = time.time() - start_time

model_results_df = pd.DataFrame(results).sort_values("CV_MAE").reset_index(drop=True)
model_results_df.to_csv("model_comparison.csv", index=False)

print("Model comparison:")
print(model_results_df[["Model", "CV_MAE", "Test_MAE", "Test_RMSE", "Test_R2"]])
print(f"\nBest model: {best_model_name}")
print(f"Best CV MAE: {best_cv_mae:.2f} mm")
print(f"Training time: {training_time:.2f} seconds")


# 5. Rolling prediction for May to October 2026

PREDICT_YEAR = 2026
OUTPUT_MONTHS = list(range(5, 11))

suburbs_info = (
    df[["State", "Suburb", "Latitude", "Longitude"]]
    .drop_duplicates()
    .set_index("Suburb")
)

# Store each suburb's historical rainfall sequence
suburb_sequences = {}

for suburb in suburbs_info.index:
    history = df[df["Suburb"] == suburb].sort_values(["Year", "Month"])

    if len(history) >= 12:
        last_year = history.iloc[-1]["Year"]
        last_month = history.iloc[-1]["Month"]

        if last_year == 2026 and last_month == 3:
            suburb_sequences[suburb] = list(history["Avg_Rainfall_mm"].values)

valid_suburbs = (
    suburbs_info[suburbs_info.index.isin(suburb_sequences.keys())]
    .reset_index()
)

all_predictions = []

for month in OUTPUT_MONTHS:
    seqs = np.array([
        suburb_sequences[suburb]
        for suburb in valid_suburbs["Suburb"]
    ])

    # Build lag features from updated historical sequences
    lag1 = seqs[:, -1]
    lag3_mean = seqs[:, -3:].mean(axis=1)
    lag6_mean = seqs[:, -6:].mean(axis=1)
    year_ago = seqs[:, -12]

    batch_df = pd.DataFrame({
        "Suburb": valid_suburbs["Suburb"].values,
        "Latitude": valid_suburbs["Latitude"].values,
        "Longitude": valid_suburbs["Longitude"].values,
        "Year": PREDICT_YEAR,
        "Month": month,
        "Month_sin": np.sin(2 * np.pi * month / 12),
        "Month_cos": np.cos(2 * np.pi * month / 12),
        "Rainfall_Lag1": lag1,
        "Rainfall_Lag3_Mean": lag3_mean,
        "Rainfall_Lag6_Mean": lag6_mean,
        "Rainfall_YearAgo": year_ago
    })

    preds = best_model.predict(batch_df[feature_cols])
    preds = np.maximum(0, np.round(preds, 2))

    # Add current predictions back into sequences for next month prediction
    for i, suburb in enumerate(valid_suburbs["Suburb"]):
        suburb_sequences[suburb].append(preds[i])

    batch_df["Predicted_Rainfall_mm"] = preds

    all_predictions.append(
        batch_df[[
            "Suburb",
            "Year",
            "Month",
            "Predicted_Rainfall_mm"
        ]]
    )


# 6. Save prediction results

results_df = pd.concat(all_predictions, ignore_index=True)

results_df = results_df.merge(
    valid_suburbs[["Suburb", "State"]],
    on="Suburb",
    how="left"
)

results_df = results_df[
    [
        "State",
        "Suburb",
        "Year",
        "Month",
        "Predicted_Rainfall_mm"
    ]
]

results_df.to_csv("rainfall_predictions_2026_May_Oct.csv", index=False)


# 7. Monthly average summary

month_names = {
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct"
}

results_df["Month_Name"] = results_df["Month"].map(month_names)

monthly_avg = results_df.groupby("Month_Name")["Predicted_Rainfall_mm"].mean()

print("\nMonthly average predicted rainfall:")

for month_name in ["May", "Jun", "Jul", "Aug", "Sep", "Oct"]:
    if month_name in monthly_avg:
        print(f"2026-{month_name}: {monthly_avg[month_name]:.2f} mm")

print("\nSaved files:")
print("model_comparison.csv")
print("rainfall_predictions_2026_May_Oct.csv")