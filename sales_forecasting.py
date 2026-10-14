
import os
import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend so graphs save without opening windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. CREATE SAMPLE SALES DATA
# ==========================================

file_name = "sales_data.csv"

if not os.path.exists(file_name):
    np.random.seed(42)

    dates = pd.date_range(
        start="2023-01-01",
        periods=36,
        freq="MS"
    )

    month_numbers = np.arange(36)

    trend = month_numbers * 500
    seasonal = 3000 * np.sin(2 * np.pi * month_numbers / 12)
    noise = np.random.normal(0, 1000, 36)

    sales = 20000 + trend + seasonal + noise

    df_sample = pd.DataFrame({
        "Date": dates,
        "Sales": np.maximum(sales, 0).round(2)
    })

    df_sample.to_csv(file_name, index=False)
    print("Sample sales dataset created.")


# ==========================================
# 2. LOAD AND CLEAN DATA
# ==========================================

df = pd.read_csv(file_name)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

df = df.dropna(subset=["Date", "Sales"])
df = df.drop_duplicates()
df = df.sort_values("Date")

if len(df) == 0:
    raise ValueError("No valid sales records found in the CSV file.")

monthly_sales = (
    df.set_index("Date")["Sales"]
    .resample("MS")
    .sum()
)

# Keep only months that contain actual records
valid_months = df["Date"].dt.to_period("M").dt.to_timestamp()
monthly_sales = monthly_sales[monthly_sales.index.isin(valid_months)]
monthly_sales = monthly_sales.dropna()

if len(monthly_sales) < 12:
    raise ValueError("Please provide at least 12 months of sales data.")

print("\n========== SALES DATA ==========")
print(df.head())

print("\nTotal valid records:", len(df))
print("Monthly records:", len(monthly_sales))


# ==========================================
# 3. EXPLORE SALES DATA
# ==========================================

print("\n========== SALES ANALYSIS ==========")

print("Average monthly sales:", round(monthly_sales.mean(), 2))
print("Highest sales month:", monthly_sales.idxmax().strftime("%B %Y"))
print("Highest sales amount:", round(monthly_sales.max(), 2))
print("Lowest sales month:", monthly_sales.idxmin().strftime("%B %Y"))
print("Lowest sales amount:", round(monthly_sales.min(), 2))


# Save historical sales graph
plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.grid(True)
plt.tight_layout()
plt.savefig("sales_trend.png", dpi=150)
plt.close()


# ==========================================
# 4. FEATURE ENGINEERING
# ==========================================

data = pd.DataFrame({
    "Date": monthly_sales.index,
    "Sales": monthly_sales.values
})

data["Year"] = data["Date"].dt.year
data["Month"] = data["Date"].dt.month

data["Month_sin"] = np.sin(2 * np.pi * data["Month"] / 12)
data["Month_cos"] = np.cos(2 * np.pi * data["Month"] / 12)

features = ["Year", "Month_sin", "Month_cos"]

X = data[features]
y = data["Sales"]


# ==========================================
# 5. TRAIN AND COMPARE MODELS
# ==========================================

split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Model 1: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

# Model 2: Random Forest Regression
random_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

random_model.fit(X_train, y_train)

random_predictions = random_model.predict(X_test)


# ==========================================
# 6. EVALUATE AND SELECT MODEL
# ==========================================

linear_mae = mean_absolute_error(y_test, linear_predictions)
linear_rmse = np.sqrt(
    mean_squared_error(y_test, linear_predictions)
)

random_mae = mean_absolute_error(y_test, random_predictions)
random_rmse = np.sqrt(
    mean_squared_error(y_test, random_predictions)
)

print("\n========== MODEL COMPARISON ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nLinear Regression")
print("MAE:", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))

print("\nRandom Forest Regression")
print("MAE:", round(random_mae, 2))
print("RMSE:", round(random_rmse, 2))

if linear_rmse <= random_rmse:
    best_model = linear_model
    predictions = linear_predictions
    best_name = "Linear Regression"
else:
    best_model = random_model
    predictions = random_predictions
    best_name = "Random Forest Regression"

print("\nSelected model:", best_name)


# ==========================================
# 7. SAVE ACTUAL VS PREDICTED RESULTS
# ==========================================

comparison = pd.DataFrame({
    "Date": data["Date"].iloc[split:].values,
    "Actual Sales": y_test.values.round(2),
    "Predicted Sales": predictions.round(2)
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(comparison.to_string(index=False))

comparison.to_csv("actual_vs_predicted.csv", index=False)


# ==========================================
# 8. FORECAST THE NEXT THREE MONTHS
# ==========================================

last_date = data["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.offsets.MonthBegin(1),
    periods=3,
    freq="MS"
)

future = pd.DataFrame({"Date": future_dates})

future["Year"] = future["Date"].dt.year
future["Month"] = future["Date"].dt.month

future["Month_sin"] = np.sin(
    2 * np.pi * future["Month"] / 12
)

future["Month_cos"] = np.cos(
    2 * np.pi * future["Month"] / 12
)

# Retrain the selected model on all available historical data
best_model.fit(X, y)

future["Predicted Sales"] = best_model.predict(
    future[features]
).round(2)

print("\n========== FUTURE SALES FORECAST ==========")
print(future[["Date", "Predicted Sales"]].to_string(index=False))

future[["Date", "Predicted Sales"]].to_csv(
    "future_sales_forecast.csv",
    index=False
)


# ==========================================
# 9. SAVE ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    data["Date"].iloc[split:],
    y_test.values,
    marker="o",
    label="Actual Sales"
)

plt.plot(
    data["Date"].iloc[split:],
    predictions,
    marker="x",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Month")
plt.ylabel("Sales Amount")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()


# ==========================================
# 10. PROJECT COMPLETION
# ==========================================

print("\nProject completed successfully!")
print("Selected model:", best_name)

print("\nGenerated files:")
print("- sales_data.csv")
print("- actual_vs_predicted.csv")
print("- future_sales_forecast.csv")
print("- sales_trend.png")
print("- actual_vs_predicted.png")
