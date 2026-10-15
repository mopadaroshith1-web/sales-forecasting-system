
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="wide"
)

# Application title
st.title("📈 Sales Forecasting System")
st.write(
    "A Machine Learning application to analyze historical sales, "
    "evaluate model predictions, and display future sales forecasts."
)

# File paths
BASE_DIR = Path(__file__).parent
data_path = BASE_DIR / "sales_data.csv"
comparison_path = BASE_DIR / "actual_vs_predicted.csv"
forecast_path = BASE_DIR / "future_sales_forecast.csv"

# Load sales dataset
if not data_path.exists():
    st.error("sales_data.csv is missing. Add it to your project folder.")
    st.stop()

sales = pd.read_csv(data_path)

# Validate and clean data
if "Date" not in sales.columns or "Sales" not in sales.columns:
    st.error("The dataset must contain Date and Sales columns.")
    st.stop()

sales["Date"] = pd.to_datetime(sales["Date"], errors="coerce")
sales["Sales"] = pd.to_numeric(sales["Sales"], errors="coerce")
sales = sales.dropna(subset=["Date", "Sales"]).sort_values("Date")

if sales.empty:
    st.error("No valid sales records were found.")
    st.stop()

# Summary metrics
st.subheader("📊 Sales Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", len(sales))

col2.metric(
    "Average Monthly Sales",
    f"{sales['Sales'].mean():,.2f}"
)

highest = sales.loc[sales["Sales"].idxmax()]

col3.metric(
    "Highest Sales Month",
    highest["Date"].strftime("%B %Y"),
    f"{highest['Sales']:,.2f}"
)

# Historical sales chart
st.subheader("📈 Historical Sales Trend")

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(
    sales["Date"],
    sales["Sales"],
    marker="o",
    label="Monthly Sales"
)
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Historical Monthly Sales")
ax.grid(True, alpha=0.3)
ax.legend()
fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)

# Actual versus predicted results
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)

    st.subheader("🎯 Actual vs Predicted Sales")
    st.dataframe(comparison, width="stretch")

    if {"Actual Sales", "Predicted Sales"}.issubset(comparison.columns):
        comparison["Actual Sales"] = pd.to_numeric(
            comparison["Actual Sales"], errors="coerce"
        )
        comparison["Predicted Sales"] = pd.to_numeric(
            comparison["Predicted Sales"], errors="coerce"
        )

        comparison = comparison.dropna(
            subset=["Actual Sales", "Predicted Sales"]
        )

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(
            comparison["Actual Sales"].to_numpy(),
            marker="o",
            label="Actual Sales"
        )
        ax2.plot(
            comparison["Predicted Sales"].to_numpy(),
            marker="o",
            label="Predicted Sales"
        )
        ax2.set_xlabel("Test Record")
        ax2.set_ylabel("Sales")
        ax2.set_title("Actual vs Predicted Sales")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()

        st.pyplot(fig2)
        plt.close(fig2)
else:
    st.info(
        "Actual vs predicted results are not available yet. "
        "Run sales_forecasting.py to generate the comparison file."
    )

# Future sales forecast
st.subheader("🔮 Future Sales Forecast")

if forecast_path.exists():
    forecast = pd.read_csv(forecast_path)
    st.dataframe(forecast, width="stretch")
else:
    st.warning(
        "future_sales_forecast.csv is missing. "
        "Run sales_forecasting.py and add the generated file."
    )

# Footer
st.divider()
st.caption(
    "Machine Learning Internship Project | Python, Pandas, "
    "Matplotlib, Scikit-learn and Streamlit"
)
st.caption(
    "Demonstration using sample synthetic sales data. "
    "Forecasts are illustrative and not actual business predictions."
)
