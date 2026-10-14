# Sales Forecasting System Using Machine Learning

## Project Overview
The Sales Forecasting System analyzes historical monthly sales data and uses machine learning to predict future sales. It compares Linear Regression and Random Forest Regression, evaluates their performance, and selects the model with the lower Root Mean Squared Error (RMSE).

> **Note:** This project currently uses synthetic sample data for demonstration. Forecasts are not based on real company sales.

## Objectives
- Analyze historical sales trends.
- Clean and prepare monthly sales data.
- Train and compare machine learning models.
- Evaluate predictions using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
- Forecast sales for upcoming months.
- Save results and charts for review.

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Files

| File | Description |
|---|---|
| `sales_forecasting.py` | Main program for analysis, model training, evaluation, and forecasting |
| `sales_data.csv` | Sample monthly sales dataset |
| `actual_vs_predicted.csv` | Actual sales compared with model predictions |
| `future_sales_forecast.csv` | Forecast sales for the next three months after the dataset |
| `sales_trend.png` | Chart showing historical sales trends |
| `actual_vs_predicted.png` | Chart comparing actual and predicted sales |

## Model Evaluation

The current run used 36 monthly records, with 28 records for training and 8 for testing.

| Model | MAE | RMSE |
|---|---:|---:|
| Linear Regression | 1,155.70 | 1,551.11 |
| Random Forest Regression | 1,405.87 | 1,789.52 |

**Selected model: Linear Regression**, because it achieved the lower RMSE on the test data.

- **MAE** measures the average absolute difference between actual and predicted sales.
- **RMSE** measures prediction error while penalizing larger errors more strongly.

These scores are from the current synthetic dataset and may change if the data or code changes.

## Forecast Results

The current run produced these forecasts:

| Month | Predicted Sales |
|---|---:|
| January 2026 | 39,287.78 |
| February 2026 | 39,885.60 |
| March 2026 | 40,540.29 |

These dates follow the sample data, which ends in December 2025. They are demonstration forecasts, not current forecasts for a real business.

## Installation and Setup

### 1. Install Python
Install Python from [python.org](https://www.python.org/downloads/) if it is not already installed.

### 2. Download the project
Clone this repository or download it as a ZIP file from GitHub.

### 3. Install dependencies
Open a terminal in the project folder and run:

```bash
python -m pip install pandas numpy matplotlib scikit-learn
```

### 4. Run the program

```bash
python sales_forecasting.py
```

The program prints the analysis and model evaluation results and saves the CSV output files and PNG charts in the project folder.

## Workflow
1. Load and clean the sales dataset.
2. Analyze monthly sales.
3. Create time-based features.
4. Split the data chronologically into training and testing sets.
5. Train Linear Regression and Random Forest Regression.
6. Compare model performance using MAE and RMSE.
7. Select the model with the lower RMSE.
8. Save actual-versus-predicted results.
9. Generate and save a three-month forecast.
10. Save charts for visualization.

## Future Improvements
- Use real business sales data.
- Add more explanatory features, such as promotions, prices, holidays, and product categories.
- Evaluate performance over multiple time-based test periods.
- Add a simple web interface for interactive forecasting.
- Retrain the model as new sales data becomes available.

## Author
Created as a machine learning / data science project for internship learning and practice.
