import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import numpy as np
import os

def run_seasonality_analysis(cleaned_transaction_df):
    """Enhanced seasonality analysis with Prophet and statistical tests"""
    print("\n📅 Running Enhanced Seasonality Analysis...")
    
    # Convert to pandas and prepare data
    df = cleaned_transaction_df.toPandas()
    df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    df['total_sales'] = df['quantity'] * df['unit_price']
    
    # Aggregate daily sales
    daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    daily_sales = daily_sales.sort_values('ds')
    
    # Ensure we have enough data
    if len(daily_sales) < 30:
        print("Not enough data for seasonality analysis")
        return {"Prophet": {"MAE": 0, "RMSE": 0, "R²": 0}}
    
    # Prophet model for seasonality detection
    print("Training Prophet model...")
    prophet_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'
    )
    
    # Add custom seasonalities
    prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    prophet_model.add_seasonality(name='quarterly', period=91.25, fourier_order=8)
    
    # Fit model
    prophet_model.fit(daily_sales)
    
    # Make future predictions
    future = prophet_model.make_future_dataframe(periods=30)
    forecast = prophet_model.predict(future)
    
    # Evaluate on historical data
    historical_pred = forecast[forecast['ds'].isin(daily_sales['ds'])]
    y_true = daily_sales['y'].values
    y_pred = historical_pred['yhat'].values
    
    # Calculate metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # Statistical seasonality tests
    stationarity_test = adfuller(daily_sales['y'])
    is_stationary = stationarity_test[1] < 0.05
    
    # Decomposition analysis
    if len(daily_sales) >= 365:
        decomposition = seasonal_decompose(
            daily_sales.set_index('ds')['y'], 
            model='multiplicative', 
            period=365
        )
        
        # Plot decomposition
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        decomposition.observed.plot(ax=axes[0], title='Original')
        decomposition.trend.plot(ax=axes[1], title='Trend')
        decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
        decomposition.resid.plot(ax=axes[3], title='Residual')
        plt.tight_layout()
        plt.savefig('results/seasonality_decomposition.png')
        plt.close()
    
    # Plot Prophet forecast
    fig = prophet_model.plot(forecast)
    plt.title('Sales Forecast with Seasonality')
    plt.savefig('results/prophet_forecast.png')
    plt.close()
    
    # Plot components
    fig = prophet_model.plot_components(forecast)
    plt.savefig('results/prophet_components.png')
    plt.close()
    
    # Monthly seasonality analysis
    analyze_seasonality(daily_sales)
    
    print("✅ Enhanced seasonality analysis completed")
    
    return {
        "Prophet": {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "Stationarity_p_value": stationarity_test[1],
            "Is_Stationary": is_stationary
        }
    }
def analyze_seasonality(monthly_sales_df):
    print("\n[Seasonality Analysis Started]")

    # Ensure the results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")

    # Add 'month' and 'year' features
    monthly_sales_df["month_num"] = monthly_sales_df["ds"].dt.month
    monthly_sales_df["year"] = monthly_sales_df["ds"].dt.year

    # Aggregate total sales by month number across all years
    seasonal_pattern = monthly_sales_df.groupby("month_num").agg({
        "y": ["sum", "mean", "count"]
    }).reset_index()
    seasonal_pattern.columns = ["month", "total_sales", "average_sales", "transaction_count"]

    # Save seasonal insights
    seasonal_pattern.to_csv("results/seasonality_summary.csv", index=False)

    # Plot seasonal total sales
    plt.figure(figsize=(10, 6))
    sns.barplot(data=seasonal_pattern, x="month", y="total_sales", palette="Blues_d")
    plt.title("Total Sales by Month")
    plt.xlabel("Month (1=Jan, 12=Dec)")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.savefig("results/total_sales_by_month.png")
    plt.close()

    # Plot seasonal average sales
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=seasonal_pattern, x="month", y="average_sales", marker="o", color="green")
    plt.title("Average Sales by Month")
    plt.xlabel("Month (1=Jan, 12=Dec)")
    plt.ylabel("Average Sales")
    plt.tight_layout()
    plt.savefig("results/average_sales_by_month.png")
    plt.close()

    print("[Seasonality Analysis Completed] — Files saved to 'results/'\n")
