import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from prophet import Prophet
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
import joblib
from pyspark.sql import functions as F
from sklearn.preprocessing import StandardScaler

# Ensure the results directory exists
if not os.path.exists("results"):
    os.makedirs("results")

def evaluate_model(y_true, y_pred, model_name):
    """Evaluate a model and return performance metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    }

# Hyperparameter tuning
def hyperparameter_tuning(model, X_train, y_train):
    if isinstance(model, RandomForestRegressor):
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    elif isinstance(model, XGBRegressor):
        param_grid = {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    else:
        return model

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters for {model.__class__.__name__}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Run forecasting process
def run_forecasting(cleaned_transaction_df):
    print("\n📦 Starting Sales Forecasting...\n")

    # Ensure consistent schema
    cleaned_transaction_df = cleaned_transaction_df.withColumn("transaction_date", F.to_date("transaction_date", "yyyy-MM-dd"))
    cleaned_transaction_df = cleaned_transaction_df.withColumn("total_sales", cleaned_transaction_df["total_sales"].cast("double"))

    # Aggregate daily sales
    sales_daily = cleaned_transaction_df.groupBy("transaction_date").agg(F.sum("total_sales").alias("y"))
    sales_daily = sales_daily.withColumnRenamed("transaction_date", "ds")

    # Convert to Pandas
    sales_daily_pd = sales_daily.toPandas()
    sales_daily_pd["ds"] = pd.to_datetime(sales_daily_pd["ds"])

    # Add time-based features
    sales_daily_pd["dayofweek"] = sales_daily_pd["ds"].dt.dayofweek
    sales_daily_pd["month"] = sales_daily_pd["ds"].dt.month
    sales_daily_pd["year"] = sales_daily_pd["ds"].dt.year
    sales_daily_pd["is_weekend"] = (sales_daily_pd["ds"].dt.dayofweek >= 5).astype(int)

    # Add lag features
    sales_daily_pd["lag_1"] = sales_daily_pd["y"].shift(1)
    sales_daily_pd["lag_7"] = sales_daily_pd["y"].shift(7)

    # Add rolling window statistics
    sales_daily_pd["rolling_mean_7"] = sales_daily_pd["y"].shift(1).rolling(7).mean()
    sales_daily_pd["rolling_std_7"] = sales_daily_pd["y"].shift(1).rolling(7).std()

    # Drop rows with NaN values (caused by lag/rolling features)
    sales_daily_pd = sales_daily_pd.dropna()

    # Split into train and test sets
    from sklearn.model_selection import train_test_split
    X = sales_daily_pd.drop(columns=["y", "ds"])
    y = sales_daily_pd["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train models
    print("Training Linear Regression...")
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print("Training Random Forest...")
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("Training XGBoost...")
    from xgboost import XGBRegressor
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # Evaluate models
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np

    metrics = {
        "Linear Regression": {
            "MAE": mean_absolute_error(y_test, y_pred_lr),
            "MSE": mean_squared_error(y_test, y_pred_lr),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),  # Manually calculate RMSE
            "R2": r2_score(y_test, y_pred_lr),
        },
        "Random Forest": {
            "MAE": mean_absolute_error(y_test, y_pred_rf),
            "MSE": mean_squared_error(y_test, y_pred_rf),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),  # Manually calculate RMSE
            "R2": r2_score(y_test, y_pred_rf),
        },
        "XGBoost": {
            "MAE": mean_absolute_error(y_test, y_pred_xgb),
            "MSE": mean_squared_error(y_test, y_pred_xgb),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_xgb)),  # Manually calculate RMSE
            "R2": r2_score(y_test, y_pred_xgb),
        },
    }

    print("\nModel Comparison (Sales Forecasting):")
    for model, scores in metrics.items():
        print(f"{model}: {scores}")

    # Save the best model
    best_model = max(metrics, key=lambda x: metrics[x]["R2"])
    print(f"\n🏆 Best Model: {best_model}")

    # Return metrics for all models
    return {
        "Linear Regression": {
            "MAE": mean_absolute_error(y_test, y_pred_lr),
            "MSE": mean_squared_error(y_test, y_pred_lr),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            "R2": r2_score(y_test, y_pred_lr),
        },
        "Random Forest": {
            "MAE": mean_absolute_error(y_test, y_pred_rf),
            "MSE": mean_squared_error(y_test, y_pred_rf),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            "R2": r2_score(y_test, y_pred_rf),
        },
        "XGBoost": {
            "MAE": mean_absolute_error(y_test, y_pred_xgb),
            "MSE": mean_squared_error(y_test, y_pred_xgb),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            "R2": r2_score(y_test, y_pred_xgb),
        },
    }

def run_sales_forecasting(X_train, X_test, y_train, y_test):
    print("Missing values in X_train:", X_train.isnull().sum())
    print("Missing values in y_train:", y_train.isnull().sum())

    # Fill missing values with column means
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    # Train Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Train XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # Evaluate models and return trained models and predictions
    metrics_df = pd.DataFrame([
        evaluate_model(y_test, y_pred_lr, "Linear Regression"),
        evaluate_model(y_test, y_pred_rf, "Random Forest"),
        evaluate_model(y_test, y_pred_xgb, "XGBoost")
    ])
    return rf, lr, xgb, y_pred_lr, metrics_df
