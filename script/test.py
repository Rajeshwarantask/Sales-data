import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import joblib

def test_model():
    # Load the saved model
    model = joblib.load('results/best_sales_forecast_model.pkl')

    # Load test data
    test_data = pd.read_csv('path_to_your_test_data.csv')
    X_test = test_data.drop(columns=['target_column'])
    y_test = test_data['target_column']

    # Predict and calculate metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Performance on Test Set:")
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}")

    # Cross-validation
    cross_val_scores = cross_val_score(model, X_test, y_test, cv=5, scoring='neg_mean_squared_error')
    print("Cross-validation MSE:", cross_val_scores)
    print(f"Mean Cross-Validation MSE: {np.mean(cross_val_scores)}")

    # Testing with real-world data
    real_world_data = pd.read_csv('path_to_real_world_data.csv')
    X_real = real_world_data.drop(columns=['target_column'])
    y_real = real_world_data['target_column']
    y_real_pred = model.predict(X_real)

    # Calculate and print real-world metrics
    real_mae = mean_absolute_error(y_real, y_real_pred)
    real_mse = mean_squared_error(y_real, y_real_pred)
    real_rmse = np.sqrt(real_mse)
    real_r2 = r2_score(y_real, y_real_pred)

    print("\nPerformance on Real-World Data:")
    print(f"Real MAE: {real_mae}, Real MSE: {real_mse}, Real RMSE: {real_rmse}, Real R2: {real_r2}")
