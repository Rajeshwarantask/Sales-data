from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.sql import functions as F  # Import added to fix "F is not defined" errors
import matplotlib.pyplot as plt
import os
os.environ['LOKY_MAX_CPU_COUNT'] = '4'  # Set to the number of cores you want to use.

import pandas as pd
print("Current working directory:", os.getcwd())

# Import project modules
import script.data_preprocessing as data_preprocessing
import script.sales_forecasting as sales_forecasting
import script.sales_trends as sales_trends
import script.churn_prediction as churn_prediction
import script.promotion_optimization as promotion_optimization
from script import price_sensitivity_discount_effectiveness as price_sensitivity
import script.seasonality_analysis as seasonality
import script.clv_prediction as clv
from script import customer_segmentation as segmentation
from script.customer_segmentation import run_customer_segmentation
from script import visualizations
from sklearn.model_selection import train_test_split

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Initialize Spark session
spark = SparkSession.builder.appName("Sales Data Analysis").getOrCreate()
from pyspark.sql import SparkSession

# Initialize the Spark session
spark = SparkSession.builder.appName("SalesData").getOrCreate()

# Read the Excel file using pandas
pandas_df  = pd.read_csv('C:/Users/Lenovo/OneDrive/Desktop/sales big data/data/retail_data.csv')

# Convert pandas DataFrame to Spark DataFrame
data_df = spark.createDataFrame(pandas_df)

# Show the schema to verify the DataFrame structure
data_df.printSchema()

# Load data using Spark
data_path = r'C:\Users\Lenovo\OneDrive\Desktop\sales big data\data\retail_data.csv'

data_df = spark.read.option("header", "true").csv(r"C:\Users\Lenovo\OneDrive\Desktop\sales big data\data\retail_data.csv")
# Print the schema to check the column names and data types
data_df.printSchema()

# Ensure the data file exists
if not os.path.exists(data_path):
    print(f"Error: Data file '{data_path}' not found.")
    spark.stop()
    exit()

data_df = spark.read.csv(data_path, header=True, inferSchema=True)

# Debug churn counts right after loading the CSV
print("Raw churn counts:", data_df.groupBy("churn").count().show())
print("Unique churn values:", data_df.select("churn").distinct().show())
data_df.show(5)

# Extract necessary tables
# Extract necessary tables with clearly defined column selections

# Customer information DataFrame
customer_columns = [
    "customer_id", "age", "gender", "income_bracket", "loyalty_program", 
    "membership_years", "churn", "marital_status", "number_of_children", 
    "education_level", "occupation", "transactions_count", "average_spent", 
    "days_since_last_purchase", "email_subscriptions", "app_usage", 
    "website_visits", "social_media_engagement"
]
customer_df = data_df.select(*customer_columns)

# Product information DataFrame
product_columns = [
    "product_id", "product_category", "unit_price", "product_name", 
    "product_brand", "product_rating", "product_review_count", "product_stock", 
    "product_return_rate", "product_size", "product_weight", "product_color", 
    "product_material", "product_manufacture_date", "product_expiry_date", 
    "product_shelf_life"
]
product_df = data_df.select(*product_columns)

# Transaction information DataFrame
transaction_columns = [
    "transaction_id", "transaction_date", "customer_id", "product_id", 
    "quantity", "unit_price", "discount_applied", "payment_method", 
    "store_location", "transaction_hour", "day_of_week", "week_of_year", 
    "month_of_year", "avg_purchase_value", "purchase_frequency", 
    "last_purchase_date", "avg_discount_used", "preferred_store", 
    "online_purchases", "in_store_purchases", "avg_items_per_transaction", 
    "total_returned_items", "total_returned_value", "total_sales", 
    "total_discounts_received", "total_items_purchased", 
    "avg_spent_per_category", "max_single_purchase_value", 
    "min_single_purchase_value"
]
transaction_df = data_df.select(*transaction_columns)

# Store information DataFrame
store_columns = [
    "store_zip_code", "store_city", "store_state", "distance_to_store", 
    "holiday_season", "season", "weekend"
]
store_df = data_df.select(*store_columns)



# Convert categorical to numerical
customer_df = customer_df.withColumn(
    "loyalty_program",
    when(col("loyalty_program") == "Yes", 1).otherwise(0)
)

# Show data samples
print("Sample transaction data:")
transaction_df.show(5)

print("Sample product data:")
product_df.show(5)

# Preprocess data
print("Running preprocessing...")
cleaned_customer_df, cleaned_transaction_df, cleaned_product_df = data_preprocessing.clean_data(customer_df, transaction_df, product_df)

# Show cleaned samples
print("Cleaned Customer DataFrame:")
cleaned_customer_df.show(3)

print("Cleaned Transaction DataFrame:")
cleaned_transaction_df.show(3)

# Initialize a results summary dictionary
results_summary = {}

# Run Churn Prediction
print("Running Churn Prediction...")
churn_results = churn_prediction.run_churn_prediction(cleaned_customer_df, cleaned_transaction_df)
results_summary["Churn Prediction"] = churn_results

# Run Sales Forecasting
print("Running Sales Forecasting...")
sales_forecasting_results = sales_forecasting.run_forecasting(cleaned_transaction_df)
results_summary["Sales Forecasting"] = sales_forecasting_results

# Run Customer Segmentation
print("Running Customer Segmentation...")
segmentation_results = segmentation.run_customer_segmentation(cleaned_customer_df)
results_summary["Customer Segmentation"] = segmentation_results

# Run Promotion Optimization
print("Running Promotion Optimization...")
promotion_results = promotion_optimization.run_promotion_optimization(cleaned_customer_df, cleaned_transaction_df)
results_summary["Promotion Optimization"] = promotion_results

# Run Seasonality Analysis
print("Running Seasonality Analysis...")
seasonality_results = seasonality.run_seasonality_analysis(cleaned_transaction_df)
results_summary["Seasonality Analysis"] = seasonality_results

# Run Customer Lifetime Value Prediction
print("Running Customer Lifetime Value Prediction...")
clv_results = clv.predict_clv(cleaned_customer_df)
results_summary["Customer Lifetime Value Prediction"] = clv_results

# Print the results summary
print("\n📊 Model Comparison Summary:")
for task, metrics in results_summary.items():
    print(f"\n{task}:")
    for model, scores in metrics.items():
        print(f"  {model}: {scores}")


# Run visualizations (uses pandas, so re-load as needed if required)
print("Generating Visualizations...")
visualizations.plot_sales_trend()
visualizations.plot_spending_vs_transactions()
visualizations.plot_churn_rate_by_age(cleaned_customer_df.toPandas())
visualizations.plot_radar_chart()
visualizations.plot_product_demand_analysis()


# Assuming `sales_df` or `data_df` is your dataset
visualizations.plot_correlation_heatmap(data_df.toPandas())
visualizations.plot_sales_distribution(data_df)
visualizations.plot_sales_by_category(data_df, category_col='region')

# Show any pending matplotlib visuals
plt.show()

# Example: Splitting data into train/test sets


# Assuming cleaned_transaction_df is already preprocessed
sales_data = cleaned_transaction_df.select("total_sales", "transaction_date").toPandas()
sales_data["transaction_date"] = pd.to_datetime(sales_data["transaction_date"])
sales_data = sales_data.set_index("transaction_date").asfreq("D").reset_index()

X = sales_data.drop(columns=["total_sales"])
y = sales_data["total_sales"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Run sales forecasting and get trained models, predictions, and metrics
rf, lr, xgb, y_pred_lr, metrics_df = sales_forecasting.run_sales_forecasting(X_train, X_test, y_train, y_test)

# Plot feature importance for Random Forest
import matplotlib.pyplot as plt
feature_importance = rf.feature_importances_
plt.barh(X_train.columns, feature_importance)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.show()

# Residuals Plot
residuals = y_test - y_pred_lr
plt.scatter(y_test, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residual Plot")
plt.xlabel("True Values")
plt.ylabel("Residuals")
plt.show()

# Log model comparison metrics
print("Model Comparison:")
print(metrics_df)

# Stop Spark session
spark.stop()

# Log current directory
print("Current working directory:", os.getcwd())

import seaborn as sns
sns.boxplot(y=sales_data["total_sales"])
plt.title("Boxplot of Total Sales")
plt.show()

sales_data["total_sales"] = sales_data["total_sales"].clip(
    lower=sales_data["total_sales"].quantile(0.01),
    upper=sales_data["total_sales"].quantile(0.99)
)

def run_churn_prediction(cleaned_customer_df, cleaned_transaction_df):
    print("\n📦 Starting Churn Prediction...\n")

    # Add `last_purchase_date` to `cleaned_customer_df`
    last_purchase = cleaned_transaction_df.groupBy("customer_id").agg(
        F.max("transaction_date").alias("last_purchase_date")
    )
    cleaned_customer_df = cleaned_customer_df.join(last_purchase, on="customer_id", how="left")

    # Add RFM features from `cleaned_transaction_df`
    rfm = cleaned_transaction_df.groupBy("customer_id").agg(
        F.max("transaction_date").alias("last_purchase"),
        F.count("transaction_id").alias("frequency"),
        F.sum("total_sales").alias("monetary")
    )

    # Calculate recency
    max_date = cleaned_transaction_df.agg(F.max("transaction_date")).collect()[0][0]
    rfm = rfm.withColumn("recency", F.datediff(F.lit(max_date), F.col("last_purchase")))

    # Add tenure (days since first purchase)
    first_purchase = cleaned_transaction_df.groupBy("customer_id").agg(
        F.min("transaction_date").alias("first_purchase_date")
    )
    rfm = rfm.join(first_purchase, on="customer_id", how="left")
    rfm = rfm.withColumn("tenure", F.datediff(F.lit(max_date), F.col("first_purchase_date")))

    # Label churners (e.g., no purchase in the last 90 days)
    rfm = rfm.withColumn("churn", F.when(F.col("recency") > 90, 1).otherwise(0))

    # Convert to Pandas for modeling
    rfm_pd = rfm.toPandas()

    # Preprocess date columns
    if "last_purchase" in rfm_pd.columns:
        rfm_pd["last_purchase"] = (pd.to_datetime(max_date) - pd.to_datetime(rfm_pd["last_purchase"])).dt.days
    if "first_purchase_date" in rfm_pd.columns:
        rfm_pd["first_purchase_date"] = (pd.to_datetime(max_date) - pd.to_datetime(rfm_pd["first_purchase_date"])).dt.days

    # Split into train and test sets
    X = rfm_pd.drop(columns=["customer_id", "churn"])
    y = rfm_pd["churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    print("Training Logistic Regression...")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    print("Training Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    print("Training XGBoost...")
    from xgboost import XGBClassifier
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=3, random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    # Evaluate models
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    metrics = {
        "Logistic Regression": {
            "Accuracy": accuracy_score(y_test, y_pred_lr),
            "Precision": precision_score(y_test, y_pred_lr),
            "Recall": recall_score(y_test, y_pred_lr),
            "F1": f1_score(y_test, y_pred_lr),
        },
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, y_pred_rf),
            "Precision": precision_score(y_test, y_pred_rf),
            "Recall": recall_score(y_test, y_pred_rf),
            "F1": f1_score(y_test, y_pred_rf),
        },
        "XGBoost": {
            "Accuracy": accuracy_score(y_test, y_pred_xgb),
            "Precision": precision_score(y_test, y_pred_xgb),
            "Recall": recall_score(y_test, y_pred_xgb),
            "F1": f1_score(y_test, y_pred_xgb),
        },
    }

    print("\nModel Comparison (Churn Prediction):")
    for model, scores in metrics.items():
        print(f"{model}: {scores}")

    # Save the best model
    best_model = max(metrics, key=lambda x: metrics[x]["F1"])
    print(f"\n🏆 Best Model: {best_model}")

    return metrics

