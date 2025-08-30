from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
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

# Run predictive models
print("Running Sales Forecasting...")
sales_forecasting.run_forecasting(cleaned_transaction_df)

print("Running Sales sales trend...")
sales_trends.run_sales_trends_analysis(cleaned_transaction_df)

print("Running Churn Prediction...")
churn_prediction.run_churn_prediction(cleaned_customer_df)

print("Running Promotion Optimization...")
promotion_optimization.run_promotion_optimization(cleaned_customer_df, cleaned_transaction_df)

print("Running Customer Segmentation...")
segmentation.run_customer_segmentation(cleaned_customer_df)

print("Running Seasonality Analysis...")
seasonality.run_seasonality_analysis(cleaned_transaction_df)

print("Running Customer Lifetime Value Prediction...")
clv.predict_clv(cleaned_customer_df)

print("Running Price Sensitivity and Discount Effectiveness Analysis...")
price_sensitivity.analyze_price_sensitivity_and_discount(cleaned_transaction_df)


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

