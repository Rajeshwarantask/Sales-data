import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import os
from script.visualizations import plot_confusion_matrix_custom
from imblearn.over_sampling import SMOTE
import pyspark.sql.functions as F

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
        F.count("transaction_id").alias("frequency"),  # Use `transaction_id` from `cleaned_transaction_df`
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

    # Split into train and test sets
    from sklearn.model_selection import train_test_split
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
def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Evaluating {model}...")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1-score:", f1_score(y_test, y_pred))
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }
    return metrics
