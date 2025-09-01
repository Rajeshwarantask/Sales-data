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
from lightgbm import LGBMClassifier

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

    # Sample the dataset for training (50k rows max)
    X_train_sample = X_train.sample(n=50000, random_state=42) if len(X_train) > 50000 else X_train
    y_train_sample = y_train.loc[X_train_sample.index]

    # Train models
    print("Training Logistic Regression...")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    from sklearn.ensemble import RandomForestClassifier

    # Lightweight Random Forest parameters
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_sample, y_train_sample)
    y_pred_rf = rf.predict(X_test)

    from xgboost import XGBClassifier

    # Lightweight XGBoost parameters
    params = {
        "max_depth": 5,              # Small trees
        "n_estimators": 200,         # Limited number of trees
        "learning_rate": 0.1,        # Decent speed
        "subsample": 0.8,            # Random sampling
        "colsample_bytree": 0.8,     # Random features
        "tree_method": "hist",       # Low memory usage
        "eval_metric": "logloss",    # Suitable for classification
        "random_state": 42
    }

    # Train on a smaller sample if the dataset is too large
    X_small = X_train.sample(n=50000, random_state=42) if len(X_train) > 50000 else X_train
    y_small = y_train.loc[X_small.index]

    # Train XGBoost
    print("Training XGBoost...")
    xgb = XGBClassifier(**params)
    xgb.fit(X_small, y_small)
    y_pred_xgb = xgb.predict(X_test)

    # Lightweight LightGBM parameters
    params = {
        "max_depth": 5,              # Small trees
        "n_estimators": 200,         # Limited number of trees
        "learning_rate": 0.1,        # Decent speed
        "subsample": 0.8,            # Random sampling
        "colsample_bytree": 0.8,     # Random features
        "random_state": 42
    }

    # Train LightGBM
    print("Training LightGBM...")
    lgbm = LGBMClassifier(**params)
    lgbm.fit(X_train, y_train)
    y_pred_lgbm = lgbm.predict(X_test)

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
        "LightGBM": {
            "Accuracy": accuracy_score(y_test, y_pred_lgbm),
            "Precision": precision_score(y_test, y_pred_lgbm),
            "Recall": recall_score(y_test, y_pred_lgbm),
            "F1": f1_score(y_test, y_pred_lgbm),
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
def run_churn_prediction(self, df):
    """Enhanced churn prediction with advanced models"""
    print("\n🔄 Running Enhanced Churn Prediction...")

    # Create churn target if not exists
    if 'churn' not in df.columns:
        df['churn'] = (df['days_since_last_purchase'] > 90).astype(int)

    # Prepare data
    X_train, X_test, y_train, y_test, scaler = self.prepare_modeling_data(
        df, 'churn', task='classification'
    )

    # Get classification models
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb
    import catboost as cb
    import xgboost as xgb

    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'xgboost': xgb.XGBClassifier(n_estimators=200, random_state=42),
        'lightgbm': lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1),
        'catboost': cb.CatBoostClassifier(iterations=200, random_state=42, verbose=False)
    }

    # Train and evaluate models
    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model

            metrics = self.evaluator.evaluate_classification_model(
                model, X_test, y_test, name
            )
            results[name] = metrics

        except Exception as e:
            print(f"Error with {name}: {e}")
            continue

    # Save models
    for name, model in trained_models.items():
        joblib.dump(model, f'models/{name}_churn_prediction.pkl')

    self.results_summary['Churn Prediction'] = results
    return results, trained_models
