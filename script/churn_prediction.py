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

    # Add RFM features
    rfm = cleaned_customer_df.groupBy("customer_id").agg(
        F.max("last_purchase_date").alias("last_purchase"),
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

    # Ensure the results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Handle class imbalance with SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Define models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = []
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Evaluate each model
    for name, model in models.items():
        try:
            model.fit(X_resampled, y_resampled)
            metrics = evaluate_classification_model(model, X_test, y_test)
            f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
            metrics["CV F1"] = f1_scores.mean()
            metrics["Model"] = name
            results.append(metrics)
        except Exception as e:
            print(f"Error fitting {name}: {e}")
            continue

    # Hyperparameter tuning (Random Forest)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20]
    }
    rf_grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring='f1', cv=3, n_jobs=-1)
    rf_grid.fit(X_resampled, y_resampled)
    best_rf = rf_grid.best_estimator_
    rf_metrics = evaluate_classification_model(best_rf, X_test, y_test)
    rf_metrics["CV F1"] = cross_val_score(best_rf, X_scaled, y, cv=cv, scoring='f1').mean()
    rf_metrics["Model"] = "Tuned Random Forest"
    results.append(rf_metrics)

    # Voting Ensemble
    ensemble_model = VotingClassifier(estimators=[ 
        ('logreg', LogisticRegression()),
        ('rf', RandomForestClassifier()),
        ('gb', GradientBoostingClassifier()),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ], voting='soft')
    ensemble_model.fit(X_resampled, y_resampled)
    ensemble_metrics = evaluate_classification_model(ensemble_model, X_test, y_test)
    ensemble_metrics["CV F1"] = cross_val_score(ensemble_model, X_scaled, y, cv=cv, scoring='f1').mean()
    ensemble_metrics["Model"] = "Voting Ensemble"
    results.append(ensemble_metrics)

    # Results summary
    results_df = pd.DataFrame(results).set_index("Model")
    print("\nModel Comparison (Churn Prediction):\n", results_df)
    results_df.to_csv("results/churn_model_comparison.csv")

    # Save best model
    best_model_name = results_df["F1 Score"].idxmax()
    best_model = (
        models.get(best_model_name)
        if best_model_name in models
        else best_rf if best_model_name == "Tuned Random Forest"
        else ensemble_model if best_model_name == "Voting Ensemble"
        else None
    )
    if best_model:
        joblib.dump(best_model, "results/best_churn_model.pkl")

        # Confusion Matrix
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix_custom(cm, best_model_name, "results/confusion_matrix_churn.png")

        # ROC Curve
        if hasattr(best_model, "predict_proba"):
            probs = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            plt.figure()
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {best_model_name}')
            plt.legend(loc='lower right')
            plt.savefig(f"results/roc_curve_{best_model_name}.png")
            plt.show()

    print("✅ Churn prediction completed.\n")

    # Return metrics for all models
    return {
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
