"""
Enhanced Sales Big Data Analysis Pipeline
Integrates advanced ML algorithms, feature engineering, and ensemble methods
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Add script directory to path
sys.path.append('script')

# Import enhanced modules
from script.external_data_integration import ExternalDataIntegrator
from script.advanced_feature_engineering import AdvancedFeatureEngineer
from script.hyperparameter_optimization import HyperparameterOptimizer
from script.ensemble_learning import create_ensemble_models, save_ensemble_models
from script.advanced_evaluation import AdvancedEvaluator
from script.advanced_algorithms import get_advanced_models
from script.data_preprocessing import clean_data
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt

class EnhancedSalesPipeline:
    def __init__(self):
        self.external_integrator = ExternalDataIntegrator()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.hyperopt = HyperparameterOptimizer(n_trials=50)
        self.evaluator = AdvancedEvaluator()
        self.results_summary = {}
        
        # Ensure directories exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare data with external sources"""
        print("🔄 Loading and preparing data...")
        
        # Load main dataset
        df = pd.read_csv(data_path)
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        # Integrate external data
        df = self.external_integrator.get_holiday_features(df)
        df = self.external_integrator.get_economic_indicators(df)
        df = self.external_integrator.get_weather_features(df)
        df = self.external_integrator.add_competitor_data(df)
        
        # Advanced feature engineering
        df = self.feature_engineer.engineer_all_features(df)
        
        print(f"After feature engineering: {len(df.columns)} features")
        return df
    
    def prepare_modeling_data(self, df, target_col, task_type='regression'):
        """Prepare data for modeling with proper preprocessing"""
        print("🔧 Preparing modeling data...")
        
        # Remove rows with missing target
        df = df.dropna(subset=[target_col])
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [
            target_col, 'transaction_date', 'customer_id', 'product_id', 'transaction_id'
        ]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle missing values in features
        X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
        
        # Feature selection
        if len(feature_cols) > 100:
            X_selected, selected_features, selector = self.feature_engineer.select_features(
                X, y, method='mutual_info', k=min(50, len(feature_cols))
            )
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            print(f"Selected {len(selected_features)} most important features")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, 
            stratify=y if task_type == 'classification' else None
        )
        
        # Handle class imbalance for classification
        if task_type == 'classification':
            print("Applying SMOTE for class imbalance...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        return X_train, X_test, y_train, y_test, scaler
    
    def run_sales_forecasting(self, df):
        """Enhanced sales forecasting with advanced models"""
        print("\n📈 Running Enhanced Sales Forecasting...")
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = self.prepare_modeling_data(
            df, 'total_sales', 'regression'
        )
        
        # Hyperparameter optimization
        print("🔍 Optimizing hyperparameters...")
        optimized_params = self.hyperopt.optimize_all_models(X_train, y_train, 'regression')
        
        # Get optimized models
        models = self.hyperopt.get_optimized_models('regression')
        
        # Add advanced models
        advanced_models = get_advanced_models()
        models.update(advanced_models)
        
        # Train and evaluate models
        results = {}
        trained_models = {}
        
        for name, model in models.items():
            print(f"Training {name}...")
            try:
                model.fit(X_train, y_train)
                trained_models[name] = model
                
                # Evaluate
                metrics = self.evaluator.evaluate_regression_model(
                    model, X_test, y_test, name
                )
                results[name] = metrics
                
                # Plot learning curves for key models
                if name in ['random_forest', 'xgboost', 'lightgbm']:
                    self.evaluator.plot_learning_curves(model, X_train, y_train, name)
                
                # Plot residuals
                y_pred = model.predict(X_test)
                self.evaluator.plot_residuals(y_test, y_pred, name)
                
                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self.evaluator.plot_feature_importance(
                        model, X_train.columns, name
                    )
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                continue
        
        # Ensemble learning
        print("🤖 Training ensemble models...")
        ensemble_models = create_ensemble_models()
        
        for name, ensemble in ensemble_models.items():
            print(f"Training {name} ensemble...")
            try:
                ensemble.fit(X_train, y_train)
                trained_models[f'ensemble_{name}'] = ensemble
                
                metrics = self.evaluator.evaluate_regression_model(
                    ensemble, X_test, y_test, f'Ensemble_{name}'
                )
                results[f'ensemble_{name}'] = metrics
                
            except Exception as e:
                print(f"Error with {name} ensemble: {e}")
        
        # Save models
        save_ensemble_models(ensemble_models)
        for name, model in trained_models.items():
            joblib.dump(model, f'models/{name}_sales_forecasting.pkl')
        
        self.results_summary['Sales Forecasting'] = results
        return results, trained_models
    
    def run_churn_prediction(self, df):
        """Enhanced churn prediction with advanced models"""
        print("\n🔄 Running Enhanced Churn Prediction...")
        
        # Create churn target if not exists
        if 'churn' not in df.columns:
            df['churn'] = (df['days_since_last_purchase'] > 90).astype(int)
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = self.prepare_modeling_data(
            df, 'churn', 'classification'
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
    
    def run_customer_segmentation(self, df):
        """Enhanced customer segmentation"""
        print("\n👥 Running Enhanced Customer Segmentation...")
        
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        from sklearn.mixture import GaussianMixture
        from sklearn.metrics import silhouette_score, calinski_harabasz_score
        
        # Prepare customer features
        customer_features = [
            'age', 'income_bracket', 'membership_years', 'customer_total_sales_sum',
            'customer_total_sales_mean', 'customer_transactions_count', 'app_usage',
            'website_visits', 'social_media_engagement'
        ]
        
        available_features = [f for f in customer_features if f in df.columns]
        X = df[available_features].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Try different clustering algorithms
        clustering_models = {
            'kmeans_3': KMeans(n_clusters=3, random_state=42),
            'kmeans_4': KMeans(n_clusters=4, random_state=42),
            'kmeans_5': KMeans(n_clusters=5, random_state=42),
            'gaussian_mixture': GaussianMixture(n_components=4, random_state=42),
            'agglomerative': AgglomerativeClustering(n_clusters=4)
        }
        
        results = {}
        for name, model in clustering_models.items():
            try:
                labels = model.fit_predict(X_scaled)
                
                # Calculate metrics
                silhouette = silhouette_score(X_scaled, labels)
                calinski = calinski_harabasz_score(X_scaled, labels)
                
                results[name] = {
                    'Silhouette Score': silhouette,
                    'Calinski-Harabasz Score': calinski,
                    'N Clusters': len(np.unique(labels))
                }
                
                # Save cluster labels
                df[f'cluster_{name}'] = labels
                
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        self.results_summary['Customer Segmentation'] = results
        return results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n📊 Generating Comprehensive Report...")
        
        # Create summary DataFrame
        all_results = []
        for task, models in self.results_summary.items():
            for model, metrics in models.items():
                row = {'Task': task, 'Model': model}
                row.update(metrics)
                all_results.append(row)
        
        summary_df = pd.DataFrame(all_results)
        
        # Save comprehensive report
        summary_df.to_csv('results/comprehensive_model_report.csv', index=False)
        
        # Create summary visualization
        self._create_summary_visualization(summary_df)
        
        print("📋 Comprehensive report generated!")
        return summary_df
    
    def _create_summary_visualization(self, summary_df):
        """Create summary visualization dashboard"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # Sales forecasting results
        sales_results = summary_df[summary_df['Task'] == 'Sales Forecasting']
        if not sales_results.empty and 'R²' in sales_results.columns:
            sales_results.plot(x='Model', y='R²', kind='bar', ax=axes[0, 0], color='lightblue')
            axes[0, 0].set_title('Sales Forecasting - R² Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Churn prediction results
        churn_results = summary_df[summary_df['Task'] == 'Churn Prediction']
        if not churn_results.empty and 'F1' in churn_results.columns:
            churn_results.plot(x='Model', y='F1', kind='bar', ax=axes[0, 1], color='lightcoral')
            axes[0, 1].set_title('Churn Prediction - F1 Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Customer segmentation results
        seg_results = summary_df[summary_df['Task'] == 'Customer Segmentation']
        if not seg_results.empty and 'Silhouette Score' in seg_results.columns:
            seg_results.plot(x='Model', y='Silhouette Score', kind='bar', ax=axes[1, 0], color='lightgreen')
            axes[1, 0].set_title('Customer Segmentation - Silhouette Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall performance heatmap
        if len(summary_df) > 0:
            numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                pivot_data = summary_df.pivot_table(
                    values=numeric_cols[0], index='Task', columns='Model', aggfunc='mean'
                )
                sns.heatmap(pivot_data, annot=True, cmap='YlOrRd', ax=axes[1, 1])
                axes[1, 1].set_title('Performance Heatmap')
        
        plt.tight_layout()
        plt.savefig('results/comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()

def run_enhanced_pipeline():
    """Run the complete enhanced pipeline"""
    pipeline = EnhancedSalesPipeline()
    
    # Data path (update this to your actual data path)
    data_path = 'data/retail_data.csv'
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please ensure the data file exists at the specified path.")
        return None, None
    
    try:
        # Load and prepare data
        df = pipeline.load_and_prepare_data(data_path)
        
        # Run sales forecasting
        sales_results, sales_models = pipeline.run_sales_forecasting(df)
        
        # Run churn prediction
        churn_results, churn_models = pipeline.run_churn_prediction(df)
        
        # Run customer segmentation
        segmentation_results = pipeline.run_customer_segmentation(df)
        
        # Generate comprehensive report
        final_report = pipeline.generate_comprehensive_report()
        
        print("\n🎉 Enhanced pipeline completed successfully!")
        print("\n📊 Final Results Summary:")
        print(final_report.to_string(index=False))
        
        # Save final processed dataset
        df.to_csv('results/final_processed_dataset.csv', index=False)
        
        return pipeline, final_report
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    pipeline, report = run_enhanced_pipeline()