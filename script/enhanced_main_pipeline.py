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
from script.advanced_evaluation import AdvancedEvaluator, evaluate_models  # Import evaluate_models
from script.advanced_algorithms import get_advanced_models
from script.data_preprocessing import clean_data, preprocess_data  # Replace load_data with clean_data
from script.feature_engineering import feature_engineering_pipeline as feature_engineering
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import lightgbm as lgb  # Import LightGBM
import xgboost as xgb  # Import XGBoost
from pandas.api.types import is_numeric_dtype

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
        
        # Frequency encode high-cardinality columns
        high_cardinality_cols = [col for col in df.select_dtypes(exclude=[np.number]).columns if df[col].nunique() > 50]
        for col in high_cardinality_cols:
            df = frequency_encode(df, col)
        
        return df
    
    def prepare_modeling_data(self, df, target_col, task='regression'):
        """Prepare data for modeling."""
        print("🔧 Preparing modeling data...")

        # Separate features (X) and target (y)
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Handle missing values
        print("🔍 Handling missing values...")
        X = X.fillna(0)  # Replace NaN values with 0

        # Encode categorical variables
        print("🔍 Encoding categorical variables...")
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        # Drop any remaining non-numeric columns
        from pandas.api.types import is_numeric_dtype
        non_numeric_cols = [col for col in X.columns if not is_numeric_dtype(X[col])]
        if non_numeric_cols:
            print(f"Dropping non-numeric columns: {non_numeric_cols}")
            X = X.drop(columns=non_numeric_cols)

        # Scale numeric features
        print("🔍 Scaling numeric features...")
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        # Train-test split
        print("🔍 Splitting data into train and test sets...")
        if task == 'classification':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test, scaler
    
    def run_sales_forecasting(self, df, demo_mode=False):
        """Enhanced sales forecasting with advanced models"""
        print("\n📈 Running Enhanced Sales Forecasting...")

        # Prepare data
        X_train, X_test, y_train, y_test, scaler = self.prepare_modeling_data(df, 'total_sales')

        # Use static hyperparameters in DEMO mode
        if demo_mode:
            print("⚡ Using static hyperparameters for DEMO mode")
            models = {
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
                'random_forest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            }
        else:
            # Hyperparameter optimization
            print("🔍 Optimizing hyperparameters...")
            optimized_params = self.hyperopt.optimize_all_models(X_train, y_train, 'regression')
            models = {
                'lightgbm': lgb.LGBMRegressor(**optimized_params['lightgbm']),
                'xgboost': xgb.XGBRegressor(**optimized_params['xgboost']),
                'random_forest': RandomForestRegressor(**optimized_params['random_forest'])
            }

        # Train and evaluate models
        results = {}
        trained_models = {}

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model

            # Evaluate
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            results[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mean_absolute_error(y_test, y_pred),
                'R²': r2_score(y_test, y_pred)
            }

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

        # Select only numeric features
        print("🔍 Selecting numeric features for clustering...")
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            raise ValueError("No numeric features available for clustering.")
        X = df[numeric_cols].fillna(0)  # Replace NaN values with 0

        # Scale features
        print("🔍 Scaling numeric features...")
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


def run_enhanced_pipeline(demo_mode=False):
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

        # DEMO MODE: use smaller dataset and fewer trials
        if demo_mode:
            print("🔎 DEMO mode active: sampling 10k rows for speed")
            df = df.sample(n=10000, random_state=42)  # Reduce dataset further for demo
            pipeline.hyperopt.n_trials = 5  # Limit to 5 trials per model
            pipeline.hyperopt.cv_folds = 2  # Use 2-fold cross-validation for speed
        
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

def frequency_encode(df, column):
    freq_map = df[column].value_counts(normalize=True).to_dict()
    df[column] = df[column].map(freq_map)
    return df

if __name__ == "__main__":
    pipeline, report = run_enhanced_pipeline()