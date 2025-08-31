import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

class AdvancedVisualizer:
    def __init__(self):
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'neutral': '#6C757D'
        }
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_model_performance_radar(self, results_df, save_path='results/model_performance_radar.png'):
        """Create radar chart for model performance comparison"""
        print("Creating model performance radar chart...")
        
        # Normalize metrics to 0-1 scale for radar chart
        metrics_cols = [col for col in results_df.columns if col not in ['Task', 'Model']]
        normalized_df = results_df.copy()
        
        for col in metrics_cols:
            if col in ['MAE', 'MSE', 'RMSE']:  # Lower is better
                normalized_df[col] = 1 - (normalized_df[col] / normalized_df[col].max())
            else:  # Higher is better
                normalized_df[col] = normalized_df[col] / normalized_df[col].max()
        
        # Create radar chart
        fig = go.Figure()
        
        for _, row in normalized_df.iterrows():
            values = [row[col] for col in metrics_cols]
            values.append(values[0])  # Close the radar chart
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics_cols + [metrics_cols[0]],
                fill='toself',
                name=f"{row['Task']} - {row['Model']}"
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Model Performance Comparison (Normalized Metrics)"
        )
        
        fig.write_html('results/model_performance_radar.html')
        print(f"Radar chart saved to results/model_performance_radar.html")
    
    def plot_feature_importance_comparison(self, models, feature_names, save_path='results/feature_importance_comparison.png'):
        """Compare feature importance across models"""
        print("Creating feature importance comparison...")
        
        importance_data = []
        
        for name, model in models.items():
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                for i, feature in enumerate(feature_names):
                    importance_data.append({
                        'Model': name,
                        'Feature': feature,
                        'Importance': importance[i] if i < len(importance) else 0
                    })
        
        if not importance_data:
            print("No feature importance data available")
            return
        
        importance_df = pd.DataFrame(importance_data)
        
        # Get top 15 features across all models
        top_features = (importance_df.groupby('Feature')['Importance']
                       .mean().sort_values(ascending=False).head(15).index)
        
        importance_df = importance_df[importance_df['Feature'].isin(top_features)]
        
        plt.figure(figsize=(15, 10))
        sns.barplot(data=importance_df, x='Importance', y='Feature', hue='Model')
        plt.title('Feature Importance Comparison Across Models')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_intervals(self, y_true, y_pred, lower_bound, upper_bound, 
                                 model_name, save_path=None):
        """Plot predictions with confidence intervals"""
        if save_path is None:
            save_path = f'results/prediction_intervals_{model_name.lower().replace(" ", "_")}.png'
        
        plt.figure(figsize=(12, 8))
        
        # Sort by true values for better visualization
        sort_idx = np.argsort(y_true)
        y_true_sorted = y_true[sort_idx]
        y_pred_sorted = y_pred[sort_idx]
        lower_sorted = lower_bound[sort_idx]
        upper_sorted = upper_bound[sort_idx]
        
        x_range = range(len(y_true_sorted))
        
        plt.plot(x_range, y_true_sorted, 'o', label='Actual', alpha=0.6, markersize=3)
        plt.plot(x_range, y_pred_sorted, 'r-', label='Predicted', alpha=0.8)
        plt.fill_between(x_range, lower_sorted, upper_sorted, alpha=0.3, label='95% Confidence Interval')
        
        plt.xlabel('Sample Index (sorted by actual value)')
        plt.ylabel('Value')
        plt.title(f'Predictions with Confidence Intervals - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_ensemble_contribution(self, ensemble_model, X_test, save_path='results/ensemble_contribution.png'):
        """Visualize individual model contributions in ensemble"""
        if not hasattr(ensemble_model, 'trained_models'):
            print("Ensemble model doesn't have individual model access")
            return
        
        contributions = {}
        for name, model in ensemble_model.trained_models.items():
            try:
                pred = model.predict(X_test)
                contributions[name] = pred
            except:
                continue
        
        if not contributions:
            return
        
        contrib_df = pd.DataFrame(contributions)
        
        plt.figure(figsize=(12, 8))
        
        # Plot individual predictions
        for i, (name, pred) in enumerate(contributions.items()):
            plt.plot(pred[:100], alpha=0.7, label=name)
        
        # Plot ensemble prediction
        if hasattr(ensemble_model, 'predict'):
            ensemble_pred = ensemble_model.predict(X_test)
            plt.plot(ensemble_pred[:100], 'k-', linewidth=2, label='Ensemble', alpha=0.9)
        
        plt.xlabel('Sample Index')
        plt.ylabel('Prediction')
        plt.title('Individual Model Contributions to Ensemble')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_series_forecast(self, df, predictions, model_name, save_path=None):
        """Plot time series forecasting results"""
        if save_path is None:
            save_path = f'results/time_series_forecast_{model_name.lower().replace(" ", "_")}.png'
        
        plt.figure(figsize=(15, 8))
        
        # Prepare time series data
        df_sorted = df.sort_values('transaction_date')
        dates = pd.to_datetime(df_sorted['transaction_date'])
        actual_sales = df_sorted['total_sales']
        
        # Plot actual vs predicted
        plt.plot(dates, actual_sales, label='Actual Sales', alpha=0.7)
        
        if len(predictions) == len(dates):
            plt.plot(dates, predictions, label=f'{model_name} Predictions', alpha=0.8)
        
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.title(f'Time Series Forecasting - {model_name}')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, df, results_summary):
        """Create interactive Plotly dashboard"""
        print("Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sales Trend', 'Model Performance', 'Customer Segments', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sales trend
        if 'transaction_date' in df.columns and 'total_sales' in df.columns:
            daily_sales = df.groupby('transaction_date')['total_sales'].sum().reset_index()
            daily_sales['transaction_date'] = pd.to_datetime(daily_sales['transaction_date'])
            
            fig.add_trace(
                go.Scatter(
                    x=daily_sales['transaction_date'],
                    y=daily_sales['total_sales'],
                    mode='lines',
                    name='Daily Sales'
                ),
                row=1, col=1
            )
        
        # Model performance (if available)
        if 'Sales Forecasting' in results_summary:
            models = list(results_summary['Sales Forecasting'].keys())
            r2_scores = [results_summary['Sales Forecasting'][m].get('R²', 0) for m in models]
            
            fig.add_trace(
                go.Bar(x=models, y=r2_scores, name='R² Scores'),
                row=1, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, title_text="Sales Analytics Dashboard")
        fig.write_html('results/interactive_dashboard.html')
        print("Interactive dashboard saved to results/interactive_dashboard.html")
    
    def plot_residual_analysis_advanced(self, y_true, y_pred, model_name):
        """Advanced residual analysis with multiple plots"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Fitted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Fitted')
        
        # 2. Normal Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot')
        
        # 3. Scale-Location plot
        standardized_residuals = np.sqrt(np.abs(residuals / np.std(residuals)))
        axes[0, 2].scatter(y_pred, standardized_residuals, alpha=0.6)
        axes[0, 2].set_xlabel('Fitted Values')
        axes[0, 2].set_ylabel('√|Standardized Residuals|')
        axes[0, 2].set_title('Scale-Location Plot')
        
        # 4. Residuals histogram
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        
        # 5. Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.6)
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        
        # 6. Residuals vs Index (time order)
        axes[1, 2].plot(residuals, alpha=0.7)
        axes[1, 2].axhline(y=0, color='red', linestyle='--')
        axes[1, 2].set_xlabel('Index')
        axes[1, 2].set_ylabel('Residuals')
        axes[1, 2].set_title('Residuals vs Index')
        
        plt.suptitle(f'Advanced Residual Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'results/advanced_residual_analysis_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

# Create global visualizer instance
visualizer = AdvancedVisualizer()