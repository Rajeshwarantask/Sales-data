import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.model_selection import learning_curve, validation_curve
import os

class AdvancedEvaluator:
    def __init__(self):
        self.results = {}
        
    def evaluate_regression_model(self, model, X_test, y_test, model_name):
        """Comprehensive regression model evaluation"""
        print(f"Evaluating {model_name}...")
        
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'MAE': mean_absolute_error(y_test, y_pred),
                'MSE': mean_squared_error(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'R²': r2_score(y_test, y_pred),
                'MAPE': self._calculate_mape(y_test, y_pred),
                'Max_Error': np.max(np.abs(y_test - y_pred)),
                'Mean_Residual': np.mean(y_test - y_pred),
                'Std_Residual': np.std(y_test - y_pred)
            }
            
            self.results[model_name] = metrics
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return {}
    
    def evaluate_classification_model(self, model, X_test, y_test, model_name):
        """Comprehensive classification model evaluation"""
        print(f"Evaluating {model_name}...")
        
        try:
            y_pred = model.predict(X_test)
            y_pred_proba = None
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'F1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                'Specificity': self._calculate_specificity(y_test, y_pred),
                'Balanced_Accuracy': self._calculate_balanced_accuracy(y_test, y_pred)
            }
            
            if y_pred_proba is not None:
                metrics['ROC_AUC'] = roc_auc_score(y_test, y_pred_proba)
            
            self.results[model_name] = metrics
            return metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            return {}
    
    def _calculate_mape(self, y_true, y_pred):
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    def _calculate_specificity(self, y_true, y_pred):
        """Calculate specificity (true negative rate)"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            return tn / (tn + fp) if (tn + fp) > 0 else 0
        return 0
    
    def _calculate_balanced_accuracy(self, y_true, y_pred):
        """Calculate balanced accuracy"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return (sensitivity + specificity) / 2
        return accuracy_score(y_true, y_pred)
    
    def plot_learning_curves(self, model, X, y, model_name, cv=5):
        """Plot learning curves to diagnose bias/variance"""
        print(f"Plotting learning curves for {model_name}...")
        
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='neg_mean_squared_error'
            )
            
            train_mean = -train_scores.mean(axis=1)
            train_std = train_scores.std(axis=1)
            val_mean = -val_scores.mean(axis=1)
            val_std = val_scores.std(axis=1)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            
            plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('MSE Score')
            plt.title(f'Learning Curves - {model_name}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'results/learning_curves_{model_name.lower().replace(" ", "_")}.png')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting learning curves for {model_name}: {e}")
    
    def plot_residuals(self, y_true, y_pred, model_name):
        """Plot residual analysis"""
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_true, y_pred, alpha=0.6)
        axes[1, 1].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'red', linestyle='--')
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'results/residual_analysis_{model_name.lower().replace(" ", "_")}.png')
        plt.close()
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=20):
        """Plot feature importance"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            else:
                print(f"No feature importance available for {model_name}")
                return
            
            # Get top features
            indices = np.argsort(importance)[::-1][:top_n]
            top_features = [feature_names[i] for i in indices]
            top_importance = importance[indices]
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_importance)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Feature Importance')
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f'results/feature_importance_{model_name.lower().replace(" ", "_")}.png')
            plt.close()
            
        except Exception as e:
            print(f"Error plotting feature importance for {model_name}: {e}")
    
    def create_model_comparison_report(self, save_path='results/model_comparison_report.csv'):
        """Create comprehensive model comparison report"""
        if not self.results:
            print("No evaluation results available")
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results).T
        
        # Sort by primary metric (R² for regression, F1 for classification)
        if 'R²' in results_df.columns:
            results_df = results_df.sort_values('R²', ascending=False)
        elif 'F1' in results_df.columns:
            results_df = results_df.sort_values('F1', ascending=False)
        
        # Save results
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        results_df.to_csv(save_path)
        
        # Create visualization
        self._plot_model_comparison(results_df)
        
        print(f"Model comparison report saved to {save_path}")
        return results_df
    
    def _plot_model_comparison(self, results_df):
        """Plot model comparison visualization"""
        # Select key metrics for visualization
        if 'R²' in results_df.columns:
            metrics = ['MAE', 'RMSE', 'R²']
        else:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        if not available_metrics:
            return
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5*len(available_metrics), 6))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            results_df[metric].plot(kind='bar', ax=axes[i], color='skyblue')
            axes[i].set_title(f'{metric} Comparison')
            axes[i].set_ylabel(metric)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('results/model_comparison_visualization.png')
        plt.close()
    
    def generate_prediction_intervals(self, models, X_test, confidence=0.95):
        """Generate prediction intervals using ensemble variance"""
        predictions = []
        
        for model in models:
            try:
                pred = model.predict(X_test)
                predictions.append(pred)
            except:
                continue
        
        if not predictions:
            return None, None, None
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        return mean_pred, lower_bound, upper_bound