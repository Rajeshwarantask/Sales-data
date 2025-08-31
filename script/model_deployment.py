import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

class ModelDeployment:
    def __init__(self, model_dir='models', results_dir='results'):
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.loaded_models = {}
        self.scalers = {}
        
    def load_best_models(self):
        """Load the best performing models for each task"""
        print("Loading best performing models...")
        
        # Load model comparison results
        comparison_file = os.path.join(self.results_dir, 'comprehensive_model_report.csv')
        
        if not os.path.exists(comparison_file):
            print("No model comparison results found")
            return
        
        results_df = pd.read_csv(comparison_file)
        
        # Find best models for each task
        best_models = {}
        for task in results_df['Task'].unique():
            task_results = results_df[results_df['Task'] == task]
            
            if 'R²' in task_results.columns:
                best_model = task_results.loc[task_results['R²'].idxmax()]
            elif 'F1' in task_results.columns:
                best_model = task_results.loc[task_results['F1'].idxmax()]
            elif 'Silhouette Score' in task_results.columns:
                best_model = task_results.loc[task_results['Silhouette Score'].idxmax()]
            else:
                continue
            
            best_models[task] = best_model['Model']
        
        # Load the actual model files
        for task, model_name in best_models.items():
            model_file = os.path.join(self.model_dir, f"{model_name}_{task.lower().replace(' ', '_')}.pkl")
            
            if os.path.exists(model_file):
                try:
                    self.loaded_models[task] = joblib.load(model_file)
                    print(f"✅ Loaded {model_name} for {task}")
                except Exception as e:
                    print(f"❌ Error loading {model_name}: {e}")
        
        return self.loaded_models
    
    def predict_sales_forecast(self, input_data):
        """Make sales forecast predictions"""
        if 'Sales Forecasting' not in self.loaded_models:
            return None
        
        model = self.loaded_models['Sales Forecasting']
        
        try:
            # Ensure input data has the right format
            if isinstance(input_data, dict):
                input_data = pd.DataFrame([input_data])
            
            prediction = model.predict(input_data)
            
            return {
                'prediction': float(prediction[0]) if len(prediction) > 0 else 0,
                'timestamp': datetime.now().isoformat(),
                'model_used': type(model).__name__
            }
        except Exception as e:
            print(f"Error in sales forecasting: {e}")
            return None
    
    def predict_churn_probability(self, customer_data):
        """Predict customer churn probability"""
        if 'Churn Prediction' not in self.loaded_models:
            return None
        
        model = self.loaded_models['Churn Prediction']
        
        try:
            if isinstance(customer_data, dict):
                customer_data = pd.DataFrame([customer_data])
            
            if hasattr(model, 'predict_proba'):
                churn_prob = model.predict_proba(customer_data)[0, 1]
            else:
                churn_prob = model.predict(customer_data)[0]
            
            return {
                'churn_probability': float(churn_prob),
                'risk_level': 'High' if churn_prob > 0.7 else 'Medium' if churn_prob > 0.3 else 'Low',
                'timestamp': datetime.now().isoformat(),
                'model_used': type(model).__name__
            }
        except Exception as e:
            print(f"Error in churn prediction: {e}")
            return None
    
    def get_customer_segment(self, customer_data):
        """Get customer segment prediction"""
        # This would use the clustering model results
        # For now, return a mock implementation
        return {
            'segment': 'High Value',
            'segment_id': 1,
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_predict(self, data_file, output_file):
        """Make batch predictions on a dataset"""
        print(f"Making batch predictions on {data_file}...")
        
        try:
            df = pd.read_csv(data_file)
            results = []
            
            for idx, row in df.iterrows():
                row_dict = row.to_dict()
                
                # Sales forecast
                sales_pred = self.predict_sales_forecast(row_dict)
                
                # Churn prediction
                churn_pred = self.predict_churn_probability(row_dict)
                
                # Customer segment
                segment_pred = self.get_customer_segment(row_dict)
                
                result = {
                    'row_id': idx,
                    'sales_forecast': sales_pred['prediction'] if sales_pred else None,
                    'churn_probability': churn_pred['churn_probability'] if churn_pred else None,
                    'churn_risk_level': churn_pred['risk_level'] if churn_pred else None,
                    'customer_segment': segment_pred['segment'] if segment_pred else None
                }
                
                results.append(result)
            
            # Save results
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_file, index=False)
            print(f"✅ Batch predictions saved to {output_file}")
            
            return results_df
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return None
    
    def create_api_endpoint_simulation(self):
        """Simulate API endpoints for model serving"""
        print("Creating API endpoint simulation...")
        
        api_examples = {
            'sales_forecast': {
                'endpoint': '/api/predict/sales',
                'method': 'POST',
                'example_input': {
                    'unit_price': 25.99,
                    'quantity': 2,
                    'discount_applied': 5.0,
                    'customer_age': 35,
                    'is_weekend': 0,
                    'is_holiday': 0,
                    'temperature': 72,
                    'market_sentiment': 1
                },
                'example_output': {
                    'predicted_sales': 45.32,
                    'confidence_interval': [40.15, 50.49],
                    'model_confidence': 0.87
                }
            },
            'churn_prediction': {
                'endpoint': '/api/predict/churn',
                'method': 'POST',
                'example_input': {
                    'customer_id': 'CUST_12345',
                    'days_since_last_purchase': 45,
                    'total_transactions': 12,
                    'average_spent': 156.78,
                    'membership_years': 2.5,
                    'app_usage': 15,
                    'website_visits': 8
                },
                'example_output': {
                    'churn_probability': 0.23,
                    'risk_level': 'Low',
                    'recommended_action': 'Monitor',
                    'retention_strategy': 'Standard engagement'
                }
            }
        }
        
        # Save API documentation
        with open('results/api_documentation.json', 'w') as f:
            json.dump(api_examples, f, indent=2)
        
        print("✅ API documentation saved to results/api_documentation.json")
        return api_examples

def run_deployment_demo():
    """Demonstrate model deployment capabilities"""
    print("\n🚀 Running Model Deployment Demo...")
    
    deployment = ModelDeployment()
    
    # Load best models
    deployment.load_best_models()
    
    # Create API simulation
    api_docs = deployment.create_api_endpoint_simulation()
    
    # Demo predictions
    print("\n🔮 Demo Predictions:")
    
    # Sales forecast demo
    sales_input = {
        'unit_price': 29.99,
        'quantity': 3,
        'discount_applied': 10.0,
        'age': 42,
        'membership_years': 1.5
    }
    
    sales_result = deployment.predict_sales_forecast(sales_input)
    if sales_result:
        print(f"Sales Forecast: ${sales_result['prediction']:.2f}")
    
    # Churn prediction demo
    churn_input = {
        'days_since_last_purchase': 60,
        'customer_transactions_count': 8,
        'customer_average_spent': 125.50,
        'membership_years': 3.0
    }
    
    churn_result = deployment.predict_churn_probability(churn_input)
    if churn_result:
        print(f"Churn Risk: {churn_result['risk_level']} ({churn_result['churn_probability']:.2%})")
    
    print("\n✅ Deployment demo completed!")

if __name__ == "__main__":
    main()
    run_deployment_demo()