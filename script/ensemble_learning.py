import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from script.advanced_algorithms import NeuralNetworkRegressor, LSTMForecaster
import joblib
import os

class StackingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, meta_learner=None, cv_folds=5, random_state=42):
        self.base_models = base_models
        self.meta_learner = meta_learner or LinearRegression()
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.trained_base_models = {}
        self.meta_features_columns = []
        
    def fit(self, X, y):
        """Train stacking ensemble"""
        print("Training stacking ensemble...")
        
        # Create meta-features using cross-validation
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((len(X), len(self.base_models)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"Processing fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    # Clone and train model
                    model_clone = self._clone_model(model)
                    model_clone.fit(X_train_fold, y_train_fold)
                    
                    # Predict on validation set
                    val_pred = model_clone.predict(X_val_fold)
                    meta_features[val_idx, i] = val_pred
                    
                except Exception as e:
                    print(f"Error in fold {fold} for {name}: {e}")
                    meta_features[val_idx, i] = np.mean(y_train_fold)
        
        # Train base models on full dataset
        for name, model in self.base_models.items():
            print(f"Training {name} on full dataset...")
            try:
                model_clone = self._clone_model(model)
                model_clone.fit(X, y)
                self.trained_base_models[name] = model_clone
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        # Train meta-learner
        meta_df = pd.DataFrame(meta_features, columns=list(self.base_models.keys()))
        self.meta_features_columns = meta_df.columns.tolist()
        self.meta_learner.fit(meta_df, y)
        
        return self
    
    def predict(self, X):
        """Make stacking predictions"""
        # Get base model predictions
        base_predictions = np.zeros((len(X), len(self.trained_base_models)))
        
        for i, (name, model) in enumerate(self.trained_base_models.items()):
            try:
                base_predictions[:, i] = model.predict(X)
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                base_predictions[:, i] = 0
        
        # Create meta-features DataFrame
        meta_df = pd.DataFrame(base_predictions, columns=self.meta_features_columns)
        
        # Meta-learner prediction
        return self.meta_learner.predict(meta_df)
    
    def _clone_model(self, model):
        """Clone a model with same parameters"""
        if hasattr(model, 'get_params'):
            params = model.get_params()
            return type(model)(**params)
        else:
            # For custom models, return a new instance
            return type(model)()

class BlendingEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, weights=None):
        self.base_models = base_models
        self.weights = weights
        self.trained_models = {}
        
    def fit(self, X, y):
        """Train blending ensemble"""
        print("Training blending ensemble...")
        
        # Train all base models
        for name, model in self.base_models.items():
            print(f"Training {name}...")
            try:
                model_clone = self._clone_model(model)
                model_clone.fit(X, y)
                self.trained_models[name] = model_clone
            except Exception as e:
                print(f"Error training {name}: {e}")
        
        # If no weights provided, use equal weights
        if self.weights is None:
            self.weights = np.ones(len(self.trained_models)) / len(self.trained_models)
        
        return self
    
    def predict(self, X):
        """Make blended predictions"""
        predictions = []
        
        for name, model in self.trained_models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                predictions.append(np.zeros(len(X)))
        
        if not predictions:
            return np.zeros(len(X))
        
        # Weighted average
        predictions = np.array(predictions).T
        return np.average(predictions, axis=1, weights=self.weights[:len(predictions[0])])
    
    def _clone_model(self, model):
        """Clone a model with same parameters"""
        if hasattr(model, 'get_params'):
            params = model.get_params()
            return type(model)(**params)
        else:
            return type(model)()

class AdaptiveEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_models, adaptation_method='performance_weighted'):
        self.base_models = base_models
        self.adaptation_method = adaptation_method
        self.trained_models = {}
        self.model_weights = {}
        
    def fit(self, X, y):
        """Train adaptive ensemble"""
        print("Training adaptive ensemble...")
        
        # Train base models and calculate weights
        model_scores = {}
        
        for name, model in self.base_models.items():
            print(f"Training and evaluating {name}...")
            try:
                # Cross-validation to get performance
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
                model_scores[name] = -scores.mean()
                
                # Train on full dataset
                model_clone = self._clone_model(model)
                model_clone.fit(X, y)
                self.trained_models[name] = model_clone
                
            except Exception as e:
                print(f"Error with {name}: {e}")
                model_scores[name] = float('inf')
        
        # Calculate adaptive weights
        if self.adaptation_method == 'performance_weighted':
            # Inverse of error (lower error = higher weight)
            inv_scores = [1 / (score + 1e-8) for score in model_scores.values()]
            total_inv_score = sum(inv_scores)
            self.model_weights = {
                name: inv_score / total_inv_score 
                for name, inv_score in zip(model_scores.keys(), inv_scores)
            }
        
        print("Model weights:", self.model_weights)
        return self
    
    def predict(self, X):
        """Make adaptive predictions"""
        weighted_predictions = np.zeros(len(X))
        
        for name, model in self.trained_models.items():
            try:
                pred = model.predict(X)
                weight = self.model_weights.get(name, 0)
                weighted_predictions += weight * pred
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
        
        return weighted_predictions
    
    def _clone_model(self, model):
        """Clone a model with same parameters"""
        if hasattr(model, 'get_params'):
            params = model.get_params()
            return type(model)(**params)
        else:
            return type(model)()

def create_ensemble_models():
    """Create different ensemble configurations"""
    
    # Base models for ensemble
    base_models = {
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
        'catboost': cb.CatBoostRegressor(iterations=100, random_state=42, verbose=False),
        'neural_network': NeuralNetworkRegressor(hidden_layers=[64, 32], epochs=50)
    }
    
    ensembles = {
        'stacking': StackingEnsemble(
            base_models=base_models.copy(),
            meta_learner=LinearRegression()
        ),
        'blending': BlendingEnsemble(
            base_models=base_models.copy()
        ),
        'adaptive': AdaptiveEnsemble(
            base_models=base_models.copy(),
            adaptation_method='performance_weighted'
        )
    }
    
    return ensembles

def save_ensemble_models(ensembles, filepath_prefix='models/ensemble_'):
    """Save trained ensemble models"""
    os.makedirs('models', exist_ok=True)
    
    for name, ensemble in ensembles.items():
        filepath = f"{filepath_prefix}{name}.pkl"
        joblib.dump(ensemble, filepath)
        print(f"Saved {name} ensemble to {filepath}")