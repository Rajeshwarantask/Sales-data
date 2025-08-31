import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, accuracy_score
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class HyperparameterOptimizer:
    def __init__(self, n_trials=100, cv_folds=5, random_state=42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = {}
        self.study_results = {}
        
    def optimize_lightgbm_regressor(self, X, y):
        """Optimize LightGBM regressor hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state,
                'verbose': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['lightgbm_regressor'] = study.best_params
        self.study_results['lightgbm_regressor'] = study
        
        return study.best_params
    
    def optimize_catboost_regressor(self, X, y):
        """Optimize CatBoost regressor hyperparameters"""
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('depth', 3, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': self.random_state,
                'verbose': False
            }
            
            model = cb.CatBoostRegressor(**params)
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['catboost_regressor'] = study.best_params
        self.study_results['catboost_regressor'] = study
        
        return study.best_params
    
    def optimize_xgboost_regressor(self, X, y):
        """Optimize XGBoost regressor hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': self.random_state
            }
            
            model = xgb.XGBRegressor(**params)
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['xgboost_regressor'] = study.best_params
        self.study_results['xgboost_regressor'] = study
        
        return study.best_params
    
    def optimize_random_forest_regressor(self, X, y):
        """Optimize Random Forest regressor hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
            
            model = RandomForestRegressor(**params)
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error')
            return -scores.mean()
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        
        self.best_params['random_forest_regressor'] = study.best_params
        self.study_results['random_forest_regressor'] = study
        
        return study.best_params
    
    def optimize_all_models(self, X, y, task='regression'):
        """Optimize all models for the given task"""
        print(f"Starting hyperparameter optimization for {task}...")
        
        optimizers = {
            'lightgbm': self.optimize_lightgbm_regressor,
            'catboost': self.optimize_catboost_regressor,
            'xgboost': self.optimize_xgboost_regressor,
            'random_forest': self.optimize_random_forest_regressor
        }
        
        results = {}
        for name, optimizer in optimizers.items():
            print(f"Optimizing {name}...")
            try:
                best_params = optimizer(X, y)
                results[name] = best_params
                print(f"✅ {name} optimization completed")
            except Exception as e:
                print(f"❌ Error optimizing {name}: {e}")
                results[name] = {}
        
        return results
    
    def get_optimized_models(self, task='regression'):
        """Get models with optimized hyperparameters"""
        models = {}
        
        if 'lightgbm_regressor' in self.best_params:
            models['lightgbm'] = lgb.LGBMRegressor(**self.best_params['lightgbm_regressor'])
        
        if 'catboost_regressor' in self.best_params:
            models['catboost'] = cb.CatBoostRegressor(**self.best_params['catboost_regressor'])
        
        if 'xgboost_regressor' in self.best_params:
            models['xgboost'] = xgb.XGBRegressor(**self.best_params['xgboost_regressor'])
        
        if 'random_forest_regressor' in self.best_params:
            models['random_forest'] = RandomForestRegressor(**self.best_params['random_forest_regressor'])
        
        return models
    
    def save_optimization_results(self, filepath='results/hyperparameter_optimization.json'):
        """Save optimization results"""
        import json
        
        results = {
            'best_params': self.best_params,
            'study_summaries': {}
        }
        
        for name, study in self.study_results.items():
            results['study_summaries'][name] = {
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'best_params': study.best_params
            }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Optimization results saved to {filepath}")