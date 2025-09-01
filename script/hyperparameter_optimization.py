import os
import optuna
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import lightgbm as lgb
import xgboost as xgb
from optuna.pruners import MedianPruner

class HyperparameterOptimizer:
    def __init__(self, n_trials=50):
        self.n_trials = n_trials
        self.best_params = {}
        self.study_results = {}
        self.random_state = 42
        self.cv_folds = 3  # Default CV folds

    def optimize_model(self, model, X_train, y_train, demo_mode=False):
        """Optimize hyperparameters for a given model."""
        print(f"🔍 Optimizing {model.__class__.__name__}...")

        # Sample the dataset for DEMO mode
        if demo_mode:
            print("🔎 DEMO mode active: sampling 10k rows for speed")
            X_train = X_train.sample(n=10000, random_state=42) if len(X_train) > 10000 else X_train
            y_train = y_train.loc[X_train.index]

        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
            # Update model parameters
            model.set_params(**params)

            # Cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring='neg_mean_squared_error')
            return -scores.mean()

        # Create a new study with a unique name for testing
        study = optuna.create_study(direction='minimize', pruner=MedianPruner(), study_name="test_sales_forecasting", load_if_exists=False)
        study.optimize(objective, n_trials=self.n_trials, timeout=300)  # Use self.n_trials for dynamic adjustment

        print(f"✅ Best parameters: {study.best_params}")
        return study.best_params

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
    
        # Use SQLite for parallel storage
        storage = optuna.storages.RDBStorage("sqlite:///optuna.db")
        study = optuna.create_study(
            direction="minimize", 
            storage=storage, 
            study_name="sales_forecasting_lightgbm", 
            load_if_exists=True
        )

        # Run trials dynamically based on mode
        study.optimize(objective, n_trials=self.n_trials, n_jobs=4)
    
        self.best_params['lightgbm_regressor'] = study.best_params
        self.study_results['lightgbm_regressor'] = study
        
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
    
        # Use SQLite for parallel storage
        storage = optuna.storages.RDBStorage("sqlite:///optuna.db")
        study = optuna.create_study(
            direction="minimize", 
            storage=storage, 
            study_name="sales_forecasting_xgboost",  # Unique study name for XGBoost
            load_if_exists=True
        )

        # Run trials in parallel (e.g., using multiprocessing)
        study.optimize(objective, n_trials=20, n_jobs=4)  # Use 4 parallel workers
        
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
    
        # Use SQLite for parallel storage
        storage = optuna.storages.RDBStorage("sqlite:///optuna.db")
        study = optuna.create_study(
            direction="minimize", 
            storage=storage, 
            study_name="sales_forecasting_randomforest",  # Unique study name for Random Forest
            load_if_exists=True
        )

        # Run trials in parallel (e.g., using multiprocessing)
        study.optimize(objective, n_trials=20, n_jobs=4)  # Use 4 parallel workers
        
        self.best_params['random_forest_regressor'] = study.best_params
        self.study_results['random_forest_regressor'] = study
        
        return study.best_params
    
    def optimize_all_models(self, X, y, task='regression'):
        """Optimize all models for the given task"""
        print(f"Starting hyperparameter optimization for {task}...")
        
        optimizers = {
            'lightgbm': self.optimize_lightgbm_regressor,
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

def apply_pca(X_train, X_test, n_components=50):
    """Apply PCA to reduce dimensionality."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_test_pca

# Example for LightGBM
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    device='gpu',  # Enable GPU
    random_state=42
)

# Example for XGBoost
model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.1,
    tree_method='gpu_hist',  # Enable GPU
    random_state=42
)