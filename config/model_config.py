"""
Configuration file for model parameters and settings
"""

# Model configurations
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'xgboost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'subsample': [0.8, 0.9, 1.0]
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 6, 9],
        'num_leaves': [31, 50, 100]
    },
    'catboost': {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 6, 9]
    }
}

# Cross-validation settings
CV_SETTINGS = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': 42
}

# Feature engineering settings
FEATURE_SETTINGS = {
    'lag_periods': [1, 3, 7, 14, 30],
    'rolling_windows': [3, 7, 14, 30],
    'seasonal_periods': [7, 30, 365]
}

# Ensemble settings
ENSEMBLE_SETTINGS = {
    'meta_learner': 'linear_regression',
    'use_feature_selection': True,
    'n_features_select': 50
}