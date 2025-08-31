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

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    'regression': {
        'r2_score': 0.85,
        'rmse_threshold': 0.15
    },
    'classification': {
        'f1_score': 0.85,
        'accuracy': 0.90,
        'precision': 0.85,
        'recall': 0.85
    },
    'clustering': {
        'silhouette_score': 0.6
    }
}

# Advanced model configurations
ADVANCED_MODEL_CONFIGS = {
    'neural_network': {
        'hidden_layers': [[128, 64, 32], [256, 128, 64], [64, 32, 16]],
        'dropout': [0.2, 0.3, 0.4],
        'epochs': [50, 100, 150],
        'batch_size': [32, 64, 128]
    },
    'lstm': {
        'sequence_length': [15, 30, 60],
        'units': [32, 50, 100],
        'dropout': [0.1, 0.2, 0.3],
        'layers': [1, 2, 3]
    }
}