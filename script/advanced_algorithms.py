import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class LSTMForecaster:
    def __init__(self, sequence_length=30, units=50, dropout=0.2):
        self.sequence_length = sequence_length
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()
        
    def create_sequences(self, data):
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(self.units, return_sequences=True, input_shape=input_shape),
            Dropout(self.dropout),
            BatchNormalization(),
            
            LSTM(self.units // 2, return_sequences=True),
            Dropout(self.dropout),
            BatchNormalization(),
            
            LSTM(self.units // 4, return_sequences=False),
            Dropout(self.dropout),
            
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X, y):
        """Train the LSTM model"""
        # Scale the data
        X_scaled = self.scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape)
        y_scaled = self.scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_seq, y_seq = self.create_sequences(X_scaled.flatten())
        
        if len(X_seq) == 0:
            raise ValueError("Not enough data to create sequences")
        
        # Reshape for LSTM
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        
        # Build model
        self.model = self.build_model((X_seq.shape[1], 1))
        
        # Callbacks
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-7)
        
        # Train model
        history = self.model.fit(
            X_seq, y_seq,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions with the LSTM model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        X_scaled = self.scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        X_seq, _ = self.create_sequences(X_scaled.flatten())
        
        if len(X_seq) == 0:
            # Return simple prediction if not enough data
            return np.full(len(X), X.mean())
        
        X_seq = X_seq.reshape((X_seq.shape[0], X_seq.shape[1], 1))
        predictions_scaled = self.model.predict(X_seq, verbose=0)
        predictions = self.scaler.inverse_transform(predictions_scaled).flatten()
        
        # Extend predictions to match input length
        if len(predictions) < len(X):
            last_pred = predictions[-1] if len(predictions) > 0 else X.mean()
            predictions = np.concatenate([
                np.full(len(X) - len(predictions), last_pred),
                predictions
            ])
        
        return predictions[:len(X)]

class AdvancedEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, base_models=None, meta_learner=None, use_stacking=True):
        self.base_models = base_models or self._get_default_models()
        self.meta_learner = meta_learner or LinearRegression()
        self.use_stacking = use_stacking
        self.trained_models = {}
        
    def _get_default_models(self):
        """Get default set of diverse models"""
        return {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgb': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'lgb': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'cb': cb.CatBoostRegressor(iterations=100, random_state=42, verbose=False)
        }
    
    def fit(self, X, y):
        """Train ensemble model"""
        print("Training ensemble model...")
        
        # Train base models
        base_predictions = np.zeros((len(X), len(self.base_models)))
        
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Training {name}...")
            try:
                model.fit(X, y)
                self.trained_models[name] = model
                base_predictions[:, i] = model.predict(X)
            except Exception as e:
                print(f"Error training {name}: {e}")
                base_predictions[:, i] = np.mean(y)
        
        # Train meta-learner if using stacking
        if self.use_stacking:
            self.meta_learner.fit(base_predictions, y)
        
        return self
    
    def predict(self, X):
        """Make ensemble predictions"""
        base_predictions = np.zeros((len(X), len(self.trained_models)))
        
        for i, (name, model) in enumerate(self.trained_models.items()):
            try:
                base_predictions[:, i] = model.predict(X)
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                base_predictions[:, i] = 0
        
        if self.use_stacking and hasattr(self.meta_learner, 'predict'):
            return self.meta_learner.predict(base_predictions)
        else:
            # Simple averaging
            return np.mean(base_predictions, axis=1)

class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layers=[100, 50], dropout=0.3, epochs=100):
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        
    def build_model(self, input_dim):
        """Build neural network architecture"""
        model = Sequential()
        
        # Input layer
        model.add(Dense(self.hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(Dropout(self.dropout))
        model.add(BatchNormalization())
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout))
            model.add(BatchNormalization())
        
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, X, y):
        """Train the neural network"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Build model
        self.model = self.build_model(X_scaled.shape[1])
        
        # Callbacks
        early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(patience=7, factor=0.5, min_lr=1e-7)
        
        # Train model
        self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

def get_advanced_models():
    """Return dictionary of advanced models"""
    return {
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbose=-1
        ),
        'catboost': cb.CatBoostRegressor(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=False
        ),
        'neural_network': NeuralNetworkRegressor(
            hidden_layers=[128, 64, 32],
            dropout=0.3,
            epochs=100
        ),
        'lstm': LSTMForecaster(
            sequence_length=30,
            units=50,
            dropout=0.2
        ),
        'ensemble': AdvancedEnsemble()
    }