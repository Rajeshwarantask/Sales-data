import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from feature_engine.datetime import DatetimeFeatures
from feature_engine.creation import CyclicalFeatures
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        
    def create_time_features(self, df, date_col='transaction_date'):
        """Create comprehensive time-based features"""
        print("Creating advanced time features...")
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Basic time features
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['dayofyear'] = df[date_col].dt.dayofyear
        df['week'] = df[date_col].dt.isocalendar().week
        df['quarter'] = df[date_col].dt.quarter
        
        # Advanced time features
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for periodic features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        df['hour_sin'] = np.sin(2 * np.pi * df[date_col].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df[date_col].dt.hour / 24)
        
        return df
    
    def create_lag_features(self, df, target_col, periods=[1, 3, 7, 14, 30]):
        """Create lag features for time series"""
        print("Creating lag features...")
        
        df = df.sort_values('transaction_date')
        for period in periods:
            df[f'{target_col}_lag_{period}'] = df[target_col].shift(period)
            
        return df
    
    def create_rolling_features(self, df, target_col, windows=[3, 7, 14, 30]):
        """Create rolling window statistics"""
        print("Creating rolling features...")
        
        df = df.sort_values('transaction_date')
        for window in windows:
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
            df[f'{target_col}_rolling_median_{window}'] = df[target_col].rolling(window).median()
            
        return df
    
    def create_interaction_features(self, df, feature_pairs):
        """Create interaction features between specified pairs"""
        print("Creating interaction features...")
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_{feat2}_interaction'] = df[feat1] * df[feat2]
                df[f'{feat1}_{feat2}_ratio'] = df[feat1] / (df[feat2] + 1e-8)
                
        return df
    
    def create_customer_features(self, df):
        """Create advanced customer-level features"""
        print("Creating customer features...")
        
        # Customer transaction patterns
        customer_stats = df.groupby('customer_id').agg({
            'total_sales': ['sum', 'mean', 'std', 'count'],
            'quantity': ['sum', 'mean'],
            'discount_applied': ['mean', 'max'],
            'transaction_date': ['min', 'max']
        }).reset_index()
        
        # Flatten column names
        customer_stats.columns = ['customer_id'] + [
            f'customer_{col[0]}_{col[1]}' for col in customer_stats.columns[1:]
        ]
        
        # Calculate customer lifetime and frequency
        customer_stats['customer_lifetime_days'] = (
            customer_stats['customer_transaction_date_max'] - 
            customer_stats['customer_transaction_date_min']
        ).dt.days
        
        customer_stats['customer_purchase_frequency'] = (
            customer_stats['customer_total_sales_count'] / 
            (customer_stats['customer_lifetime_days'] + 1)
        )
        
        # Merge back to main dataframe
        df = df.merge(customer_stats, on='customer_id', how='left')
        
        return df
    
    def create_product_features(self, df):
        """Create advanced product-level features"""
        print("Creating product features...")
        
        # Product performance metrics
        product_stats = df.groupby('product_id').agg({
            'total_sales': ['sum', 'mean', 'count'],
            'quantity': ['sum', 'mean'],
            'unit_price': ['mean', 'std'],
            'discount_applied': ['mean', 'count']
        }).reset_index()
        
        # Flatten column names
        product_stats.columns = ['product_id'] + [
            f'product_{col[0]}_{col[1]}' for col in product_stats.columns[1:]
        ]
        
        # Product popularity and pricing strategy
        product_stats['product_popularity_score'] = (
            product_stats['product_total_sales_count'] / 
            product_stats['product_total_sales_count'].max()
        )
        
        product_stats['product_discount_frequency'] = (
            product_stats['product_discount_applied_count'] / 
            product_stats['product_total_sales_count']
        )
        
        # Merge back to main dataframe
        df = df.merge(product_stats, on='product_id', how='left')
        
        return df
    
    def create_polynomial_features(self, df, feature_cols, degree=2):
        """Create polynomial features for specified columns"""
        print("Creating polynomial features...")
        
        poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=True)
        poly_features = poly.fit_transform(df[feature_cols])
        
        feature_names = poly.get_feature_names_out(feature_cols)
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Add only new polynomial features (exclude original features)
        new_features = [col for col in poly_df.columns if col not in feature_cols]
        df = pd.concat([df, poly_df[new_features]], axis=1)
        
        return df
    
    def select_features(self, X, y, method='mutual_info', k=50):
        """Select top k features using specified method"""
        print(f"Selecting top {k} features using {method}...")
        
        if method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)
            
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        return X_selected, selected_features, selector
    
    def engineer_all_features(self, df, target_col='total_sales'):
        """Apply all feature engineering techniques"""
        print("Starting comprehensive feature engineering...")
        
        # Time features
        df = self.create_time_features(df)
        
        # Lag and rolling features
        df = self.create_lag_features(df, target_col)
        df = self.create_rolling_features(df, target_col)
        
        # Customer and product features
        df = self.create_customer_features(df)
        df = self.create_product_features(df)
        
        # Interaction features
        interaction_pairs = [
            ('unit_price', 'discount_applied'),
            ('quantity', 'unit_price'),
            ('age', 'income_bracket'),
            ('membership_years', 'loyalty_program')
        ]
        df = self.create_interaction_features(df, interaction_pairs)
        
        # Polynomial features for key numerical columns
        numerical_cols = ['unit_price', 'quantity', 'age', 'membership_years']
        available_cols = [col for col in numerical_cols if col in df.columns]
        if available_cols:
            df = self.create_polynomial_features(df, available_cols)
        
        print("Feature engineering completed!")
        return df