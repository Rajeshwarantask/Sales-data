import pandas as pd
import requests
import holidays
import yfinance as yf
from datetime import datetime, timedelta
import os

class ExternalDataIntegrator:
    def __init__(self):
        self.weather_api_key = None  # Set your API key here
        
    def get_holiday_features(self, df, date_col='transaction_date', country='US'):
        """Add holiday features to the dataset"""
        print("Adding holiday features...")
        
        df[date_col] = pd.to_datetime(df[date_col])
        us_holidays = holidays.country_holidays(country)
        
        df['is_holiday'] = df[date_col].dt.date.isin(us_holidays)
        df['days_to_holiday'] = df[date_col].apply(
            lambda x: min([abs((x.date() - h).days) for h in us_holidays 
                          if abs((x.date() - h).days) <= 30], default=30)
        )
        
        # Add specific holiday types
        df['is_christmas_season'] = ((df[date_col].dt.month == 12) & 
                                   (df[date_col].dt.day >= 15)).astype(int)
        df['is_black_friday'] = df[date_col].apply(self._is_black_friday).astype(int)
        df['is_new_year'] = ((df[date_col].dt.month == 1) & 
                           (df[date_col].dt.day <= 7)).astype(int)
        
        return df
    
    def _is_black_friday(self, date):
        """Check if date is Black Friday (4th Thursday of November + 1 day)"""
        if date.month != 11:
            return False
        
        # Find 4th Thursday of November
        first_day = date.replace(day=1)
        first_thursday = first_day + timedelta(days=(3 - first_day.weekday()) % 7)
        fourth_thursday = first_thursday + timedelta(days=21)
        black_friday = fourth_thursday + timedelta(days=1)
        
        return date.date() == black_friday.date()
    
    def get_economic_indicators(self, df, date_col='transaction_date'):
        """Add economic indicators using Yahoo Finance"""
        print("Adding economic indicators...")
        
        df[date_col] = pd.to_datetime(df[date_col])
        start_date = df[date_col].min() - timedelta(days=30)
        end_date = df[date_col].max() + timedelta(days=30)
        
        try:
            # Download economic indicators
            sp500 = yf.download('^GSPC', start=start_date, end=end_date)['Close']
            vix = yf.download('^VIX', start=start_date, end=end_date)['Close']
            dxy = yf.download('DX-Y.NYB', start=start_date, end=end_date)['Close']
            
            # Create economic indicators DataFrame
            econ_df = pd.DataFrame({
                'date': sp500.index,
                'sp500': sp500.values,
                'vix': vix.values,
                'dxy': dxy.values
            })
            
            # Add moving averages and volatility
            econ_df['sp500_ma_30'] = econ_df['sp500'].rolling(30).mean()
            econ_df['sp500_volatility'] = econ_df['sp500'].rolling(30).std()
            econ_df['market_sentiment'] = (econ_df['sp500'] > econ_df['sp500_ma_30']).astype(int)
            
            # Merge with main dataset
            df['date_only'] = df[date_col].dt.date
            econ_df['date_only'] = econ_df['date'].dt.date
            
            df = df.merge(econ_df.drop('date', axis=1), on='date_only', how='left')
            df = df.drop('date_only', axis=1)
            
            # Forward fill missing values
            econ_cols = ['sp500', 'vix', 'dxy', 'sp500_ma_30', 'sp500_volatility', 'market_sentiment']
            for col in econ_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(method='ffill')
            
        except Exception as e:
            print(f"Warning: Could not fetch economic data: {e}")
            # Add dummy economic indicators
            df['sp500'] = 4000
            df['vix'] = 20
            df['market_sentiment'] = 1
            
        return df
    
    def get_weather_features(self, df, date_col='transaction_date'):
        """Add weather features (mock implementation)"""
        print("Adding weather features...")
        
        # Mock weather data (replace with actual weather API)
        np.random.seed(42)
        df['temperature'] = np.random.normal(70, 15, len(df))
        df['precipitation'] = np.random.exponential(0.1, len(df))
        df['is_rainy'] = (df['precipitation'] > 0.5).astype(int)
        df['is_extreme_weather'] = ((df['temperature'] < 32) | 
                                   (df['temperature'] > 90) | 
                                   (df['precipitation'] > 1.0)).astype(int)
        
        return df
    
    def add_competitor_data(self, df):
        """Add competitor pricing and promotion data (mock)"""
        print("Adding competitor data...")
        
        np.random.seed(42)
        df['competitor_price_ratio'] = np.random.normal(1.0, 0.2, len(df))
        df['competitor_promotion_active'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        df['market_share_estimate'] = np.random.normal(0.15, 0.05, len(df))
        
        return df

import numpy as np