import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb

class HybridProphetLGBM:
    def __init__(self):
        # Prophet handles the structural macro-seasonality and trends
        self.prophet = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        # LightGBM learns the non-linear residuals and micro-shocks
        self.lgbm = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, max_depth=6)

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts temporal features and autoregressive Prophet lags."""
        df = df.copy()
        df['hour'] = df['ds'].dt.hour
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Use Prophet's yhat for lags to prevent data leakage during multi-step inference
        df['yhat_lag_1'] = df['yhat'].shift(1)
        df['yhat_lag_24'] = df['yhat'].shift(24)
        return df.dropna()

    def fit(self, df: pd.DataFrame):
        """Trains the hybrid architecture on historical time-series data."""
        # 1. Fit Prophet Baseline
        self.prophet.fit(df)
        prophet_pred = self.prophet.predict(df[['ds']])
        
        # 2. Calculate Residuals (Actual - Prophet Prediction)
        df_merged = df.copy()
        df_merged['yhat'] = prophet_pred['yhat'].values
        df_merged['residual'] = df_merged['y'] - df_merged['yhat']
        
        # 3. Extract Features for LightGBM
        df_features = self.extract_features(df_merged)
        self.feature_cols = ['yhat', 'hour', 'dayofweek', 'is_weekend', 'yhat_lag_1', 'yhat_lag_24']
        
        X = df_features[self.feature_cols]
        y_residual = df_features['residual']
        
        # 4. Train LightGBM on the residuals
        self.lgbm.fit(X, y_residual)
        print("Hybrid Engine Trained: Prophet (Macro) + LightGBM (Micro-Residuals).")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generates the final hybrid forecast."""
        prophet_pred = self.prophet.predict(df[['ds']])
        
        df_merged = df.copy()
        df_merged['yhat'] = prophet_pred['yhat'].values
        
        # Handle NA values for early lags by backfilling in inference
        df_features = self.extract_features(df_merged)
        if df_features.empty:
            return df_merged # Fallback to pure Prophet if insufficient lag history
            
        X = df_features[self.feature_cols]
        residual_pred = self.lgbm.predict(X)
        
        # Final Forecast = Prophet Baseline + LightGBM Residual Correction
        df_features['hybrid_forecast'] = df_features['yhat'] + residual_pred
        return df_features

    def evaluate_wape(self, actual: np.array, forecast: np.array) -> float:
        """Weighted Absolute Percentage Error (WAPE) - highly resistant to zero-demand periods."""
        return np.sum(np.abs(actual - forecast)) / np.sum(actual)
