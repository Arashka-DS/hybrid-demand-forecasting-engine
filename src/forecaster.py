import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb

class HybridProphetLGBM:
    def __init__(self):
        # Prophet models structural macro trends and daily/weekly Fourier terms
        self.prophet = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            interval_width=0.95
        )
        # LightGBM learns the non-linear residual shocks and lag interactions
        self.lgbm = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            random_state=42
        )
        self.feature_cols = [
            'yhat', 'hour', 'dayofweek', 'is_weekend',
            'yhat_lag_1', 'yhat_lag_24', 'rolling_mean_24'
        ]

    def _engineer_features(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        df = df_merged.copy()
        df['hour'] = df['ds'].dt.hour
        df['dayofweek'] = df['ds'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([4, 5]).astype(int) # MENA weekend: Thu/Fri
        
        # Autoregressive features engineered on structural yhat to prevent data leakage
        df['yhat_lag_1'] = df['yhat'].shift(1)
        df['yhat_lag_24'] = df['yhat'].shift(24)
        df['rolling_mean_24'] = df['yhat'].rolling(window=24, min_periods=1).mean()
        return df

    def fit(self, df: pd.DataFrame):
        """Fits Prophet and then trains LightGBM on the residual error."""
        self.prophet.fit(df[['ds', 'y']])
        prophet_pred = self.prophet.predict(df[['ds']])
        
        df_merged = df.copy()
        df_merged['yhat'] = prophet_pred['yhat'].values
        df_merged['residual'] = df_merged['y'] - df_merged['yhat']
        
        df_features = self._engineer_features(df_merged)
        # Drop warm-up rows where 24h lags are NaN
        train_data = df_features.dropna(subset=['yhat_lag_24'])
        
        X = train_data[self.feature_cols]
        y_residual = train_data['residual']
        
        self.lgbm.fit(X, y_residual)
        return self

    def predict(self, df_future: pd.DataFrame, context_history: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generates predictions for future periods.
        If context_history (tail 48h of training data) is provided, it seeds the autoregressive
        lags so zero future rows are dropped.
        """
        # 1. Prophet Baseline on future window
        prophet_future = self.prophet.predict(df_future[['ds']])
        
        if context_history is not None:
            # Combine history with future to preserve rolling lookbacks
            prophet_hist = self.prophet.predict(context_history[['ds']])
            hist_block = context_history[['ds']].copy()
            hist_block['yhat'] = prophet_hist['yhat'].values
            
            future_block = df_future[['ds']].copy()
            future_block['yhat'] = prophet_future['yhat'].values
            
            full_block = pd.concat([hist_block, future_block], ignore_index=True)
            engineered = self._engineer_features(full_block)
            # Slice strictly the future segment
            future_features = engineered.iloc[len(hist_block):].copy()
        else:
            future_block = df_future[['ds']].copy()
            future_block['yhat'] = prophet_future['yhat'].values
            future_features = self._engineer_features(future_block).bfill().ffill()

        X = future_features[self.feature_cols]
        residual_corrections = self.lgbm.predict(X)
        
        future_features['hybrid_forecast'] = np.clip(
            future_features['yhat'] + residual_corrections, 0, None
        )
        return future_features

    @staticmethod
    def evaluate_wape(actual: np.ndarray, forecast: np.ndarray) -> float:
        """Weighted Absolute Percentage Error."""
        denominator = np.sum(actual)
        if denominator == 0:
            return 0.0
        return float(np.sum(np.abs(actual - forecast)) / denominator)
