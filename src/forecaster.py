import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os

class HybridLiquidityForecaster:
    def __init__(self):
        self.prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        self.lgbm_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        self.cost_idle_cash = 0.02   # Penalty per unused unit of fiat
        self.cost_failed_tx = 5.0    # Penalty per failed transaction unit

    def engineer_features(self, df):
        """Generates exogenous covariates, lags, and rolling metrics."""
        df['day_of_week'] = df['ds'].dt.dayofweek
        df['is_payday'] = df['ds'].dt.day.isin([1, 28, 29, 30, 31]).astype(int)
        df['is_holiday'] = 0 # Placeholder: Integrate actual Persian calendar API
        
        # Lag Features
        df['lag_1'] = df['y'].shift(1)
        df['lag_7'] = df['y'].shift(7)
        df['lag_30'] = df['y'].shift(30)
        
        # Rolling Features
        df['rolling_7_mean'] = df['y'].shift(1).rolling(7).mean()
        df['rolling_7_std'] = df['y'].shift(1).rolling(7).std()
        
        return df.dropna().reset_index(drop=True)

    def calculate_business_metrics(self, y_true, y_pred):
        """WAPE and Custom Financial Cost Function."""
        # Weighted Absolute Percentage Error
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
        
        # Financial Impact Cost
        errors = y_pred - y_true
        over_prediction = np.sum(errors[errors > 0]) * self.cost_idle_cash
        under_prediction = np.sum(np.abs(errors[errors < 0])) * self.cost_failed_tx
        total_cost = over_prediction + under_prediction
        
        return wape, total_cost

    def train_hybrid_model(self, data):
        """Step 1: Prophet, Step 2: Extract Residuals, Step 3: LightGBM."""
        data = self.engineer_features(data)
        
        # Time-Series Expanding Window Cross-Validation
        tscv = TimeSeriesSplit(n_splits=3)
        print("Executing Walk-Forward Validation...")
        
        for train_index, test_index in tscv.split(data):
            train_cv, test_cv = data.iloc[train_index], data.iloc[test_index]
            
            # Step 1: Prophet Baseline
            prophet_cv = Prophet().fit(train_cv[['ds', 'y']])
            prophet_train_preds = prophet_cv.predict(train_cv[['ds']])['yhat']
            
            # Step 2: Extract Residuals
            train_cv['residual'] = train_cv['y'] - prophet_train_preds
            
            # Step 3: Train LightGBM on Residuals
            features = ['day_of_week', 'is_payday', 'lag_1', 'lag_7', 'lag_30', 'rolling_7_mean', 'rolling_7_std']
            lgbm_cv = lgb.LGBMRegressor().fit(train_cv[features], train_cv['residual'])

        # Final Full-Pass Training
        self.prophet_model.fit(data[['ds', 'y']])
        data['prophet_base'] = self.prophet_model.predict(data[['ds']])['yhat']
        data['residual'] = data['y'] - data['prophet_base']
        
        self.lgbm_model.fit(data[features], data['residual'])
        
        # Evaluate Training Set Fit
        data['lgbm_residual_pred'] = self.lgbm_model.predict(data[features])
        data['final_forecast'] = data['prophet_base'] + data['lgbm_residual_pred']
        
        wape, cost = self.calculate_business_metrics(data['y'], data['final_forecast'])
        print(f"Training Complete. WAPE: {wape:.4f} | Financial Cost: ${cost:,.2f}")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.prophet_model, "models/prophet_base.pkl")
        joblib.dump(self.lgbm_model, "models/lgbm_residual.pkl")

# Simulation execution block
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=365, freq="D")
    base_volume = 50000 + (np.sin(np.arange(365) * 2 * np.pi / 7) * 10000) # Weekly seasonality
    
    df = pd.DataFrame({'ds': dates, 'y': base_volume + np.random.normal(0, 2000, 365)})
    
    forecaster = HybridLiquidityForecaster()
    forecaster.train_hybrid_model(df)
