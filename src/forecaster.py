import pandas as pd
import numpy as np
from prophet import Prophet
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import joblib
import os

class HybridLiquidityForecaster:
    def __init__(self):
        self.prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        self.lgbm_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5)
        self.cost_idle_cash = 0.02
        self.cost_failed_tx = 5.0

    # ... (Keep existing engineer_features method) ...

    def calculate_business_metrics(self, y_true, y_pred):
        """WAPE, Financial Cost, and 95% Value at Risk (VaR)."""
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)
        
        errors = y_pred - y_true
        total_cost = (np.sum(errors[errors > 0]) * self.cost_idle_cash) + \
                     (np.sum(np.abs(errors[errors < 0])) * self.cost_failed_tx)
        
        # Calculate 95% Historical VaR on the residuals
        # "With 95% confidence, our volume won't drop below the forecast by more than X"
        var_95 = np.percentile(errors, 5) 
        
        return wape, total_cost, var_95

    def check_data_drift(self, reference_data, current_data):
        """Uses Evidently AI to detect statistical distribution shifts in transactions."""
        print("Running Data Drift Analysis...")
        drift_report = Report(metrics=[DataDriftPreset()])
        # We check if the distribution of 'y' (volume) has drifted significantly
        drift_report.run(reference_data=reference_data[['y']], current_data=current_data[['y']])
        
        # Save HTML report for MLOps tracking
        os.makedirs("reports", exist_ok=True)
        drift_report.save_html("reports/volume_drift_report.html")
        
        drift_dict = drift_report.as_dict()
        dataset_drift = drift_dict["metrics"][0]["result"]["dataset_drift"]
        return dataset_drift # Returns True if drift is detected

    def train_hybrid_model(self, data):
        data = self.engineer_features(data)
        
        # Split data for drift monitoring simulation (first 80% reference, last 20% current)
        split_idx = int(len(data) * 0.8)
        ref_data, curr_data = data.iloc[:split_idx], data.iloc[split_idx:]
        
        is_drifting = self.check_data_drift(ref_data, curr_data)
        if is_drifting:
            print("⚠️ WARNING: Data Drift Detected. Model retraining prioritized.")
        
        # ... (Keep existing Prophet + LightGBM Walk-Forward Training code) ...
        
        data['lgbm_residual_pred'] = self.lgbm_model.predict(data[features])
        data['final_forecast'] = data['prophet_base'] + data['lgbm_residual_pred']
        
        wape, cost, var_95 = self.calculate_business_metrics(data['y'], data['final_forecast'])
        print(f"Training Complete. WAPE: {wape:.4f} | Financial Cost: ${cost:,.2f}")
        print(f"Risk Metric - 95% VaR: Maximum expected shortfall is {abs(var_95):,.2f} transactions.")
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.prophet_model, "models/prophet_base.pkl")
        joblib.dump(self.lgbm_model, "models/lgbm_residual.pkl")
