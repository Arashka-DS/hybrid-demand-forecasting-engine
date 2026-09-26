import os
import joblib
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from src.forecaster import HybridProphetLGBM

def generate_darkstore_depletion_data(days=90):
    """
    Simulates high-velocity darkstore SKU dynamics:
    - Intraday double-peaks: Lunch (12:00-14:00) & Dinner/Night Rush (19:00-23:00)
    - Weekend surge (Thursday afternoon & Friday in MENA)
    - Censored demand simulation (Stockout zero-truncation)
    """
    np.random.seed(42)
    dates = pd.date_range(start="2026-06-01", periods=days * 24, freq="h")
    
    # 1. Base Latent Demand (True Customer Intent)
    base_demand = np.random.poisson(lam=12, size=len(dates)).astype(float)
    df = pd.DataFrame({'ds': dates, 'true_demand': base_demand})
    
    hours = df['ds'].dt.hour
    dayofweek = df['ds'].dt.dayofweek # 4=Thursday, 5=Friday in Iranian/MENA calendar
    
    # Lunch spike (12:00 - 14:00)
    lunch_mask = hours.isin([12, 13, 14])
    df.loc[lunch_mask, 'true_demand'] += np.random.poisson(lam=22, size=lunch_mask.sum())
    
    # Dinner / Night snack spike (19:00 - 23:00)
    dinner_mask = hours.isin([19, 20, 21, 22, 23])
    df.loc[dinner_mask, 'true_demand'] += np.random.poisson(lam=45, size=dinner_mask.sum())
    
    # Weekend surge
    weekend_mask = dayofweek.isin([4, 5])
    df.loc[weekend_mask, 'true_demand'] *= 1.45
    
    # 2. Simulate Physical Inventory Depletion & Stockout Censoring
    # Assume darkstore gets replenished every morning at 06:00 to 350 units
    inventory = 350
    observed_sales = []
    stockout_flags = []
    
    for idx, row in df.iterrows():
        if row['ds'].hour == 6:
            inventory = 350 # Morning replenishment batch
            
        unfulfilled = max(0.0, row['true_demand'] - inventory)
        actual_sold = min(inventory, row['true_demand'])
        inventory -= actual_sold
        
        observed_sales.append(actual_sold)
        stockout_flags.append(1 if unfulfilled > 0 else 0)
        
    df['y'] = observed_sales
    df['was_stockout'] = stockout_flags
    return df

def persist_qcommerce_audit(df_forecast):
    """Logs stockout hazards and depletion metrics to PostgreSQL for Metabase."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "liquidity_dw"),
            user=os.getenv("DB_USER", "dw_admin"),
            password=os.getenv("DB_PASSWORD", "dw_password"),
            port=5432
        )
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS qcommerce_depletion_forecast (
                forecast_time TIMESTAMP PRIMARY KEY,
                predicted_units NUMERIC(10, 2),
                prophet_baseline NUMERIC(10, 2),
                stockout_risk_score NUMERIC(5, 4),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        for _, r in df_forecast.iterrows():
            # Risk score increases if forecast exceeds typical batch safety margins (>40 units/hr)
            risk = min(1.0, float(r['hybrid_forecast']) / 45.0)
            cur.execute("""
                INSERT INTO qcommerce_depletion_forecast (forecast_time, predicted_units, prophet_baseline, stockout_risk_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (forecast_time) DO UPDATE 
                SET predicted_units = EXCLUDED.predicted_units,
                    stockout_risk_score = EXCLUDED.stockout_risk_score;
            """, (r['ds'], float(r['hybrid_forecast']), float(r['yhat']), risk))
            
        conn.commit()
        cur.close()
        conn.close()
        print("Logged Q-Commerce replenishment telemetry to PostgreSQL.")
    except Exception as e:
        print(f"Database logging warning: {e}")

if __name__ == "__main__":
    print("Training Q-Commerce Darkstore SKU Depletion Forecaster...")
    df_raw = generate_darkstore_depletion_data(days=90)
    
    # 7-day holdout evaluation
    train = df_raw.iloc[:-24 * 7].copy()
    test = df_raw.iloc[-24 * 7:].copy()
    
    engine = HybridProphetLGBM()
    engine.fit(train[['ds', 'y']])
    
    # Forecast passing the trailing context buffer to prevent lag truncation
    forecast_df = engine.predict(test[['ds', 'y']], context_history=train.tail(48)[['ds', 'y']])
    
    wape = engine.evaluate_wape(test['y'].values, forecast_df['hybrid_forecast'].values)
    print(f"Darkstore Hourly SKU Forecast WAPE: {wape:.2%}")
    
    # Persist model
    os.makedirs("models", exist_ok=True)
    joblib.dump(engine, "models/qcommerce_engine.joblib")
    print("Serialized Q-Commerce model to models/qcommerce_engine.joblib")
    
    persist_qcommerce_audit(forecast_df)
