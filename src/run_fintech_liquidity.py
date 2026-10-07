import os
import joblib
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from src.forecaster import HybridProphetLGBM

def generate_interbank_clearing_data(days=90):
    np.random.seed(1337)
    dates = pd.date_range(start="2026-06-01", periods=days * 24, freq="h")
    
    base_volume = np.random.normal(loc=120.0, scale=15.0, size=len(dates))
    df = pd.DataFrame({'ds': dates, 'y': np.clip(base_volume, 10.0, None)})
    
    hours = df['ds'].dt.hour
    dayofweek = df['ds'].dt.dayofweek 
    day = df['ds'].dt.day
    
    # 1. Paya Settlement Hours
    paya_mask = hours.isin([3, 10, 13, 17]) & (dayofweek != 4)
    df.loc[paya_mask, 'y'] += np.random.normal(loc=350.0, scale=40.0, size=paya_mask.sum())
    
    # 2. Friday Holiday Closure
    friday_mask = (dayofweek == 4)
    df.loc[friday_mask, 'y'] *= 0.25
    
    # 3. Saturday Morning Backlog Flush
    sat_squeeze_mask = (dayofweek == 5) & hours.isin([3, 10])
    df.loc[sat_squeeze_mask, 'y'] += np.random.normal(loc=650.0, scale=60.0, size=sat_squeeze_mask.sum())
    
    # 4. End-of-Month Payroll Shock
    salary_mask = day.isin([27, 28, 29, 30]) & hours.isin([10, 11, 12, 13])
    df.loc[salary_mask, 'y'] += np.random.normal(loc=500.0, scale=80.0, size=salary_mask.sum())
    
    return df

def calculate_dynamic_risk(df_forecast: pd.DataFrame, base_sigma: float):
    """Computes regime-dependent, heteroskedastic VaR99 and CVaR99 buffers."""
    hours = df_forecast['ds'].dt.hour
    dayofweek = df_forecast['ds'].dt.dayofweek
    
    # Volatility multiplier based on settlement clearing regime
    paya_mult = np.where(hours.isin([3, 10, 13, 17]) & (dayofweek != 4), 1.85, 1.0)
    sat_mult = np.where((dayofweek == 5) & hours.isin([3, 10]), 2.30, 1.0)
    regime_mult = np.maximum(paya_mult, sat_mult)
    
    # Parametric Gaussian 99% VaR (2.326 * sigma) + proportional liquidity buffer
    var_99 = (2.326 * base_sigma * regime_mult) + (0.05 * df_forecast['hybrid_forecast'].values)
    # Expected Shortfall (CVaR) tail expansion
    cvar_99 = var_99 * 1.28
    
    return np.round(var_99, 2), np.round(cvar_99, 2)

def persist_fintech_audit(df_forecast):
    """Logs dynamic liquidity buffers to PostgreSQL."""
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
            CREATE TABLE IF NOT EXISTS fintech_liquidity_forecast (
                settlement_time TIMESTAMP PRIMARY KEY,
                forecasted_volume_irr NUMERIC(15, 2),
                prophet_trend_irr NUMERIC(15, 2),
                recommended_var99_buffer NUMERIC(15, 2),
                cvar_tail_risk_irr NUMERIC(15, 2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        for _, r in df_forecast.iterrows():
            cur.execute("""
                INSERT INTO fintech_liquidity_forecast (
                    settlement_time, forecasted_volume_irr, prophet_trend_irr, 
                    recommended_var99_buffer, cvar_tail_risk_irr
                )
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (settlement_time) DO UPDATE 
                SET forecasted_volume_irr = EXCLUDED.forecasted_volume_irr,
                    recommended_var99_buffer = EXCLUDED.recommended_var99_buffer,
                    cvar_tail_risk_irr = EXCLUDED.cvar_tail_risk_irr;
            """, (r['ds'], float(r['hybrid_forecast']), float(r['yhat']), float(r['var_99']), float(r['cvar_99'])))
            
        conn.commit()
        cur.close()
        conn.close()
        print("Logged dynamic FinTech capital allocation telemetry to PostgreSQL.")
    except Exception as e:
        print(f"Database logging warning: {e}")

if __name__ == "__main__":
    print("Training FinTech Liquidity & Interbank Settlement Forecaster...")
    df_raw = generate_interbank_clearing_data(days=90)
    
    train = df_raw.iloc[:-24 * 7].copy()
    test = df_raw.iloc[-24 * 7:].copy()
    
    engine = HybridProphetLGBM()
    engine.fit(train[['ds', 'y']])
    
    forecast_df = engine.predict(test[['ds', 'y']], context_history=train.tail(48)[['ds', 'y']])
    
    residuals = test['y'].values - forecast_df['hybrid_forecast'].values
    base_sigma = float(np.std(residuals))
    
    # Calculate unique dynamic buffers per row
    forecast_df['var_99'], forecast_df['cvar_99'] = calculate_dynamic_risk(forecast_df, base_sigma)
    wape = engine.evaluate_wape(test['y'].values, forecast_df['hybrid_forecast'].values)
    
    print(f"FinTech Settlement WAPE: {wape:.2%}")
    print(f"Base Residual Std Dev: {base_sigma:,.2f} Billion IRR")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(engine, "models/fintech_engine.joblib")
    joblib.dump({"base_sigma": base_sigma}, "models/fintech_risk_params.joblib")
    print("Serialized FinTech engine and risk calibrator.")
    
    persist_fintech_audit(forecast_df)
