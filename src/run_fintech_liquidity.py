import os
import joblib
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from src.forecaster import HybridProphetLGBM

def generate_interbank_clearing_data(days=90):
    """
    Simulates high-throughput interbank clearinghouse demand (Satna & Paya):
    - Paya 4-cycle daily batch settlements (03:45, 10:45, 13:45, 17:45)
    - Friday closure causing Saturday backlog flush
    - End-of-month salary dispersion shock waves (Days 27-30)
    """
    np.random.seed(1337)
    dates = pd.date_range(start="2026-06-01", periods=days * 24, freq="h")
    
    # Baseline continuous volume (in Billions IRR)
    base_volume = np.random.normal(loc=120.0, scale=15.0, size=len(dates))
    df = pd.DataFrame({'ds': dates, 'y': np.clip(base_volume, 10.0, None)})
    
    hours = df['ds'].dt.hour
    dayofweek = df['ds'].dt.dayofweek # 5=Friday, 6=Saturday in international pandas index
    day = df['ds'].dt.day
    
    # 1. Paya Batch Settlement Cycle Hours (Surges at 03:00, 10:00, 13:00, 17:00)
    paya_mask = hours.isin([3, 10, 13, 17]) & (dayofweek != 4) # Exclude Friday holiday
    df.loc[paya_mask, 'y'] += np.random.normal(loc=350.0, scale=40.0, size=paya_mask.sum())
    
    # 2. Friday Closure (Zero Paya interbank clearing - volume drops significantly)
    friday_mask = (dayofweek == 4)
    df.loc[friday_mask, 'y'] *= 0.25
    
    # 3. Saturday Morning Liquidity Squeeze (Accumulated weekend backlog flushing at 03:00 & 10:00)
    sat_squeeze_mask = (dayofweek == 5) & hours.isin([3, 10])
    df.loc[sat_squeeze_mask, 'y'] += np.random.normal(loc=650.0, scale=60.0, size=sat_squeeze_mask.sum())
    
    # 4. End-of-Month Corporate Payroll Shock (Days 27 to 30)
    salary_mask = day.isin([27, 28, 29, 30]) & hours.isin([10, 11, 12, 13])
    df.loc[salary_mask, 'y'] += np.random.normal(loc=500.0, scale=80.0, size=salary_mask.sum())
    
    return df

def persist_fintech_audit(df_forecast, var_99, cvar_99):
    """Logs liquidity requirements and regulatory capital buffers to PostgreSQL."""
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
                    recommended_var99_buffer = EXCLUDED.recommended_var99_buffer;
            """, (r['ds'], float(r['hybrid_forecast']), float(r['yhat']), float(var_99), float(cvar_99)))
            
        conn.commit()
        cur.close()
        conn.close()
        print("Logged FinTech capital allocation telemetry to PostgreSQL.")
    except Exception as e:
        print(f"Database logging warning: {e}")

if __name__ == "__main__":
    print("Training FinTech Liquidity & Interbank Settlement Forecaster...")
    df_raw = generate_interbank_clearing_data(days=90)
    
    # 7-day holdout
    train = df_raw.iloc[:-24 * 7].copy()
    test = df_raw.iloc[-24 * 7:].copy()
    
    engine = HybridProphetLGBM()
    engine.fit(train[['ds', 'y']])
    
    forecast_df = engine.predict(test[['ds', 'y']], context_history=train.tail(48)[['ds', 'y']])
    
    # Financial Tail-Risk Metrics
    residuals = test['y'].values - forecast_df['hybrid_forecast'].values
    var_99 = np.percentile(residuals, 99)
    cvar_99 = residuals[residuals >= var_99].mean() if len(residuals[residuals >= var_99]) > 0 else var_99
    wape = engine.evaluate_wape(test['y'].values, forecast_df['hybrid_forecast'].values)
    
    print(f"FinTech Settlement WAPE: {wape:.2%}")
    print(f"Regulatory 99% Value at Risk Buffer: {var_99:,.2f} Billion IRR")
    print(f"Expected Shortfall (CVaR tail risk): {cvar_99:,.2f} Billion IRR")
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(engine, "models/fintech_engine.joblib")
    print("Serialized FinTech model to models/fintech_engine.joblib")
    
    persist_fintech_audit(forecast_df, var_99, cvar_99)
