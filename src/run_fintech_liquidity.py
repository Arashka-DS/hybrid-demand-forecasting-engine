import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.forecaster import HybridProphetLGBM

def generate_atm_settlement_data(days=60):
    """Simulates interbank clearinghouse liquidity demands (Satna/Paya cycles)."""
    dates = pd.date_range(start="2026-07-01", periods=days*24, freq="h")
    # Base fiat volume (in Millions IRR)
    demand = np.random.normal(loc=500, scale=50, size=len(dates))
    df = pd.DataFrame({'ds': dates, 'y': demand})
    
    # Inject Salary Day Spikes (28th-30th of the month)
    salary_mask = df['ds'].dt.day.isin([28, 29, 30])
    df.loc[salary_mask, 'y'] += np.random.normal(loc=800, scale=100, size=salary_mask.sum())
    
    return df

if __name__ == "__main__":
    print("Initializing FinTech Liquidity Capital Lockup Forecaster...")
    df_liquidity = generate_atm_settlement_data()
    
    train = df_liquidity.iloc[:-24*7]
    test = df_liquidity.iloc[-24*7:]
    
    engine = HybridProphetLGBM()
    engine.fit(train)
    
    forecast_df = engine.predict(test)
    
    # Calculate 95% Value at Risk (VaR) for capital buffer allocation
    residuals = forecast_df['y'] - forecast_df['hybrid_forecast']
    var_95 = np.percentile(residuals, 95)
    
    wape = engine.evaluate_wape(forecast_df['y'].values, forecast_df['hybrid_forecast'].values)
    print(f"Liquidity Demand Forecast WAPE: {wape:.2%}")
    print(f"Recommended Cash Buffer (VaR 95%): {var_95:.2f} Million IRR")
