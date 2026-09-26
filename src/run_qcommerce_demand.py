import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.forecaster import HybridProphetLGBM

def generate_dark_store_data(days=60):
    """Simulates hyper-local FMCG demand (e.g., milk, snacks) for a single Dark Store."""
    dates = pd.date_range(start="2026-07-01", periods=days*24, freq="h")
    # Base velocity
    demand = np.random.poisson(lam=15, size=len(dates))
    df = pd.DataFrame({'ds': dates, 'y': demand})
    
    # Inject Q-Commerce operational realities: 6 PM to 11 PM evening spikes
    evening_mask = df['ds'].dt.hour.isin([18, 19, 20, 21, 22, 23])
    df.loc[evening_mask, 'y'] += np.random.poisson(lam=35, size=evening_mask.sum())
    
    # Weekend impulse buying surges
    weekend_mask = df['ds'].dt.dayofweek.isin([5, 6])
    df.loc[weekend_mask, 'y'] = (df.loc[weekend_mask, 'y'] * 1.3).astype(int)
    
    return df

if __name__ == "__main__":
    print("Initializing Q-Commerce Dark Store SKU Depletion Forecaster...")
    df_sku = generate_dark_store_data()
    
    # Train-Test Split (Last 7 days for holdout)
    train = df_sku.iloc[:-24*7]
    test = df_sku.iloc[-24*7:]
    
    engine = HybridProphetLGBM()
    engine.fit(train)
    
    forecast_df = engine.predict(test)
    
    wape = engine.evaluate_wape(forecast_df['y'].values, forecast_df['hybrid_forecast'].values)
    print(f"Dark Store SKU Forecast WAPE: {wape:.2%}")
    print("Action: Trigger automated replenishment routes for high-velocity SKUs before 4 PM to prevent stockouts.")
