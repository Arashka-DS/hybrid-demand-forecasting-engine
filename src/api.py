from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from prometheus_fastapi_instrumentator import Instrumentator # NEW IMPORT
import pandas as pd
import numpy as np
import psycopg2
import os

app = FastAPI(title="Liquidity & Gateway Monitor API")

# NEW: Instrument the API to expose real-time metrics to Prometheus
Instrumentator().instrument(app).expose(app)

class LiveTransactionData(BaseModel):
    timestamp: str
    actual_volume: float
    forecasted_volume: float

def log_to_postgres(payload, is_anomaly, residual):
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database="liquidity_warehouse", user="ts_admin", password="ts_password"
        )
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO transaction_forecasts (forecast_date, actual_volume, forecasted_volume, residual_error, is_anomaly, horizon)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (payload.timestamp, payload.actual_volume, payload.forecasted_volume, residual, is_anomaly, 't+0'))
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Error: {e}")

@app.post("/monitor/gateway")
def check_gateway_health(data: LiveTransactionData, bg_tasks: BackgroundTasks):
    residual = data.actual_volume - data.forecasted_volume
    historical_std = 2000.0 # In production, pull this dynamically from Redis/Postgres
    
    is_outage_anomaly = False
    status = "System Healthy"
    
    if residual <= (-3 * historical_std):
        is_outage_anomaly = True
        status = "CRITICAL: Gateway Outage Detected"

    bg_tasks.add_task(log_to_postgres, data, is_outage_anomaly, residual)

    return {
        "status": status,
        "residual_error": residual,
        "is_anomaly": is_outage_anomaly
    }
