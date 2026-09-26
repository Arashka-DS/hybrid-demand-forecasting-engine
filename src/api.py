import os
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="Hybrid Liquidity & Demand Forecaster API", version="2.0.0")

# --- PROMETHEUS METRIC REGISTRY ---
REQUEST_COUNT = Counter("http_requests_total", "Total incoming HTTP requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency in seconds", ["endpoint"])
LIQUIDITY_FORECAST_GAUGE = Gauge("fintech_forecast_volume_irr", "Latest forecasted clearinghouse volume in Billion IRR")
DARKSTORE_DEMAND_GAUGE = Gauge("qcommerce_forecast_units", "Latest forecasted darkstore SKU depletion units")

# Model singletons
models = {}

def load_artifacts():
    try:
        if os.path.exists("models/fintech_engine.joblib"):
            models["fintech"] = joblib.load("models/fintech_engine.joblib")
        if os.path.exists("models/qcommerce_engine.joblib"):
            models["qcommerce"] = joblib.load("models/qcommerce_engine.joblib")
    except Exception as e:
        print(f"Artifact loading error: {e}")

load_artifacts()

class ForecastRequest(BaseModel):
    timestamp: str = Field(..., example="2026-09-28 10:00:00")

@app.get("/metrics")
def get_metrics():
    """Prometheus metrics scrape target."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/forecast/fintech-liquidity")
def forecast_fintech(req: ForecastRequest):
    with REQUEST_LATENCY.labels(endpoint="/forecast/fintech-liquidity").time():
        if "fintech" not in models:
            load_artifacts()
            if "fintech" not in models:
                REQUEST_COUNT.labels(endpoint="/forecast/fintech-liquidity", status="503").inc()
                raise HTTPException(status_code=503, detail="FinTech model not trained yet. Run src/run_fintech_liquidity.py first.")

        try:
            target_time = pd.to_datetime(req.timestamp)
            df_target = pd.DataFrame({'ds': [target_time]})
            pred = models["fintech"].predict(df_target)
            val = float(pred['hybrid_forecast'].iloc[0])
            
            LIQUIDITY_FORECAST_GAUGE.set(val)
            REQUEST_COUNT.labels(endpoint="/forecast/fintech-liquidity", status="200").inc()
            
            return {
                "domain": "FINTECH_INTERBANK_CLEARING",
                "timestamp": req.timestamp,
                "forecasted_volume_irr_billions": round(val, 2),
                "is_paya_cycle_window": bool(target_time.hour in [3, 10, 13, 17])
            }
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/forecast/fintech-liquidity", status="500").inc()
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast/qcommerce-demand")
def forecast_qcommerce(req: ForecastRequest):
    with REQUEST_LATENCY.labels(endpoint="/forecast/qcommerce-demand").time():
        if "qcommerce" not in models:
            load_artifacts()
            if "qcommerce" not in models:
                REQUEST_COUNT.labels(endpoint="/forecast/qcommerce-demand", status="503").inc()
                raise HTTPException(status_code=503, detail="Q-Commerce model not trained yet. Run src/run_qcommerce_demand.py first.")

        try:
            target_time = pd.to_datetime(req.timestamp)
            df_target = pd.DataFrame({'ds': [target_time]})
            pred = models["qcommerce"].predict(df_target)
            val = float(pred['hybrid_forecast'].iloc[0])
            
            DARKSTORE_DEMAND_GAUGE.set(val)
            REQUEST_COUNT.labels(endpoint="/forecast/qcommerce-demand", status="200").inc()
            
            return {
                "domain": "QCOMMERCE_DARKSTORE_INVENTORY",
                "timestamp": req.timestamp,
                "predicted_depletion_units": round(val, 1),
                "reorder_flag": bool(val > 40.0)
            }
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/forecast/qcommerce-demand", status="500").inc()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {
        "status": "HEALTHY",
        "loaded_models": list(models.keys()),
        "time": datetime.utcnow().isoformat()
    }
