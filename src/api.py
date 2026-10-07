import os
import joblib
import psycopg2
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="Hybrid Liquidity & Demand Forecaster API", version="2.0.0")

# --- PROMETHEUS METRIC REGISTRY ---
REQUEST_COUNT = Counter("http_requests_total", "Total incoming HTTP requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency in seconds", ["endpoint"])
LIQUIDITY_FORECAST_GAUGE = Gauge("fintech_forecast_volume_irr", "Latest forecasted clearinghouse volume in Billion IRR")
DARKSTORE_DEMAND_GAUGE = Gauge("qcommerce_forecast_units", "Latest forecasted darkstore SKU depletion units")

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

def log_inference_to_db(domain: str, timestamp: str, val: float, yhat: float):
    """Asynchronously logs live API forecasts to PostgreSQL using the correct schema."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "postgres_dw"),
            database=os.getenv("DB_NAME", "liquidity_dw"),
            user=os.getenv("DB_USER", "dw_admin"),
            password=os.getenv("DB_PASSWORD", "dw_password")
        )
        cursor = conn.cursor()
        
        if domain == "FINTECH":
            cursor.execute("""
                INSERT INTO fintech_liquidity_forecast (settlement_time, forecasted_volume_irr, prophet_trend_irr)
                VALUES (%s, %s, %s)
                ON CONFLICT (settlement_time) DO UPDATE 
                SET forecasted_volume_irr = EXCLUDED.forecasted_volume_irr,
                    prophet_trend_irr = EXCLUDED.prophet_trend_irr;
            """, (timestamp, val, yhat))
            
        elif domain == "QCOMMERCE":
            # Recreate the risk score logic from the training script
            risk_score = min(1.0, val / 45.0)
            cursor.execute("""
                INSERT INTO qcommerce_depletion_forecast (forecast_time, predicted_units, prophet_baseline, stockout_risk_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (forecast_time) DO UPDATE 
                SET predicted_units = EXCLUDED.predicted_units,
                    prophet_baseline = EXCLUDED.prophet_baseline,
                    stockout_risk_score = EXCLUDED.stockout_risk_score;
            """, (timestamp, val, yhat, risk_score))
            
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"DB Logging Error: {e}", flush=True)


@app.get("/metrics")
def get_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/forecast/fintech-liquidity")
def forecast_fintech(req: ForecastRequest, background_tasks: BackgroundTasks):
    with REQUEST_LATENCY.labels(endpoint="/forecast/fintech-liquidity").time():
        if "fintech" not in models:
            load_artifacts()
            if "fintech" not in models:
                REQUEST_COUNT.labels(endpoint="/forecast/fintech-liquidity", status="503").inc()
                raise HTTPException(status_code=503, detail="Model not trained.")

        try:
            target_time = pd.to_datetime(req.timestamp)
            df_target = pd.DataFrame({'ds': [target_time]})
            pred = models["fintech"].predict(df_target)
            
            val = float(pred['hybrid_forecast'].iloc[0])
            yhat = float(pred['yhat'].iloc[0])
            
            LIQUIDITY_FORECAST_GAUGE.set(val)
            REQUEST_COUNT.labels(endpoint="/forecast/fintech-liquidity", status="200").inc()
            
            background_tasks.add_task(log_inference_to_db, "FINTECH", req.timestamp, val, yhat)
            
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
def forecast_qcommerce(req: ForecastRequest, background_tasks: BackgroundTasks):
    with REQUEST_LATENCY.labels(endpoint="/forecast/qcommerce-demand").time():
        if "qcommerce" not in models:
            load_artifacts()
            if "qcommerce" not in models:
                REQUEST_COUNT.labels(endpoint="/forecast/qcommerce-demand", status="503").inc()
                raise HTTPException(status_code=503, detail="Model not trained.")

        try:
            target_time = pd.to_datetime(req.timestamp)
            df_target = pd.DataFrame({'ds': [target_time]})
            pred = models["qcommerce"].predict(df_target)
            
            val = float(pred['hybrid_forecast'].iloc[0])
            yhat = float(pred['yhat'].iloc[0])
            
            DARKSTORE_DEMAND_GAUGE.set(val)
            REQUEST_COUNT.labels(endpoint="/forecast/qcommerce-demand", status="200").inc()
            
            background_tasks.add_task(log_inference_to_db, "QCOMMERCE", req.timestamp, val, yhat)
            
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
    return {"status": "HEALTHY", "loaded_models": list(models.keys()), "time": datetime.utcnow().isoformat()}
