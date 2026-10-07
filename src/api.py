import os
import joblib
import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

app = FastAPI(title="Hybrid Liquidity & Demand Forecaster API", version="2.1.0")

# --- PROMETHEUS METRIC REGISTRY ---
REQUEST_COUNT = Counter("http_requests_total", "Total incoming HTTP requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("http_request_duration_seconds", "HTTP request latency in seconds", ["endpoint"])
LIQUIDITY_FORECAST_GAUGE = Gauge("fintech_forecast_volume_irr", "Latest forecasted clearinghouse volume in Billion IRR")
DARKSTORE_DEMAND_GAUGE = Gauge("qcommerce_forecast_units", "Latest forecasted darkstore SKU depletion units")

models = {}
risk_params = {"base_sigma": 28.5}

def load_artifacts():
    try:
        if os.path.exists("models/fintech_engine.joblib"):
            models["fintech"] = joblib.load("models/fintech_engine.joblib")
        if os.path.exists("models/qcommerce_engine.joblib"):
            models["qcommerce"] = joblib.load("models/qcommerce_engine.joblib")
        if os.path.exists("models/fintech_risk_params.joblib"):
            risk_params.update(joblib.load("models/fintech_risk_params.joblib"))
    except Exception as e:
        print(f"Artifact loading error: {e}")

load_artifacts()

class ForecastRequest(BaseModel):
    timestamp: str = Field(..., example="2026-09-28 10:00:00")

def compute_realtime_var_cvar(timestamp: pd.Timestamp, val: float):
    hour = timestamp.hour
    dayofweek = timestamp.dayofweek
    
    paya_mult = 1.85 if (hour in [3, 10, 13, 17] and dayofweek != 4) else 1.0
    sat_mult = 2.30 if (dayofweek == 5 and hour in [3, 10]) else 1.0
    regime_mult = max(paya_mult, sat_mult)
    
    var_99 = round((2.326 * risk_params["base_sigma"] * regime_mult) + (0.05 * val), 2)
    cvar_99 = round(var_99 * 1.28, 2)
    return var_99, cvar_99

def log_fintech_inference(timestamp: str, val: float, yhat: float, var_99: float, cvar_99: float):
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "postgres_dw"),
            database=os.getenv("DB_NAME", "liquidity_dw"),
            user=os.getenv("DB_USER", "dw_admin"),
            password=os.getenv("DB_PASSWORD", "dw_password")
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO fintech_liquidity_forecast (
                settlement_time, forecasted_volume_irr, prophet_trend_irr, 
                recommended_var99_buffer, cvar_tail_risk_irr
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (settlement_time) DO UPDATE 
            SET forecasted_volume_irr = EXCLUDED.forecasted_volume_irr,
                prophet_trend_irr = EXCLUDED.prophet_trend_irr,
                recommended_var99_buffer = EXCLUDED.recommended_var99_buffer,
                cvar_tail_risk_irr = EXCLUDED.cvar_tail_risk_irr;
        """, (timestamp, val, yhat, var_99, cvar_99))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Fintech DB Logging Error: {e}", flush=True)

def log_qcommerce_inference(timestamp: str, val: float, yhat: float, risk_score: float):
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "postgres_dw"),
            database=os.getenv("DB_NAME", "liquidity_dw"),
            user=os.getenv("DB_USER", "dw_admin"),
            password=os.getenv("DB_PASSWORD", "dw_password")
        )
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO qcommerce_depletion_forecast (forecast_time, predicted_units, prophet_baseline, stockout_risk_score)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (forecast_time) DO UPDATE 
            SET predicted_units = EXCLUDED.predicted_units,
                prophet_baseline = EXCLUDED.prophet_baseline,
                stockout_risk_score = EXCLUDED.stockout_risk_score;
        """, (timestamp, val, yhat, risk_score))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Q-Commerce DB Logging Error: {e}", flush=True)

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
                raise HTTPException(status_code=503, detail="FinTech model artifacts not loaded.")

        try:
            target_time = pd.to_datetime(req.timestamp)
            df_target = pd.DataFrame({'ds': [target_time]})
            pred = models["fintech"].predict(df_target)
            
            val = float(pred['hybrid_forecast'].iloc[0])
            yhat = float(pred['yhat'].iloc[0])
            var_99, cvar_99 = compute_realtime_var_cvar(target_time, val)
            
            LIQUIDITY_FORECAST_GAUGE.set(val)
            REQUEST_COUNT.labels(endpoint="/forecast/fintech-liquidity", status="200").inc()
            
            background_tasks.add_task(log_fintech_inference, req.timestamp, val, yhat, var_99, cvar_99)
            
            return {
                "domain": "FINTECH_INTERBANK_CLEARING",
                "timestamp": req.timestamp,
                "forecasted_volume_irr_billions": round(val, 2),
                "prophet_structural_baseline": round(yhat, 2),
                "recommended_var99_buffer": var_99,
                "cvar_tail_risk_irr": cvar_99,
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
                raise HTTPException(status_code=503, detail="Q-Commerce model artifacts not loaded.")

        try:
            target_time = pd.to_datetime(req.timestamp)
            df_target = pd.DataFrame({'ds': [target_time]})
            pred = models["qcommerce"].predict(df_target)
            
            val = float(pred['hybrid_forecast'].iloc[0])
            yhat = float(pred['yhat'].iloc[0])
            
            # Logistic stockout hazard function
            risk_score = round(float(np.clip(1.0 / (1.0 + np.exp(-0.25 * (val - 22.0))), 0.01, 0.99)), 4)
            
            # Operational triggers
            reorder_flag = bool(val >= 22.0 or risk_score >= 0.55)
            urgency = "CRITICAL_STOCKOUT_RISK" if val >= 30.0 else ("ELEVATED_DEPLETION" if val >= 20.0 else "OPTIMAL_INVENTORY")
            replenishment_units = int(np.ceil(val * 1.5 + 15)) if reorder_flag else 0
            
            DARKSTORE_DEMAND_GAUGE.set(val)
            REQUEST_COUNT.labels(endpoint="/forecast/qcommerce-demand", status="200").inc()
            
            background_tasks.add_task(log_qcommerce_inference, req.timestamp, val, yhat, risk_score)
            
            return {
                "domain": "QCOMMERCE_DARKSTORE_INVENTORY",
                "timestamp": req.timestamp,
                "predicted_depletion_units": round(val, 1),
                "prophet_baseline_units": round(yhat, 1),
                "stockout_risk_score": risk_score,
                "reorder_flag": reorder_flag,
                "recommended_replenishment_batch": replenishment_units,
                "operational_urgency": urgency
            }
        except Exception as e:
            REQUEST_COUNT.labels(endpoint="/forecast/qcommerce-demand", status="500").inc()
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "HEALTHY", "loaded_models": list(models.keys()), "time": datetime.utcnow().isoformat()}
