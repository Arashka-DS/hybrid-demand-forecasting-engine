# FinTech Liquidity & Demand Forecasting Pipeline

An enterprise-grade MLOps time-series architecture designed to forecast daily fiat liquidity requirements and monitor real-time payment gateway health for BNPL and E-commerce platforms.

## 🏗️ Dual-Layer Architecture
1. **The Executive Layer (Metabase & Batch ML):** A hybrid `Prophet` + `LightGBM` model forecasts $t+1$ to $t+30$ liquidity demand. It minimizes a custom financial cost function (Idle Cash vs. Failed Transactions) and calculates the 95% Value at Risk (VaR). Metrics are visualized via Metabase.
2. **The SRE Layer (Grafana & Real-Time API):** A `FastAPI` endpoint ingests live transactional telemetry. If actual volume drops $-3\sigma$ below the forecasted threshold, the system flags an API gateway outage, immediately visible on the Grafana monitoring dashboard.

## ⚙️ Senior MLOps Features
- **Data Drift Detection:** Integrates `Evidently AI` to monitor statistical shifts in transaction distributions, dynamically altering retraining schedules in Airflow.
- **Walk-Forward Validation:** Evaluated using strict expanding-window cross-validation (`TimeSeriesSplit`) to prevent temporal data leakage.
- **Exogenous Feature Engineering:** Automatically generates rolling lags and boolean flags for macro events (Paydays, Holidays).

## 🚀 Quick Start
1. `docker-compose up -d --build` (Spins up Postgres, Grafana, Metabase, and FastAPI).
2. Run `python src/forecaster.py` to execute the hybrid walk-forward training and generate the HTML Drift Report.
3. Access **Grafana** at `http://localhost:3000` for live API monitoring.
4. Access **Metabase** at `http://localhost:3001` for the Executive level financial dashboard.
