# FinTech Liquidity & Transaction Demand Forecaster

A production MLOps pipeline designed to predict fiat liquidity requirements and monitor payment gateway health. This system predicts multi-horizon transaction volumes ($t+1, t+7, t+30$) and alerts operations if real-time volume drops below expected bounds.

## Architectural Highlights
- **Hybrid Modeling:** Uses **Prophet** to capture broad macro-seasonality and trends, while passing the residual errors to **LightGBM** to model non-linear exogenous shocks (Paydays, Holidays, rolling lags).
- **Time-Series Integrity:** Strictly evaluated using `TimeSeriesSplit` (Walk-Forward Validation) to prevent future data leakage.
- **CTO-Level Metrics:** Optimized using $WAPE$ (Weighted Absolute Percentage Error) and a custom business cost function:
  $$Total\_Cost = (\text{Over\_Prediction} \times C_{idle}) + (\text{Under\_Prediction} \times C_{failed})$$
- **Orchestration & DevOps:** Models are retrained nightly via **Apache Airflow**. Inferences are served via a **FastAPI** endpoint that checks live gateway traffic against the forecast to detect $-3\sigma$ outages, automatically alerting a **Grafana** SRE dashboard via **PostgreSQL**.

## Quick Start
1. `docker-compose up -d --build`
2. Run `python src/forecaster.py` to train the initial hybrid models and generate the baseline threshold.
3. Access **Grafana** at `http://localhost:3000` (auto-provisioned to the Postgres Time-Series DB) to monitor the live API health.
4. Send live telemetry to the FastAPI instance at `http://localhost:8000/monitor/gateway` to simulate real-time traffic anomalies.
