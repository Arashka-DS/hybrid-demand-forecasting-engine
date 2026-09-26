# Hybrid Demand & Liquidity Forecasting Engine

An enterprise-grade time-series forecasting platform combining additive structural seasonality (Facebook Prophet) with gradient-boosted residual error modeling (LightGBM). Designed to eliminate data leakage in multi-step predictions while supporting dual high-throughput operational domains: interbank clearinghouse liquidity allocations and darkstore grocery SKU depletion.

## 🏛️ Multi-Domain Execution Prongs

### 1. FinTech Settlement & Liquidity Allocation (`src/run_fintech_liquidity.py`)
* **Core Problem:** Payment clearinghouses and digital wallets face liquidity volatility during automated clearing cycles. Under-provisioning incurs severe central bank overnight penalty interest (34%), while over-provisioning locks up working capital.
* **Domain Realities Modeled:** Four daily Paya batch settlement windows (03:45, 10:45, 13:45, 17:45), the "Saturday Liquidity Squeeze" (accumulated weekend settlement flushes), and month-end payroll shocks.
* **Risk Quant:** Calculates 99% Value at Risk ($VaR_{99\\%}$) and Expected Shortfall ($CVaR$) to establish optimal regulatory capital buffers.

### 2. Q-Commerce Darkstore Depletion (`src/run_qcommerce_demand.py`)
* **Core Problem:** Hyperlocal 15-minute delivery services face stockout censoring. When on-hand inventory drops to zero, recorded sales cease, biasing naive algorithms into under-forecasting replenishment needs.
* **Domain Realities Modeled:** Intraday lunch (12:00–14:00) and dinner (19:00–23:00) order rushes, perishable batch write-offs, and stockout truncation.
* **Operational Metric:** Computes hourly stockout risk scores to automate warehouse cross-docking before inventory hits zero.

## ⚙️ Mathematical Architecture

1. **Stage 1 — Structural Macro Seasonality:** Prophet fits multi-period Fourier terms (daily, weekly) to capture additive macro-trends without lookahead bias.
2. **Stage 2 — Leak-Free Feature Engineering:** Autoregressive lags are derived strictly from Prophet's baseline ($\hat{y}$) and calendar covariates, utilizing a trailing context history buffer to prevent horizon truncation.
3. **Stage 3 — Residual Gradient Boosting:** LightGBM trains on the residual error ($y - \hat{y}$) to learn non-linear calendar interactions and micro-shocks.
4. **Loss Metric:** Evaluated using Weighted Absolute Percentage Error (WAPE) to maintain mathematical stability during zero-volume periods:
   $$\text{WAPE} = \frac{\sum \vert{}y - \hat{y}\vert{}}{\sum y}$$

## 📊 Telemetry & Observability
* **FastAPI Service:** Exposes `/forecast/fintech-liquidity` and `/forecast/qcommerce-demand`.
* **Prometheus:** Scrapes `/metrics` to track request rates, API latency, and real-time forecast gauges.
* **Grafana:** Visualizes SRE performance metrics and service throughput.
* **Metabase:** Renders executive settlement dashboards, liquidity buffer ceilings, and darkstore stockout hazard tables.

## 🚀 Quick Start

1. **Spin up the complete infrastructure:**
   ```bash
   docker-compose up -d --build
   ```
   *(Note: The API container automatically trains both domain models on boot and seeds the PostgreSQL data warehouse).*

2. **Access Platform Dashboards:**
   * **FastAPI Interactive Docs:** `http://localhost:8000/docs`
   * **Prometheus Metrics Target:** `http://localhost:9090`
   * **Grafana SRE Dashboard:** `http://localhost:3001` (admin / admin)
   * **Metabase Executive BI:** `http://localhost:3000`
