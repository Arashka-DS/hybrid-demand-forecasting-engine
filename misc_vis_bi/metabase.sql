-- CARD 1: FinTech Interbank Clearing - Forecasted Volume vs Regulatory VaR99 Buffer (Time-Series Area Chart)
SELECT 
    settlement_time,
    forecasted_volume_irr,
    prophet_trend_irr,
    (forecasted_volume_irr + recommended_var99_buffer) AS total_capital_adequacy_ceiling
FROM fintech_liquidity_forecast
WHERE settlement_time >= NOW() - INTERVAL '7 days'
ORDER BY settlement_time ASC;

-- CARD 2: Saturday Liquidity Squeeze Detector (Table / Highlight Card)
-- Identifies critical Paya settlement cycles where volume spikes post-weekend
SELECT 
    settlement_time,
    forecasted_volume_irr,
    recommended_var99_buffer,
    cvar_tail_risk_irr
FROM fintech_liquidity_forecast
WHERE EXTRACT(DOW FROM settlement_time) = 6 -- Saturday
  AND EXTRACT(HOUR FROM settlement_time) IN (3, 10)
ORDER BY forecasted_volume_irr DESC
LIMIT 10;

-- CARD 3: Darkstore Stockout Hazard Monitor (Gauge / Progress Card)
-- Highlights SKUs reaching dangerous inventory burn velocity (> 40 units/hour)
SELECT 
    forecast_time,
    predicted_units,
    stockout_risk_score,
    CASE 
        WHEN stockout_risk_score >= 0.85 THEN '🚨 CRITICAL: Stockout Imminent'
        WHEN stockout_risk_score >= 0.60 THEN '⚠️ WARNING: High Depletion'
        ELSE '✅ NORMAL: Stock Sufficient'
    END AS operational_status
FROM qcommerce_depletion_forecast
ORDER BY forecast_time DESC
LIMIT 20;

-- CARD 4: Overnight Repo Penalty Cost Avoidance (Scalar Metric)
-- Calculates hypothetical central bank penalty fees (34% APR) saved via algorithmic sizing
SELECT 
    ROUND(SUM(recommended_var99_buffer * 0.34 / 365.0), 2) AS estimated_daily_penalty_cost_prevented_irr_billions
FROM fintech_liquidity_forecast;
