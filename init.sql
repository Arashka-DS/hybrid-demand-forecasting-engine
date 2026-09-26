CREATE TABLE IF NOT EXISTS transaction_forecasts (
    forecast_date TIMESTAMP PRIMARY KEY,
    actual_volume NUMERIC,
    forecasted_volume NUMERIC,
    residual_error NUMERIC,
    is_anomaly BOOLEAN,
    horizon VARCHAR(10),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
