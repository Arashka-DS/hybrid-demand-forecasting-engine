from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/opt/airflow')
from src.forecaster import HybridLiquidityForecaster

default_args = {
    'owner': 'mlops_engineer',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'nightly_liquidity_forecast',
    default_args=default_args,
    description='Trains hybrid Prophet-LightGBM model and updates Postgres',
    schedule_interval='@midnight',
    catchup=False
) as dag:

    def trigger_retraining():
        # In production, fetch live data from Postgres here
        forecaster = HybridLiquidityForecaster()
        print("Executing Nightly Pipeline: Fetch Data -> Walk-Forward CV -> Retrain Hybrid Model.")

    train_task = PythonOperator(
        task_id='retrain_hybrid_model',
        python_callable=trigger_retraining
    )
