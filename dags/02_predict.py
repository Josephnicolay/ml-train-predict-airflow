import os
import pytz
from datetime import datetime
from airflow.decorators import dag
from tasks.predict import preprocess_data, predict
from utils.monitoring.callbacks import success_email, failure_email

TZ = os.getenv("TZ", "UTC")

@dag(
    dag_id="model_predict",   
    start_date=datetime(2026, 1, 1, tzinfo=pytz.timezone(TZ)),
    schedule_interval=None,
    # on_failure_callback = lambda context: failure_email(context),
    # on_success_callback = lambda context: success_email(context)
)
def dag_model_predict():
    data = preprocess_data()
    predict(data)

dag_model_predict()