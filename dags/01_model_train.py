import os
import pytz
from datetime import datetime
from airflow.decorators import dag
from tasks.model_train import get_data, preprocess_data, train_model, evaluate_model
from utils.monitoring.callbacks import success_email, failure_email

TZ = os.getenv("TZ", "UTC")

@dag(
    dag_id="model_train",   
    start_date=datetime(2026, 1, 1, tzinfo=pytz.timezone(TZ)),
    schedule_interval=None,
    on_failure_callback = lambda context: failure_email(context),
    on_success_callback = lambda context: success_email(context)
)
def dag_model_train():
    data = get_data()
    processed_data = preprocess_data(data)
    artifacts = train_model(processed_data["train"])
    evaluation = evaluate_model(artifacts["model"], processed_data["test"])

dag_model_train()
