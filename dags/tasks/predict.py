from airflow.decorators import task

@task.virtualenv
def preprocess_data():
    # Your preprocessing logic here
    return "processed_data"

@task.virtualenv
def predict(processed_data):
    # Your prediction logic here
    print(f"Predictions based on {processed_data}")