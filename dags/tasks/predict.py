import os
from airflow.decorators import task

@task.virtualenv(
    task_id="preprocess_data",
    requirements=["pandas"],
    pip_install_options=["--no-cache-dir"],
    multiple_outputs=True
)
def preprocess_data():
    import os
    from pathlib import Path
    import pandas as pd

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    DATA_DIR = Path(AIRFLOW_HOME) / "dags/data"

    data = pd.read_csv(DATA_DIR / "raw/data.csv")

    data_processed = data.copy()
    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    data_processed_path = DATA_DIR / "processed/data_processed.csv"
    data_processed.to_csv(data_processed_path, index=False)

    return {"processed_data": str(data_processed_path)}

@task.virtualenv(
    task_id="predict",
    requirements=["pandas", "scikit-learn", "pytz"],
    pip_install_options=["--no-cache-dir"],
    multiple_outputs=True,
)
def predict(data_path, params=None):
    import os
    from pathlib import Path
    import pickle
    import pandas as pd
    from datetime import datetime
    import pytz

    TZ = os.getenv("TZ")

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    DATA_DIR = Path(AIRFLOW_HOME) / "dags/data"
    MODELS_DIR = Path(AIRFLOW_HOME) / "dags/model"
    
    with open(MODELS_DIR / params["model"], "rb") as f:
        model = pickle.load(f)

    data = pd.read_csv(data_path)
    predictions = model.predict(data[["feature1", "feature2"]])

    predictions_path = DATA_DIR / "processed/predictions.csv"
    predictions_df = pd.DataFrame(predictions, columns=["prediction"])
    predictions_df["date"] = datetime.now(tz=pytz.timezone(TZ)).isoformat()
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Predictions based on {data_path} saved to {predictions_path}")
    return {
        "predictions": str(predictions_path),
        "numberOfPredictions": len(predictions)
    }