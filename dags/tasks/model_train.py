import os
from airflow.decorators import task


@task.virtualenv(
    task_id="get_data",
    requirements=["pandas"],
    pip_install_options=["--no-cache-dir"]
)
def get_data():
    import os
    from pathlib import Path
    import pandas as pd

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")

    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "target": [0, 1, 0, 1, 0]
    })
    data_path = Path(AIRFLOW_HOME) / "dags/data/raw/data.csv"
    data.to_csv(data_path, index=False)

    return str(data_path)

@task.virtualenv(
    task_id="preprocess_data",
    multiple_outputs=True,
    requirements=["pandas", "scikit-learn"],
    pip_install_options=["--no-cache-dir"]
)
def preprocess_data(data_path):
    import os
    from pathlib import Path
    import pandas as pd
    from sklearn.model_selection import train_test_split

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    DATA_DIR = Path(AIRFLOW_HOME) / "dags/data/"

    data = pd.read_csv(data_path)    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    (DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
    train_data_path = DATA_DIR / "processed/train_data.csv"
    test_data_path = DATA_DIR / "processed/test_data.csv"
    train_data.to_csv(train_data_path, index=False)
    test_data.to_csv(test_data_path, index=False)

    return {"train": str(train_data_path), "test": str(test_data_path)}

@task.virtualenv(
    task_id="train_model",
    requirements=["pandas", "scikit-learn"],
    pip_install_options=["--no-cache-dir"],
    multiple_outputs=True,
)
def train_model(train_data_path):    
    import os
    from pathlib import Path
    import pickle
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    AIRFLOW_HOME = os.getenv("AIRFLOW_HOME", "/opt/airflow")
    WORKING_DIR = Path(AIRFLOW_HOME) / "dags/"

    model = LogisticRegression()
    train_data = pd.read_csv(train_data_path)
    model.fit(train_data[["feature1", "feature2"]], train_data["target"])
    predictions = model.predict(train_data[["feature1", "feature2"]])
    accuracy = accuracy_score(train_data["target"], predictions)
    print(f"Training accuracy: {accuracy}")

    (WORKING_DIR / "model").mkdir(parents=True, exist_ok=True)
    model_path = WORKING_DIR / f"model/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return {"model": str(model_path), "accuracy": accuracy}

@task.virtualenv(
    task_id="evaluate_model",
    requirements=["pandas", "scikit-learn"],
    pip_install_options=["--no-cache-dir"],
    multiple_outputs=True,
)
def evaluate_model(model_path, test_data_path):
    import pickle
    import pandas as pd
    from sklearn.metrics import accuracy_score

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    test_data = pd.read_csv(test_data_path)
    predictions = model.predict(test_data[["feature1", "feature2"]])
    accuracy = accuracy_score(test_data["target"], predictions)
    print(f"Evaluation accuracy: {accuracy}")
    return {"accuracy": accuracy}


