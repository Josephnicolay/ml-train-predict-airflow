# ML Train & Predict with Airflow

## Introduction

Small MLOps project that trains and serves a scikit-learn model using Apache Airflow. The model predicts **nuclear mass excess** (`MassExcess`) from the IAEA AME2020 atomic mass table, using engineered physics-inspired features and a `RandomForestRegressor` pipeline.

Two DAGs, both triggered manually:

- **train**: download data → preprocess → train → evaluate, saving `model.pkl`.
- **predict**: preprocess raw data → batch predict with the saved model → `predictions.csv`.

## Tools

- [Apache Airflow](https://airflow.apache.org/) 2.9.3 (TaskFlow API, `@task.virtualenv`)
- scikit-learn, pandas, numpy (installed per-task in isolated virtualenvs)
- Docker Compose (CeleryExecutor stack: Postgres, Redis, Flower)
- cloudpickle (model serialization)

## Project structure

```
.
├── docker-compose.yaml     # Airflow local stack
├── .env.example            # Template for required env variables
├── dags/
│   ├── 01_model_train.py   # DAG: model_train
│   ├── 02_predict.py       # DAG: model_predict
│   ├── tasks/              # TaskFlow task implementations
│   │   ├── model_train.py  #   get_data, preprocess, train, evaluate
│   │   └── predict.py      #   preprocess, predict
│   ├── utils/monitoring/   # Email success/failure callbacks
│   ├── data/
│   │   ├── raw/            # Downloaded AME2020 source data
│   │   └── processed/      # Train/test splits, predictions
│   └── model/model.pkl     # Trained model artifact
└── notebooks/              # Exploratory experiments
```

## DAGs

| DAG | ID | Tasks | Output |
|---|---|---|---|
| Train | `model_train` | `get_data` → `preprocess_data` → `train_model` → `evaluate_model` | `dags/model/model.pkl`, RMSE metrics |
| Predict | `model_predict` | `preprocess_data` → `predict` | `dags/data/processed/predictions.csv` |

Both DAGs are `schedule_interval=None` (manual trigger) and run each task in an isolated virtualenv with its own pip requirements.

## Getting started

**Required:** copy the env template and fill in your values — the project won't run locally without it:

```bash
cp .env.example .env
```

| Variable | Purpose |
|---|---|
| `AIRFLOW_UID` | Host user ID for the containers (Linux: use your `id -u`) |
| `TZ` / `AIRFLOW__*_TIMEZONE` | Timezone used by Airflow and task timestamps |
| `AIRFLOW__SMTP__SMTP_USER` / `AIRFLOW__SMTP__SMTP_PASSWORD` | Gmail credentials for the email callbacks (use an app password) |

Then start Airflow:

```bash
docker compose up -d
```

Open the Airflow UI at http://localhost:8080 (default: `airflow` / `airflow`) and trigger `model_train` first, then `model_predict`.
