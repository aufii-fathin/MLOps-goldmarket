<div align="center">
<h1>Gold Market Intelligence</h1>
<h3>Adaptive Forecasting & Risk Monitoring System</h3>
</div>

<div align="center">

![Python](https://img.shields.io/badge/python-3.10+-blue)
![MLflow](https://img.shields.io/badge/MLflow-Model%20Registry-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![CI/CD](https://img.shields.io/badge/GitHub%20Actions-CI/CD-black)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-green)

Production-ready MLOps system for forecasting gold prices and monitoring market risk using time-series modeling, drift detection, continual learning, and dataset versioning.

</div>

## Table of Contents

- [Overview](#overview)
- [Machine Learning Tasks](#machine-learning-tasks)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Running with Docker Compose](#running-with-docker-compose)
- [Running with GitHub Codespaces](#running-with-github-codespaces)
- [Data Ingestion & Preprocessing](#data-ingestion--preprocessing)
- [Data Versioning with DVC](#data-versioning-with-dvc)
- [Model Versioning & Registry](#model-versioning--registry)
- [Model Serving](#model-serving)
- [Horizontal Scaling](#horizontal-scaling)
- [Tech Stack](#tech-stack)
- [Contributors](#contributors)
- [License](#license)


## Overview

Gold Market Intelligence is an end-to-end MLOps project designed to forecast gold prices and classify market risk levels within a production-ready machine learning pipeline.

The system integrates:

- Data ingestion from external financial APIs  
- Feature engineering for time-series forecasting  
- Model training and evaluation  
- Drift monitoring and continual learning  
- Model registry with MLflow  
- Dataset versioning with DVC  
- REST API deployment with FastAPI  
- CI/CD automation with GitHub Actions
  
## Machine Learning Tasks

### 1. Time-Series Regression

- Objective: Forecast gold prices 7 days ahead  
- Validation: Time-based split & rolling window backtesting  
- Metrics: MAE, RMSE, MAPE, Mean Directional Accuracy  

### 2. Risk Classification

- Objective: Classify market condition into Low, Medium, High Risk  
- Labeling based on volatility distribution  
- Metric: F1-score (focus on High Risk class)


## System Architecture

### Data Engineering Layer

- Daily ingestion from financial APIs  
- Incremental raw data update  
- Schema validation and anomaly checks  
- Feature engineering  
- Leakage-safe preprocessing

### Machine Learning Layer

- Rolling window training  
- Time-based validation  
- Backtesting evaluation  
- MLflow model registry

### Monitoring Layer

- Performance monitoring  
- Drift detection  
- Trigger-based retraining

### Deployment Layer

- FastAPI serving  
- Docker containerization  
- Horizontal scaling with replicas  
- GitHub Actions CI/CD


## Project Structure

```bash
gold-market-mlops/
│
├── api/
├── configs/
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
├── notebooks/
├── src/
│   ├── data/
│   ├── models/
│   ├── monitoring/
│   ├── registry/
│   └── retraining/
│
├── .dvc/
├── README.md
└── requirements.txt
```

## Running with Docker Compose
 
Run the entire system — PostgreSQL, MLflow Server, and API Service (3 replicas) — with a single command:
 
```bash
docker compose up -d --build
```
 
Verify all containers are running:
 
```bash
docker compose ps
```
 
| Service | URL |
|---|---|
| API Service (replica 1) | http://localhost:8000 |
| API Service (replica 2) | http://localhost:8001 |
| API Service (replica 3) | http://localhost:8002 |
| API Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
 
To stop all services:
 
```bash
docker compose down
```
 
> **Codespaces users:** Access services via the forwarded URLs in the **Ports** tab. Add port `5000` manually if it doesn't appear, then set visibility to **Public**.
 
## Running with GitHub Codespaces

1. Open repository on GitHub
2. Click **Code → Codespaces → Create Codespace on main**
3. Wait until environment setup is complete
4. Set environment variables:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
```

5. Run Docker Compose:

```bash
docker compose up -d --build
```

## Data Ingestion & Preprocessing

### 1. Raw Data Ingestion

Fetch latest market datasets:

```bash
python src/data/ingestion.py
```

Generated files:

```bash
data/raw/gold_prices.csv
data/raw/oil_prices.csv
data/raw/macro_fred.csv
```

### 2. Preprocessing

```bash
python src/data/preprocess_gold.py
python src/data/preprocess_oil.py
python src/data/preprocess_fred.py
python src/data/preprocess_merge.py
```

Final merged dataset:

```bash
data/processed/market_dataset.csv
```

### Notes

* Ingestion supports incremental updates
* New records are appended automatically
* Duplicate dates are removed
* Suitable for continual learning workflow


## Data Versioning with DVC

This project uses **DVC (Data Version Control)** to manage dataset versions without storing large CSV files directly inside Git.

### Tracked Dataset

```bash
data/raw/
├── gold_prices.csv
├── oil_prices.csv
└── macro_fred.csv
```

Tracked through:

```bash
data/raw.dvc
```

### Initial Dataset Tracking

```bash
dvc init
dvc add data/raw
git add .dvc .dvcignore data/raw.dvc .gitignore
git commit -m "feat(data): track raw datasets with DVC"
```

### Update Dataset Version

```bash
python src/data/ingestion.py
dvc add data/raw
git add data/raw.dvc
git commit -m "feat(data): update raw datasets with new records"
```

### Compare Dataset Versions

```bash
dvc diff HEAD~1 HEAD
```

## Model Versioning & Registry

This project trains two model families — **sklearn (Linear Regression)** and **ARIMA** — and registers both to MLflow Model Registry per horizon (+1, +3, +5, +7 days).

### Training

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/models/train.py
python src/models/train_arima.py
```

### Registration & Promotion

```bash
python src/registry/transition_stage.py
```

### Alias Strategy

| Alias | Model | Reason |
|---|---|---|
| `@production` | Linear Regression | Compatible with real-time feature-based serving |
| `@staging` | ARIMA | Lower RMSE but requires static series input, not suitable for real-time serving |

### Registered Models

| Model Name | Production Version | Staging Version |
|---|---|---|
| gold-close-model-h1 | Linear Regression (v2) | ARIMA (v1) |
| gold-close-model-h3 | Linear Regression (v2) | ARIMA (v1) |
| gold-close-model-h5 | Linear Regression (v2) | ARIMA (v1) |
| gold-close-model-h7 | Linear Regression (v2) | ARIMA (v1) |


## Model Serving

### 1. Serve via FastAPI (Recommended)

The API loads models directly from MLflow Registry at startup:

```bash
curl http://localhost:8000/predict
```

Example response:

```json
{
  "current_date": "2026-05-22",
  "current_price": 4515.0,
  "model": "Linear Regression @production",
  "predictions": {
    "h1": {"horizon_days": 1, "forecast_date": "2026-05-23", "predicted_close": 4532.82},
    "h3": {"horizon_days": 3, "forecast_date": "2026-05-25", "predicted_close": 4559.48},
    "h5": {"horizon_days": 5, "forecast_date": "2026-05-27", "predicted_close": 4566.12},
    "h7": {"horizon_days": 7, "forecast_date": "2026-05-29", "predicted_close": 4573.75}
  }
}
```

### Available Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API status |
| `/health` | GET | Health check |
| `/predict` | GET | Gold price predictions for all horizons |
| `/models/info` | GET | Registered model info |
| `/test-mlflow` | GET | Test MLflow connectivity |

### 2. Serve via MLflow Models Serve (Local)

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export PATH=$PATH:/home/codespace/.python/current/bin
python -m mlflow models serve -m "models:/gold-close-model-h1@production" --host 0.0.0.0 --port 7000 --no-conda
```

Test endpoint:

```bash
curl http://localhost:7000/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_records": [{"Close": 4515.0, "return_1d": 0.001, "lag_1": 4510.0, "lag_2": 4505.0, "lag_3": 4500.0, "lag_5": 4490.0, "lag_7": 4480.0, "lag_10": 4470.0, "rolling_mean_7": 4500.0, "rolling_std_7": 15.0, "volatility_20": 0.008}]}'
```


## Horizontal Scaling

The API service is configured with 3 replicas to simulate horizontal scaling:

```yaml
deploy:
  replicas: 3
```

To scale up or down dynamically:

```bash
docker compose up -d --scale api-service=5
```

Verify replicas are running:

```bash
docker compose ps
```

All replicas share the same MLflow tracking server and model registry, ensuring consistent model serving across instances.


## Tech Stack

* Python
* Pandas / NumPy / Scikit-learn
* XGBoost / LightGBM
* MLflow
* DVC
* FastAPI
* Docker
* GitHub Actions


## Contributors

* Aufii Fathin Nabila


## License

MIT License
