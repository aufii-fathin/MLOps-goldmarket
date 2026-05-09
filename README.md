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
- [Running with GitHub Codespaces](#running-with-github-codespaces)
- [Data Ingestion & Preprocessing](#data-ingestion--preprocessing)
- [Data Versioning with DVC](#data-versioning-with-dvc)
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
- Metrics: MAE, RMSE, RMSE, MAPE, Mean Directional Accuracy  

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
````

## Running with GitHub Codespaces

1. Open repository on GitHub
2. Click **Code → Codespaces → Create Codespace on main**
3. Wait until environment setup is complete

Then run commands directly from cloud terminal.

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

Run ingestion again to fetch latest data:

```bash
python src/data/ingestion.py
```

Then update DVC version:

```bash
dvc add data/raw
git add data/raw.dvc
git commit -m "feat(data): update raw datasets with new records"
```

### Compare Dataset Versions

```bash
dvc diff HEAD~1 HEAD
```

## Model Versioning & Active Model for Inference

This project uses MLflow Model Registry to manage model versioning and deployment readiness.

### Registered Model

- **Model Name:** gold-price-model  
- **Current Production Version:** Version 2  
- **Staging Version:** Version 1  

### Active Model for Inference

The model currently used for inference is:

```bash
models:/gold-price-model@production
```

### Reason for Selection

Version 2 is promoted to production because it represents the most recent trained model with updated configuration (Linear Regression with modified parameters). This version reflects the latest improvements and is considered the most suitable for deployment.

Version 1 is retained in the staging environment for comparison, testing, and rollback purposes if needed.

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