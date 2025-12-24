
---

# SmartLend-Explainable-AI 🚀

### 📌 Overview

An end-to-end Smart Loan Approval System that leverages Machine Learning to automate credit decisions. The system doesn't just approve or reject; it predicts the optimal interest rate, assesses risk levels, and provides clear explanations for its decisions using SHAP.

### 🏗️ Project Architecture (Planned)

* **Classification Model:** To decide Loan Approval (Approve/Reject).
* **Regression Model:** To predict the appropriate Interest Rate.
* **Risk Scoring:** Categorizing applications into (Low, Medium, High) risk.
* **Explainability Layer:** Using SHAP to explain "Why" a decision was made.
* **Monitoring:** Data Drift detection to ensure long-term reliability.

### 📊 Data Sources

The models will be trained using a combination of:

1. **Lending Club Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
2. **Credit Risk Dataset:** [Kaggle Link](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

### 🛠️ Tech Stack

* **Languages:** Python
* **ML Frameworks:** Scikit-Learn, XGBoost, LightGBM, CatBoost
* **Optimization:** Optuna
* **API:** FastAPI & Uvicorn
* **Deployment:** Docker & Docker Compose
* **Monitoring:** Evidently AI

### 📂 Folder Structure

```text
├── data/               # Raw and processed data
├── Notebooks/          # Exploratory Data Analysis & Model Experiments
├── src/                # Source code for the production pipeline
│   ├── components/     # Data ingestion, transformation, and training
│   ├── pipeline/       # Training and prediction pipelines
│   └── utils/          # Common utility functions
├── artifacts/          # Saved models and transformation objects
├── app.py              # FastAPI application entry point
├── Dockerfile          # Containerization script
└── requirements.txt    # Project dependencies

```

### 🚧 Status: Work In Progress

Currently setting up the environment and performing initial Data Exploration.

---
