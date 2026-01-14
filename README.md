
# SmartLend-Explainable-AI 🚀

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.121.2-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green.svg)](https://lightgbm.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **Explainable AI (XAI)** loan default prediction system that provides transparent, interpretable credit risk assessments using SHAP (SHapley Additive exPlanations) values.

---

## 📐 System Architecture

<!-- Add your system architecture image here -->
![System Architecture](assets/architecture.png)

### Architecture Overview

The diagram above illustrates the end-to-end flow of the SmartLend system:

1. **Client Layer**: External applications send loan application data via REST API requests with API key authentication.

2. **API Gateway (FastAPI)**: 
   - Receives incoming requests at the `/predict` endpoint
   - Validates input data against Pydantic schemas (`RequestDataModel`)
   - Enforces API key security via the `Loan-API-Key` header

3. **Data Processing Pipeline**:
   - Raw loan application data passes through the fitted `preprocessor.pkl`
   - Handles categorical encoding, numerical scaling, and feature transformation
   - Outputs processed features aligned with the trained model's expectations
   - **SHAP Feature Alignment** : The pipeline ensures the outputted processed features precisely align with the specific subset of features the model was originally trained on (which were selected based on SHAP importance during the training phase).

4. **Prediction Engine (LightGBM)**:
   - Loads the trained `lgbm_model.pkl` for inference
   - Computes default probability and applies business-configured threshold (default: 0.45)
   - Classifies loans as `DEFAULT` or `NO DEFAULT`

5. **Explainability Module (SHAP)**:
   - Generates SHAP values for each prediction using `shap_explainer.pkl`
   - Identifies top 5 **risk factors** (positive SHAP impact → increases default probability)
   - Identifies top 5 **protective factors** (negative SHAP impact → decreases default probability)

6. **Response Builder**:
   - Constructs a structured JSON response (`ResponseDataModel`)
   - Includes prediction result, probability, risk level, and feature explanations
   - Returns interpretable insights for downstream decision-making

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Explainable Predictions** | Every prediction includes SHAP-based explanations showing which factors contribute to the risk assessment |
| **Risk Stratification** | Automatic classification into HIGH, MEDIUM, or LOW risk levels |
| **Configurable Threshold** | Business-adjustable decision threshold for balancing precision vs. recall |
| **Production-Ready API** | FastAPI-powered REST endpoint with OpenAPI documentation |
| **Secure Access** | API key authentication for controlled access |
| **Docker Support** | Containerized deployment for consistent environments |

---

## 📊 Data Schema

### Input Features

| Field | Type | Range / Values | Description |
|-------|------|----------------|-------------|
| `LoanID` | string (optional) | Unique identifier | Loan application identifier |
| `Age` | integer | 18 – 69 | Borrower's age |
| `Income` | float | 15,000 – 150,000 | Annual income (£) |
| `LoanAmount` | float | 5,000 – 250,000 | Requested loan amount (£) |
| `CreditScore` | integer | 300 – 849 | Borrower's credit score |
| `MonthsEmployed` | integer | 0 – 119 | Employment duration (months) |
| `NumCreditLines` | integer | 1 – 4 | Number of open credit lines |
| `InterestRate` | float | 2.0 – 25.0 | Loan interest rate (%) |
| `LoanTerm` | integer | 12, 24, 36, 48, 60 | Loan duration (months) |
| `DTIRatio` | float | 0.1 – 0.9 | Debt-to-Income ratio |
| `Education` | string | Bachelor's, Master's, PhD, High School, Other | Education level |
| `EmploymentType` | string | Full-time, Part-time, Self-employed, Unemployed, Other | Employment status |
| `MaritalStatus` | string | Married, Single, Divorced, Other | Marital status |
| `HasMortgage` | string | Yes, No | Has existing mortgage |
| `HasDependents` | string | Yes, No | Has dependents |
| `LoanPurpose` | string | Auto, Business, Education, Home, Other | Purpose of loan |
| `HasCoSigner` | string | Yes, No | Has co-signer |

### Response Structure

```json
{
  "status": "success",
  "loan_id": "LOAN-12345",
  "prediction": {
    "result": "NO DEFAULT",
    "probability": 0.2341,
    "threshold_used": 0.45,
    "risk_level": "LOW"
  },
  "explanation": {
    "risk_factors": [
      {"feature": "continuous__DTIRatio", "impact": 0.122},
      {"feature": "yesNo__HasCoSigner_No", "impact": 0.087}
    ],
    "protective_factors": [
      {"feature": "continuous__CreditScore", "impact": -0.387},
      {"feature": "continuous__Income", "impact": -0.334}
    ]
  },
  "metadata": {
    "model_type": "LightGBM",
    "num_features": 23
  }
}
```

### Risk Level Classification

| Risk Level | Probability Range | Interpretation |
|------------|-------------------|----------------|
| **LOW** | < 0.4 | Low likelihood of default |
| **MEDIUM** | 0.4 – 0.6 | Moderate risk, review recommended |
| **HIGH** | ≥ 0.6 | High likelihood of default |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pip or conda

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Over-Mind1/SmartLend-Explainable-AI.git
   cd SmartLend-Explainable-AI
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and set your APP_KEY
   ```

### Running the API

```bash
cd app
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t smartlend-api .

# Run the container
docker run -p 8000:8000 --env-file .env smartlend-api
```

### Docker Compose (optional)

```yaml
version: '3.8'
services:
  smartlend-api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

---

## 📡 API Usage

### Authentication

All requests to `/predict` require an API key in the header:

```
Loan-API-Key: your-api-key-here
```

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "Loan-API-Key: your-api-key" \
  -d '{
    "LoanID": "LOAN-12345",
    "Age": 35,
    "Income": 75000,
    "LoanAmount": 150000,
    "CreditScore": 720,
    "MonthsEmployed": 60,
    "NumCreditLines": 3,
    "InterestRate": 5.5,
    "LoanTerm": 36,
    "DTIRatio": 0.35,
    "Education": "Bachelor'\''s",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "Yes",
    "LoanPurpose": "Home",
    "HasCoSigner": "No"
  }'
```

### Python Client Example

```python
import requests

url = "http://localhost:8000/predict"
headers = {
    "Content-Type": "application/json",
    "Loan-API-Key": "your-api-key"
}
payload = {
    "Age": 35,
    "Income": 75000,
    "LoanAmount": 150000,
    "CreditScore": 720,
    "MonthsEmployed": 60,
    "NumCreditLines": 3,
    "InterestRate": 5.5,
    "LoanTerm": 36,
    "DTIRatio": 0.35,
    "Education": "Bachelor's",
    "EmploymentType": "Full-time",
    "MaritalStatus": "Married",
    "HasMortgage": "Yes",
    "HasDependents": "Yes",
    "LoanPurpose": "Home",
    "HasCoSigner": "No"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

---

## 🔧 Model Training

### Train LightGBM Model (Production)

```bash
python -m src.models.lgbm.pipeline
```

### Tune Decision Threshold

```bash
python -m src.models.lgbm.threshold_tunning
```

> **Note**: Adjust the threshold in `predictor.py` based on business requirements (default: 0.45)

### Train XGBoost Model (Alternative)

```bash
python -m src.models.xgb.pipeline
```

> We use **LightGBM** in production for its speed and stability. Performance gap with XGBoost is minimal.

---

## 📁 Project Structure

```
SmartLend-Explainable-AI/
│
├── app/                         # Application layer (API entry point)
│   ├── __init__.py
│   └── main.py                  # FastAPI bootstrap & endpoints
│
├── Data/                        # Dataset storage
│   ├── raw/                     # Raw, unprocessed data
│   └── processed/               # Cleaned & feature-engineered data
│
├── Notebooks/                   # Research & experimentation
│   ├── EDA.ipynb                # Exploratory Data Analysis
│   └── Loan.ipynb               # Model experiments & analysis
│
├── src/                         # Core project logic
│   │
│   ├── artifacts/               # Serialized models & assets
│   │   ├── lgbm/
│   │   │   ├── lgbm_model.pkl   # Trained LightGBM model
│   │   │   ├── report/          # Evaluation reports
│   │   │   └── shap/            # SHAP explainability assets
│   │   │       ├── feature_names.pkl
│   │   │       ├── shap_explainer.pkl
│   │   │       └── shap_values.pkl
│   │   ├── xgb/                 # XGBoost artifacts
│   │   └── preprocessor.pkl     # Fitted preprocessing pipeline
│   │
│   ├── config/                  # Configuration management
│   │   ├── config.py            # Python config loader
│   │   └── config.yaml          # YAML configuration
│   │
│   ├── models/                  # Model training & evaluation logic
│   │   ├── evaluate.py          # Metrics & evaluation scripts
│   │   ├── processor.py         # Feature processing logic
│   │   ├── lgbm/
│   │   │   ├── pipeline.py      # LightGBM training pipeline
│   │   │   ├── shap_features.py # SHAP explainer generation
│   │   │   ├── threshold_tunning.py  # Business threshold tuning
│   │   │   └── tuner.py         # Optuna hyperparameter tuning
│   │   └── xgb/
│   │       ├── pipeline.py      # XGBoost training pipeline
│   │       └── tuner.py         # Optuna hyperparameter tuning
│   │
│   ├── router/                  # API routing & validation
│   │   ├── __init__.py
│   │   ├── predictor.py         # Prediction logic with SHAP
│   │   └── validator.py         # Pydantic schemas
│   │
│   └── utils/                   # Shared utilities
│       ├── Data_Loader.py       # Data loading helpers
│       └── __init__.py
│
├── .env.example                 # Environment variables template
├── Dockerfile                   # Container configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| **API Framework** | FastAPI 0.121.2 |
| **ML Model** | LightGBM 4.6.0 |
| **Explainability** | SHAP 0.50.0 |
| **Hyperparameter Tuning** | Optuna 4.6.0 |
| **Experiment Tracking** | MLflow 3.8.1 |
| **Data Processing** | scikit-learn 1.7.2, pandas, numpy |
| **Validation** | Pydantic v2 |
| **Containerization** | Docker |

---

## 📚 Data Source

Training data sourced from:  
🔗 [Coursera - Data Science Coding Challenge: Loan Default Prediction](https://www.coursera.org/projects/data-science-coding-challenge-loan-default-prediction)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

**Over-Mind1** - [GitHub Profile](https://github.com/Over-Mind1)

Project Link: [https://github.com/Over-Mind1/SmartLend-Explainable-AI](https://github.com/Over-Mind1/SmartLend-Explainable-AI)