import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pandas import DataFrame
from starlette.status import HTTP_403_FORBIDDEN

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config.config import APP_ENV, APP_KEY, APP_NAME, APP_VERSION
from src.router.predictor import predict_with_explanation,predict_batch_with_explanation
from src.router.validator import RequestDataModel, ResponseDataModel


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description="API for SmartLend Explainable AI Model",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


api_key_header = APIKeyHeader(name="Loan-API-Key")


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    if api_key != APP_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    return api_key


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": f"Welcome to {APP_NAME} API!"}


@app.post("/predict", response_model=ResponseDataModel)
async def predict(
    data: RequestDataModel, api_key: str = Depends(verify_api_key)
) -> ResponseDataModel:
    """Return a validated prediction and SHAP explanation for the provided loan input."""

    processed_data = DataFrame([data.dict(exclude_none=True)])
    prediction_payload = predict_with_explanation(input_data=processed_data)
    return ResponseDataModel(**prediction_payload)

@app.post("/predict_batch", response_model=list[ResponseDataModel])
async def predict_batch(
    data: list[RequestDataModel], api_key: str = Depends(verify_api_key)
) -> list[ResponseDataModel]:
    """Return validated predictions and SHAP explanations for the provided batch of loan inputs."""

    processed_data = DataFrame([item.dict(exclude_none=True) for item in data])
    prediction_payloads = predict_batch_with_explanation(input_data=processed_data)
    return [ResponseDataModel(**payload) for payload in prediction_payloads]