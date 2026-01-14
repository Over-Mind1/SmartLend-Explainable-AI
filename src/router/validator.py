#----------------iniatlize data validation models----------------#
from pydantic import BaseModel, Field, field_validator
from typing import Literal, List
 
#  sample_data = pd.DataFrame([{
#         "LoanID":"jnc7vsd885d",
#         "Age": 58,
#         "Income": 36970,
#         "LoanAmount": 161875,
#         "CreditScore": 446,
#         "MonthsEmployed": 57,
#         "NumCreditLines": 1,
#         "InterestRate": 4.9,
#         "LoanTerm": 36,
#         "DTIRatio": 0.42,
#         "Education": "Master's",
#         "EmploymentType": "Self-employed",
#         "MaritalStatus": "Married",
#         "HasMortgage": "No",
#         "HasDependents": "No",
#         "LoanPurpose": "Other",
#         "HasCoSigner": "No"
#     }])

class RequestDataModel(BaseModel):
    LoanID: str | None = Field(default=None, description="Unique identifier for the loan application")
    Age: int = Field(..., ge=18, le=69, description="Age of the borrower")
    Income: float = Field(..., ge=15000, le=150000, description="Annual income of the borrower")
    LoanAmount: float = Field(..., ge=5000, le=250000, description="Amount of money borrowed")
    CreditScore: int = Field(..., ge=300, le=849, description="Credit score of the borrower")
    MonthsEmployed: int = Field(..., ge=0, le=119, description="Number of months employed")
    NumCreditLines: int = Field(..., ge=1, le=4, description="Total number of open credit lines")
    InterestRate: float = Field(..., ge=2.0, le=25.0, description="Interest rate applied to the loan")
    LoanTerm: Literal[12, 24, 36, 48, 60] = Field(..., description="Duration of the loan in months")
    DTIRatio: float = Field(..., ge=0.1, le=0.9, description="Debt-to-Income ratio")
    Education: str = Field(..., description="Highest level of education attained")
    EmploymentType: str = Field(..., description="Borrower's employment status")
    MaritalStatus: str = Field(..., description="Marital status of the borrower")
    HasMortgage: str = Field(..., description="Whether the borrower has a mortgage")
    HasDependents: str = Field(..., description="Whether the borrower has dependents")
    LoanPurpose: str = Field(..., description="Purpose for which the loan was taken")
    HasCoSigner: str = Field(..., description="Whether the loan has a co-signer")
    
    @field_validator('Education')
    @classmethod
    def validate_education(cls, v: str) -> str:
        valid_values = ["Bachelor's", "Master's", "PhD", "High School", "Other"]
        if v not in valid_values:
            raise ValueError(f"Education must be one of {valid_values}")
        return v
    
    @field_validator('EmploymentType')
    @classmethod
    def validate_employment_type(cls, v: str) -> str:
        valid_values = ["Full-time", "Part-time", "Self-employed", "Unemployed", "Other"]
        if v not in valid_values:
            raise ValueError(f"EmploymentType must be one of {valid_values}")
        return v
    
    @field_validator('MaritalStatus')
    @classmethod
    def validate_marital_status(cls, v: str) -> str:
        valid_values = ["Married", "Single", "Divorced", "Other"]
        if v not in valid_values:
            raise ValueError(f"MaritalStatus must be one of {valid_values}")
        return v
    
    @field_validator('HasMortgage', 'HasDependents', 'HasCoSigner')
    @classmethod
    def validate_yes_no(cls, v: str) -> str:
        valid_values = ["Yes", "No"]
        if v not in valid_values:
            raise ValueError(f"Value must be 'Yes' or 'No'")
        return v
    
    @field_validator('LoanPurpose')
    @classmethod
    def validate_loan_purpose(cls, v: str) -> str:
        valid_values = ["Auto", "Business", "Education", "Home", "Other"]
        if v not in valid_values:
            raise ValueError(f"LoanPurpose must be one of {valid_values}")
        return v





class PredictionModel(BaseModel):
    result: str
    probability: float = Field(..., ge=0.0, le=1.0)
    threshold_used: float
    risk_level: Literal["HIGH", "MEDIUM", "LOW"]


class FeatureContribution(BaseModel):
    feature: str
    impact: float


class ExplanationModel(BaseModel):
    risk_factors: List[FeatureContribution] = Field(default_factory=list)
    protective_factors: List[FeatureContribution] = Field(default_factory=list)


class MetadataModel(BaseModel):
    model_type: str
    num_features: int = Field(..., ge=0)


class ResponseDataModel(BaseModel):
    status: str
    loan_id: str | None
    prediction: PredictionModel
    explanation: ExplanationModel
    metadata: MetadataModel

    # # Example for response from predictor.py
    #    response = {
    #     "status": "success",
    #     "loan_id": loan_id,
    #     "prediction": {
    #         "result": prediction,
    #         "probability": round(proba, 4),
    #         "threshold_used": threshold,
    #         "risk_level": "HIGH" if proba >= 0.6 else "MEDIUM" if proba >= 0.4 else "LOW"
    #     },
    #     "explanation": {
    #         "risk_factors": risk_factors,
    #         "protective_factors": protective_factors,

    #     },
    #     "metadata": {
    #         "model_type": "LightGBM",
    #         "num_features": len(feature_names),
    #     }
    # }
     