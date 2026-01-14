from src.config.config import lgbm_shap_path, processor_path, lgbm_save_path
import joblib
import pandas as pd
import numpy as np
import logging

#----------------load artifacts----------------#
def load_artifacts():
    """Load necessary artifacts for prediction and SHAP explanation"""
    # Load processor
    processor = joblib.load(processor_path/"preprocessor.pkl")
    logging.info("Processor loaded successfully.")
    
    # Load LightGBM model
    lgbm_model = joblib.load(lgbm_save_path/"lgbm_model.pkl")
    logging.info("LightGBM model loaded successfully.")
    
    # Load SHAP explainer
    shap_explainer = joblib.load(lgbm_shap_path / "shap_explainer.pkl")
    logging.info("SHAP explainer loaded successfully.")
    
    # Load feature names
    feature_names = joblib.load(lgbm_shap_path / "feature_names.pkl")
    logging.info("Feature names loaded successfully.")
    return processor, lgbm_model, shap_explainer, feature_names


#----------------predict with explanation----------------#
def predict_with_explanation(input_data: pd.DataFrame, threshold: float = 0.45) -> dict:
    """
    Make prediction with LightGBM model and provide SHAP explanation.
    Returns API-friendly JSON-serializable response.
    
    Parameters:
    - input_data: Raw input data (DataFrame with single row)
    - threshold: Decision threshold for classification
    
    Returns:
    - dict with prediction and SHAP explanation (JSON-serializable)
    """
    # Load artifacts
    processor, model, explainer, feature_names = load_artifacts()
    
    # Extract LoanID if present
    loan_id = None
    if "LoanID" in input_data.columns:
        loan_id = str(input_data["LoanID"].iloc[0])
        input_data = input_data.drop(columns=["LoanID"])
    
    # Preprocess input data
    X_processed = processor.transform(input_data)
    feature_names_out = processor.get_feature_names_out()
    X_processed = pd.DataFrame(np.asarray(X_processed), columns=feature_names_out)
    X_input = X_processed[feature_names]
    
    # Get prediction probability
    proba = float(model.predict_proba(X_input)[0, 1])
    prediction = "DEFAULT" if proba >= threshold else "NO DEFAULT"
    
    # Get SHAP values
    shap_vals = explainer(X_input)
    
    # Get top contributing features (sorted by absolute impact)
    feature_contributions = list(zip(feature_names, shap_vals.values[0]))
    feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Separate positive (risk) and negative (protective) contributors
    risk_factors = [
        {"feature": f, "impact": float(v)} 
        for f, v in feature_contributions if v > 0
    ][:5]
    
    protective_factors = [
        {"feature": f, "impact": float(v)} 
        for f, v in feature_contributions if v < 0
    ][:5]
    
    # All feature contributions for detailed analysis
    all_contributions = [
        {"feature": f, "impact": float(v)} 
        for f, v in feature_contributions
    ]
    
    # Build API-friendly response
    response = {
        "status": "success",
        "loan_id": loan_id,
        "prediction": {
            "result": prediction,
            "probability": round(proba, 4),
            "threshold_used": threshold,
            "risk_level": "HIGH" if proba >= 0.6 else "MEDIUM" if proba >= 0.4 else "LOW"
        },
        "explanation": {
            "risk_factors": risk_factors,
            "protective_factors": protective_factors,

        },
        "metadata": {
            "model_type": "LightGBM",
            "num_features": len(feature_names),
        }
    }
    
    return response

#------------------predict more than one sample with explanation------------------#
def predict_batch_with_explanation(input_data: pd.DataFrame, threshold: float = 0.45) -> list[dict]:
    """
    Make predictions with LightGBM model and provide SHAP explanations for a batch of samples.
    Returns a list of API-friendly JSON-serializable responses.
    
    Parameters:
    - input_data: Raw input data (DataFrame with multiple rows)
    - threshold: Decision threshold for classification
    
    Returns:
    - list of dicts with predictions and SHAP explanations (JSON-serializable)
    """
    responses = []
    for _, row in input_data.iterrows():
        single_input = pd.DataFrame([row])
        response = predict_with_explanation(single_input, threshold)
        responses.append(response)
    return responses

# example usage:
if __name__ == "__main__":
    sample_data = pd.DataFrame([{
        "LoanID":"jnc7vsd885d",
        "Age": 58,
        "Income": 36970,
        "LoanAmount": 161875,
        "CreditScore": 446,
        "MonthsEmployed": 57,
        "NumCreditLines": 1,
        "InterestRate": 4.9,
        "LoanTerm": 36,
        "DTIRatio": 0.42,
        "Education": "Master's",
        "EmploymentType": "Self-employed",
        "MaritalStatus": "Married",
        "HasMortgage": "No",
        "HasDependents": "No",
        "LoanPurpose": "Other",
        "HasCoSigner": "No"
    }])
    
    result = predict_with_explanation(sample_data, threshold=0.45)
    print(result)


