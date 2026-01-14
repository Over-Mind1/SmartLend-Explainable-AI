import shap
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Get SHAP values
def get_shap_values(lgb_model, x_val_processed):
    """
    Calculate SHAP values and feature importance for the validation set
    
    Parameters:
    - lgb_model: Trained LightGBM model
    - x_val_processed: Processed validation features (DataFrame)
    Returns:
    - shap_importance: DataFrame with feature importance based on SHAP values
    - shap_values: SHAP values object
    - explainer: SHAP explainer object
    """
    explainer = shap.Explainer(lgb_model)
    shap_values = explainer(x_val_processed)

    # Calculate mean absolute SHAP values for feature importance
    shap_importance = pd.DataFrame({
        'feature': x_val_processed.columns,
        'importance': np.abs(shap_values.values).mean(axis=0)
    }).sort_values('importance', ascending=False)
    return shap_importance ,shap_values, explainer

# Select top N features
def select_top_features_shap(shap_importance, top_n=None, threshold=None):
    """
    Select features based on SHAP importance.
    
    Parameters:
    - top_n: Select top N features
    - threshold: Select features with importance > threshold
    """
    if top_n:
        selected = shap_importance.head(top_n)['feature'].tolist()
    elif threshold:
        selected = shap_importance[shap_importance['importance'] > threshold]['feature'].tolist()
    else:
        # Cumulative importance (select features that contribute to 95% importance)
        shap_importance['cumsum'] = shap_importance['importance'].cumsum()
        total = shap_importance['importance'].sum()
        selected = shap_importance[shap_importance['cumsum'] <= 0.90 * total]['feature'].tolist()
    
    return selected

def save_shap_artifacts(explainer, shap_values, feature_names, save_path):
    """Save all SHAP-related artifacts"""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save explainer
    with open(save_path / "shap_explainer.pkl", "wb") as f:
        joblib.dump(explainer, f)
    
    # Save SHAP values
    with open(save_path / "shap_values.pkl", "wb") as f:
        joblib.dump(shap_values, f)
    
    # Save feature names
    with open(save_path / "feature_names.pkl", "wb") as f:
        joblib.dump(feature_names, f)
    
    print(f"All SHAP artifacts saved to: {save_path}")


