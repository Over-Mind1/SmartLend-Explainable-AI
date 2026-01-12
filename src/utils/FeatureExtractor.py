import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class LoanFeatureExtractor(BaseEstimator, TransformerMixin):
    """feature engineering with domain knowledge"""
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_ext = X.copy()

        # 1. Core Financial Ratios
        X_ext['Loan_To_Income_Ratio'] = X_ext['LoanAmount'] / (X_ext['Income'] + 1)
        X_ext['Monthly_Payment'] = X_ext['LoanAmount'] / X_ext['LoanTerm']
        X_ext['Payment_To_Income_Ratio'] = (X_ext['Monthly_Payment'] * 12) / (X_ext['Income'] + 1)
        
        # 2. Credit Utilization & Risk
        # Normalize credit score (higher is better)
        X_ext['Credit_Score_Normalized'] = X_ext['CreditScore'] / 850.0
        X_ext['Risk_Score'] = X_ext['DTIRatio'] * (1 - X_ext['Credit_Score_Normalized'])
        
        # 3. Interest & Debt Burden
        X_ext['Total_Interest_Cost'] = X_ext['LoanAmount'] * (X_ext['InterestRate'] / 100) * (X_ext['LoanTerm'] / 12)
        X_ext['Interest_To_Income'] = X_ext['Total_Interest_Cost'] / (X_ext['Income'] + 1)
        
        # 4. Employment & Stability
        X_ext['Employment_To_Age_Ratio'] = X_ext['MonthsEmployed'] / ((X_ext['Age'] - 18) * 12 + 1)
        X_ext['Income_Per_Year_Employed'] = X_ext['Income'] / ((X_ext['MonthsEmployed'] / 12) + 1)
        
        # 5. Credit Line Utilization
        X_ext['Loan_Per_Credit_Line'] = X_ext['LoanAmount'] / (X_ext['NumCreditLines'] + 1)
        
        # 6. Interaction Features
        X_ext['Age_Income_Interaction'] = X_ext['Age'] * np.log1p(X_ext['Income'])
        X_ext['Credit_DTI_Interaction'] = X_ext['CreditScore'] * (1 - X_ext['DTIRatio'])
        
        # 7. Risk Flags (keep as binary, don't log transform)
        X_ext['High_Risk_Flag'] = (
            (X_ext['DTIRatio'] > 0.6) & 
            (X_ext['CreditScore'] < 600)
        ).astype(int)
        
        X_ext['Low_Income_High_Loan_Flag'] = (
            (X_ext['Loan_To_Income_Ratio'] > 3) & 
            (X_ext['Income'] < 40000)
        ).astype(int)
        
        X_ext['Unstable_Employment_Flag'] = (
            (X_ext['MonthsEmployed'] < 12) & 
            (X_ext['Age'] > 25)
        ).astype(int)
        
        # Handle inf/nan
        X_ext = X_ext.replace([np.inf, -np.inf], np.nan)
        
        return X_ext