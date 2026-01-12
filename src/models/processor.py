from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,RobustScaler,OrdinalEncoder

def build_preprocessor(X):
    """
    Build preprocessing pipeline with separate handling for:
    - Continuous numeric features (scale)
    - Ordinal features
    - Categorical features
    """
    
    # Define original columns
    yesNoColumns = ["HasMortgage", "HasDependents", "HasCoSigner"]
    categorical_features = list(set(X.select_dtypes(include=['object'])) - set(yesNoColumns))
    
    # Original numeric features
    original_numeric = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 
        'MonthsEmployed', 'NumCreditLines', 'InterestRate', 
        'LoanTerm', 'DTIRatio'
    ]
    
    
    # Continuous numeric transformer (with log)
    continuous_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # Better for outliers
    ])
    

    
    # Yes/No columns
    yesNoColumns_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ordinal', OrdinalEncoder())
    ])
    
    # Categorical
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine all
    preprocessor = ColumnTransformer(
        transformers=[
            ('continuous', continuous_transformer, original_numeric ),
            ('yesNo', yesNoColumns_transformer, yesNoColumns),
            ('cat', categorical_transformer, categorical_features)
        ],
    )
    
    
    return preprocessor
