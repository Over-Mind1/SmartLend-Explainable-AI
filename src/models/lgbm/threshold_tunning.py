from src.models.evaluate import run_full_evaluation
from src.config.config import processor_path, lgbm_save_path,lgbm_shap_path
from src.utils.Data_Loader import data_loader
import pickle
import pandas as pd
import numpy as np

#------------------loading data----------------#
x_train, x_val, x_test, y_train, y_val,y_test=data_loader()
#------------------load processor------------------#
with open(processor_path / "preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)
#------------------load model------------------#
with open(lgbm_save_path/'lgbm_model.pkl', 'rb') as f:
    model = pickle.load(f)

#------------------process data------------------#
x_train_processed=preprocessor.transform(x_train)
x_val_processed=preprocessor.transform(x_val)
x_test_processed=preprocessor.transform(x_test)


feature_names = preprocessor.get_feature_names_out()

# Convert to DataFrames with feature names (ensure dense array for type safety)
x_train_processed = pd.DataFrame(np.asarray(x_train_processed), columns=feature_names, index=x_train.index)
x_val_processed = pd.DataFrame(np.asarray(x_val_processed), columns=feature_names, index=x_val.index)
x_test_processed = pd.DataFrame(np.asarray(x_test_processed), columns=feature_names, index=x_test.index)

#------------------load shap features------------------#
with open(lgbm_shap_path / "feature_names.pkl", "rb") as f:
    selected_features = pickle.load(f)

x_train_processed = x_train_processed[selected_features]
x_val_processed = x_val_processed[selected_features]
x_test_processed = x_test_processed[selected_features]    

#------------------------------evaluate------------------#
run_full_evaluation(model, x_train_processed, y_train, model_type="lgbm", dataset_name="Train")
run_full_evaluation(model, x_val_processed, y_val, model_type="lgbm", dataset_name="Validation")
run_full_evaluation(model, x_test_processed, y_test, model_type="lgbm", dataset_name="Test")
