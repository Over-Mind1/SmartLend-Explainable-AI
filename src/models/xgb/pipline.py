from src.utils.Data_Loader import data_loader
from src.models.processor import build_preprocessor
from src.models.xgb.tuner import XgbTuner
from src.models.evaluate import run_full_evaluation
from src.config.config import xgb_save_path
import xgboost as xgb
import pickle

def main():
    #------------------loading data----------------#
  
    x_train, x_val, x_test, y_train, y_val,y_test=data_loader()

    # # Slice first 1000 samples for debugging
    # x_train, y_train = x_train[:1000], y_train[:1000]
    # x_val, y_val = x_val[:1000], y_val[:1000]
    # x_test, y_test = x_test[:1000], y_test[:1000]
    
    print('=='*20)
    print("X_train shape:", x_train.shape)
    print("X_val shape:", x_val.shape)
    print("X_test shape:", x_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_val shape:", y_val.shape)
    print("y_test shape:", y_test.shape)
    print('=='*20)

    #--------------------processor-----------------#
    preprocessor=build_preprocessor(x_train)
    x_train_processed=preprocessor.fit_transform(x_train)
    x_val_processed=preprocessor.transform(x_val)
    x_test_processed=preprocessor.transform(x_test)

    print('=='*20)
    print("Processed X_train shape:", x_train_processed.shape)
    print("Processed X_val shape:", x_val_processed.shape)
    print("Processed X_test shape:", x_test_processed.shape)
    print('=='*20)

    #------------------scale pos weight----------------#
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight_value = num_neg / num_pos

    #---------------------SMOTE-------------------#
    # smote = SMOTE(random_state=42, sampling_strategy='auto',k_neighbors=3)
    # x_train_processed, y_train = smote.fit_resample(x_train_processed, y_train)
    # print('After SMOTE:')
    # print('X_train shape:', x_train_processed.shape)
    # print('y_train shape:', y_train.shape)
    # print('=='*20)

    # #---------------------tuner--------------------#
    tuner_instance=XgbTuner(X_train=x_train_processed,y_train=y_train,X_val=x_val_processed,y_val=y_val,scale_pos_weight=scale_pos_weight_value)
    best_params,best_score=tuner_instance.tune_xgb(n_trials=50)
    print('Best auc Score from tuning:',best_score)
    print('Best hyperparameters from tuning:',best_params)
    #---------------------training------------------#
    model=xgb.XGBClassifier(**best_params,
                            objective='binary:logistic',
                            eval_metric='auc',
                            random_state=42,
                            scale_pos_weight=scale_pos_weight_value,
                            verbosity=0,
                            tree_method="hist")  # no XGBoost logs

    model.fit(x_train_processed,y_train)
    # Save the trained model
    model_file_path = xgb_save_path / "xgb_model.json"
    model.save_model(model_file_path)
    pickle_file_path = xgb_save_path / "xgb_model.pkl"
    with open(pickle_file_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Trained XGBoost model saved at: {model_file_path}")
    #------------------------------evaluate------------------#
    results=run_full_evaluation(model, x_train_processed, y_train, model_type="xgb", dataset_name="Train")
    results=run_full_evaluation(model, x_val_processed, y_val, model_type="xgb", dataset_name="Validation")
    results=run_full_evaluation(model, x_test_processed, y_test, model_type="xgb", dataset_name="Test")
    #---------------------end------------------#
if __name__ == "__main__":
    main()