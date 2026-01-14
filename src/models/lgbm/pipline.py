from src.utils.Data_Loader import data_loader
from src.models.processor import build_preprocessor
from src.models.lgbm.tuner import LGBMTuner
from src.models.lgbm.shap_features import get_shap_values, select_top_features_shap, save_shap_artifacts # type: ignore
from mlflow import log_metric, log_params, log_artifacts, set_tracking_uri, set_experiment
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from src.config.config import lgbm_save_path,processor_path,lgbm_shap_path
import lightgbm as lgb
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.lightgbm  # type: ignore


def main():
    #------------------MLflow Setup----------------#
    set_tracking_uri("mlruns")  # Local tracking directory
    set_experiment("SmartLend-LGBM-Pipeline")
    
    with mlflow.start_run(run_name="lgbm_training"):
        mlflow.set_tags({
            "model_type": "LightGBM",
            "task": "binary_classification",
            "target": "loan_default",
            "developer": "mohamed sabry (over-mind)"
        })
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
        
        # Log data shapes
        log_params({
            "train_samples": x_train.shape[0],
            "val_samples": x_val.shape[0],
            "test_samples": x_test.shape[0],
            "num_features": x_train.shape[1]
        })

        #--------------------processor-----------------#
        preprocessor=build_preprocessor(x_train)
        x_train_processed=preprocessor.fit_transform(x_train)
        x_val_processed=preprocessor.transform(x_val)
        x_test_processed=preprocessor.transform(x_test)
        
        # Save the preprocessor
        preprocessor_save_path = processor_path / "preprocessor.pkl"
        with open(preprocessor_save_path, "wb") as f:
            pickle.dump(preprocessor, f)
        print(f"Preprocessor saved at: {preprocessor_save_path}")
        
        # Log preprocessor artifact
        mlflow.log_artifact(preprocessor_save_path)

        print('=='*20)
        print("Processed X_train shape:", x_train_processed.shape)
        print("Processed X_val shape:", x_val_processed.shape)
        print("Processed X_test shape:", x_test_processed.shape)
        print('=='*20)
        
        # Log processed feature count
        log_metric("processed_features", x_train_processed.shape[1])

        feature_names = preprocessor.get_feature_names_out()

        # Convert to DataFrames with feature names (ensure dense array for type safety)
        x_train_processed = pd.DataFrame(np.asarray(x_train_processed), columns=feature_names, index=x_train.index)
        x_val_processed = pd.DataFrame(np.asarray(x_val_processed), columns=feature_names, index=x_val.index)
        x_test_processed = pd.DataFrame(np.asarray(x_test_processed), columns=feature_names, index=x_test.index)

        #------------------scale pos weight----------------#
        num_neg = (y_train == 0).sum()
        num_pos = (y_train == 1).sum()
        scale_pos_weight_value = num_neg / num_pos
        
        # Log class imbalance info
        log_params({
            "num_negative_samples": int(num_neg),
            "num_positive_samples": int(num_pos),
            "scale_pos_weight": float(scale_pos_weight_value)
        })

        # #---------------------SMOTE-------------------#
        # smote = SMOTE(random_state=42, sampling_strategy='auto',k_neighbors=5)
        # resampled = smote.fit_resample(x_train_processed, y_train)
        # x_train_processed, y_train = resampled[0], resampled[1]
        # print('After SMOTE:')
        # print('X_train shape:', x_train_processed.shape)
        # print('y_train shape:', y_train.shape)
        # print('=='*20)
        
        # # Log SMOTE parameters and results
        # log_params({
        #     "smote_sampling_strategy": "auto",
        #     "smote_k_neighbors": 5,
        #     "smote_random_state": 42,
        #     "train_samples_after_smote": x_train_processed.shape[0]
        # })

        # #---------------------tuner--------------------#
        tuner_instance=LGBMTuner(X_train=x_train_processed,y_train=y_train,X_val=x_val_processed,y_val=y_val,scale_pos_weight=scale_pos_weight_value)
        best_params,best_score=tuner_instance.tune_lgbm(n_trials=25)
        print('Best auc Score from tuning:',best_score)
        print('Best hyperparameters from tuning:',best_params)
        
        # Log tuning results
        log_params({"n_tuning_trials": 25})
        log_metric("best_tuning_auc", float(best_score) if best_score is not None else 0.0)
        log_params({f"best_{k}": v for k, v in best_params.items()})
        
        #---------------------training------------------#
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            scale_pos_weight=scale_pos_weight_value,
            random_state=42,
            subsample_freq=1,
            **best_params
        )

        model.fit(x_train_processed, y_train,
                    eval_set=[(x_val_processed, y_val)],
                    callbacks=[lgb.early_stopping(100), 
                                lgb.log_evaluation(period=0)])
        
        #---------------------shape------------------#
        # SHAP explainability
        shap_importance ,_, _ = get_shap_values(model, x_val_processed)
        selected_features = select_top_features_shap(shap_importance)
        print(f"Selected top {len(selected_features)} features based on SHAP importance.")

        #---------------------train with selected features------------------#
        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            scale_pos_weight=scale_pos_weight_value,
            random_state=42,
            subsample_freq=1,
            **best_params
        )
        model.fit(x_train_processed[selected_features], y_train,
                    eval_set=[(x_val_processed[selected_features], y_val)],
                    callbacks=[lgb.early_stopping(100), 
                                lgb.log_evaluation(period=0)])
        
        #---------------------shape and save explainer with new dim------------------#
        # SHAP explainability
        shap_importance ,shap_values, explainer = get_shap_values(model, x_val_processed[selected_features])
        save_shap_artifacts(explainer, shap_values, selected_features, lgbm_shap_path)
        print(f"Selected top {len(selected_features)} features based on SHAP importance.")

        #---------------------log metrics------------------#
        # Calculate and log metrics for all datasets
        for name, X, y in [("train", x_train_processed[selected_features], y_train), 
                           ("val", x_val_processed[selected_features], y_val), 
                           ("test", x_test_processed[selected_features], y_test)]:
            y_pred_proba = np.asarray(model.predict_proba(X))[:, 1]
            y_pred = np.asarray(model.predict(X))
            print(f"{name} AUC: {roc_auc_score(y, y_pred_proba)}")
            log_metric(f"{name}_auc", float(roc_auc_score(y, y_pred_proba)))
            log_metric(f"{name}_accuracy", float(accuracy_score(y, y_pred)))
            log_metric(f"{name}_precision", float(precision_score(y, y_pred)))
            log_metric(f"{name}_recall", float(recall_score(y, y_pred)))
            log_metric(f"{name}_f1", float(f1_score(y, y_pred)))
        
        #---------------------save model------------------#
        save_path=lgbm_save_path/'lgbm_model.pkl'
        pickle.dump(model, open(save_path, 'wb'))
        print(f"Trained LightGBM model saved at: {save_path}")
        
        # Log model artifact
        mlflow.log_artifact(save_path)
        
        # Log model with MLflow's native LightGBM support
        mlflow.lightgbm.log_model(model, "lgbm_model")
        
        # Log the artifacts directory
        log_artifacts(str(lgbm_save_path))
        
        print("MLflow run completed. Check 'mlruns' directory for tracking data.")
        active_run = mlflow.active_run()
        if active_run:
            print(f"Run ID: {active_run.info.run_id}")
        
    #---------------------end------------------#
if __name__ == "__main__":
    main()