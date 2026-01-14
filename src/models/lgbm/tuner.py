from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna
import numpy as np
import logging
logging.getLogger("optuna").setLevel(logging.WARNING)


class LGBMTuner:
    """Hyperparameter tuning for LightGBM with proper AUC calculation and CV"""
    
    def __init__(self, X_train, y_train, X_val, y_val, scale_pos_weight, use_cv=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scale_pos_weight = scale_pos_weight
        self.use_cv = use_cv

    def lgbm_objective(self, trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "random_state": 42,
            "scale_pos_weight": self.scale_pos_weight,
            # "device": "gpu",
            # "gpu_platform_id": 0,
            # "gpu_device_id": 0,
            "verbosity": -1,
            "subsample_freq": 1,
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        }

        if self.use_cv:
            # Use cross-validation on training data
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_tr, X_vl = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_tr, y_vl = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=0)])
                
                probs = np.asarray(model.predict_proba(X_vl))[:, 1]
                cv_scores.append(roc_auc_score(y_vl, probs))
            
            return float(np.mean(cv_scores))
        else:
            # Single validation set
            model = lgb.LGBMClassifier(**params)
            model.fit(self.X_train, self.y_train, callbacks=[lgb.log_evaluation(period=0)])
            
            probs = np.asarray(model.predict_proba(self.X_val))[:, 1]
            return float(roc_auc_score(self.y_val, probs))
    
    def tune_lgbm(self, n_trials=100):
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(func=self.lgbm_objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest AUC: {study.best_trial.value:.4f}")
        return study.best_trial.params, study.best_trial.value