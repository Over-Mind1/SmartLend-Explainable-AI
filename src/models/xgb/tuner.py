from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import optuna
import numpy as np
import logging
logging.getLogger("optuna").setLevel(logging.WARNING)


class XgbTuner:
    """Hyperparameter tuning with proper AUC calculation and CV"""
    
    def __init__(self, X_train, y_train, X_val, y_val, scale_pos_weight, use_cv=True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.scale_pos_weight = scale_pos_weight
        self.use_cv = use_cv

    def xgb_objective(self, trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": 42,
            "scale_pos_weight": self.scale_pos_weight,
            "device":"cuda",
            "tree_method":"hist",
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0)
        }

        if self.use_cv:
            # Use cross-validation on training data
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_scores = []
            
            for train_idx, val_idx in cv.split(self.X_train, self.y_train):
                X_tr, X_vl = self.X_train[train_idx], self.X_train[val_idx]
                y_tr, y_vl = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params, verbosity=0)
                model.fit(X_tr, y_tr)
                
                probs = model.predict_proba(X_vl)[:, 1]
                cv_scores.append(roc_auc_score(y_vl, probs))
            
            return float(np.mean(cv_scores))
        else:
            # Single validation set
            model = xgb.XGBClassifier(**params, verbosity=0)
            model.fit(self.X_train, self.y_train)
            
            # CRITICAL FIX: Use predict_proba
            probs = model.predict_proba(self.X_val)[:, 1]
            return float(roc_auc_score(self.y_val, probs))
    
    def tune_xgb(self, n_trials=100):
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(func=self.xgb_objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\nBest AUC: {study.best_trial.value:.4f}")
        return study.best_trial.params, study.best_trial.value
