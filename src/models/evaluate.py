"""
Model Evaluation Module
=======================
This module provides functions for evaluating machine learning models,
including ROC curve analysis, threshold optimization, and business impact assessment.
All results are saved based on model type to appropriate paths.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, 
    RocCurveDisplay,
    confusion_matrix, 
    classification_report,
)
from src.config.config import xgb_save_path, lgbm_report_path as lgbm_save_path


def get_model_save_path(model_type: str) -> Path:
    """
    Get the save path based on model type.
    
    Parameters:
    -----------
    model_type : str
        Type of model ('xgb', 'lgbm', 'xgboost', 'lightgbm')
        
    Returns:
    --------
    Path: The save path for the model type
    """
    model_type = model_type.lower()
    if model_type in ['xgb', 'xgboost']:
        return xgb_save_path
    elif model_type in ['lgbm', 'lightgbm']:
        return lgbm_save_path
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'xgb' or 'lgbm'")


def evaluate_model(model, X, y, model_type: str, dataset_name: str = "Dataset"):
    """
    Evaluate model performance and save ROC curve.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict_proba method
    X : array-like
        Feature matrix
    y : array-like
        True labels
    model_type : str
        Type of model ('xgb' or 'lgbm')
    dataset_name : str
        Name of the dataset for display
        
    Returns:
    --------
    float: AUC score
    """
    save_path = get_model_save_path(model_type)
    save_path.mkdir(parents=True, exist_ok=True)
    
    y_pred_proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred_proba)
    print(f"AUC on {dataset_name}: {auc:.4f}")
    
    # Create and save ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X, y, name=f"{model_type.upper()} ROC Curve", ax=ax)
    ax.set_title(f"{model_type.upper()} ROC Curve - {dataset_name}")
    plt.tight_layout()
    
    roc_save_path = save_path / f"roc_curve_{dataset_name.lower().replace(' ', '_')}.png"
    plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to: {roc_save_path}")
    
    # Save AUC score to file
    auc_results = pd.DataFrame([{
        'dataset': dataset_name,
        'auc_score': auc,
        'model_type': model_type
    }])
    auc_save_path = save_path / f"auc_score_{dataset_name.lower().replace(' ', '_')}.csv"
    auc_results.to_csv(auc_save_path, index=False)
    print(f"AUC score saved to: {auc_save_path}")
    
    return auc


def find_optimal_threshold(model, X, y_true, model_type: str, cost_fn: float = 10, cost_fp: float = 1):
    """
    Find optimal threshold based on business costs.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict_proba method
    X : array-like
        Feature matrix
    y_true : array-like
        True labels
    model_type : str
        Type of model ('xgb' or 'lgbm')
    cost_fn : float
        Cost of missing a defaulter (False Negative) - e.g., loan loss
    cost_fp : float
        Cost of rejecting a good customer (False Positive) - e.g., lost revenue
    
    Returns:
    --------
    pd.DataFrame: DataFrame with threshold analysis results
    
    Notes:
    ------
    Higher cost_fn means we want to catch more defaults (higher recall).
    Results are saved to CSV in the model's save path.
    """
    save_path = get_model_save_path(model_type)
    save_path.mkdir(parents=True, exist_ok=True)
    
    y_proba = model.predict_proba(X)[:, 1]
    
    thresholds = np.arange(0.1, 0.9, 0.01)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Business cost calculation
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        
        # Defaults caught rate
        default_catch_rate = recall
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
            'total_cost': total_cost,
            'default_catch_rate': default_catch_rate
        })
    
    threshold_df = pd.DataFrame(results)
    
    # Save threshold analysis to CSV
    threshold_save_path = save_path / "threshold_analysis.csv"
    threshold_df.to_csv(threshold_save_path, index=False)
    print(f"Threshold analysis saved to: {threshold_save_path}")
    
    return threshold_df


def visualize_threshold_analysis(threshold_df: pd.DataFrame, model_type: str):
    """
    Visualize threshold analysis and save plots.
    
    Parameters:
    -----------
    threshold_df : pd.DataFrame
        DataFrame from find_optimal_threshold function
    model_type : str
        Type of model ('xgb' or 'lgbm')
    """
    save_path = get_model_save_path(model_type)
    save_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Precision-Recall vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(threshold_df['threshold'], threshold_df['precision'], 'b-', label='Precision', linewidth=2)
    ax1.plot(threshold_df['threshold'], threshold_df['recall'], 'r-', label='Recall (Default Catch Rate)', linewidth=2)
    ax1.plot(threshold_df['threshold'], threshold_df['f1'], 'g--', label='F1 Score', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall & F1 vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Business Cost vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(threshold_df['threshold'], threshold_df['total_cost'], 'purple', linewidth=2)
    optimal_cost_idx = threshold_df['total_cost'].idxmin()
    optimal_cost_thresh = threshold_df.loc[optimal_cost_idx, 'threshold']
    ax2.axvline(x=optimal_cost_thresh, color='red', linestyle='--', label=f'Optimal (min cost): {optimal_cost_thresh:.2f}')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Total Business Cost')
    ax2.set_title('Business Cost vs Threshold\n(Cost_FN=10, Cost_FP=1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Confusion Matrix components vs Threshold
    ax3 = axes[1, 0]
    ax3.plot(threshold_df['threshold'], threshold_df['tp'], label='True Positives (Caught Defaults)', linewidth=2)
    ax3.plot(threshold_df['threshold'], threshold_df['fn'], label='False Negatives (Missed Defaults)', linewidth=2)
    ax3.plot(threshold_df['threshold'], threshold_df['fp'], label='False Positives (Rejected Good)', linewidth=2)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Count')
    ax3.set_title('Confusion Matrix Components vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Default Catch Rate with different thresholds
    ax4 = axes[1, 1]
    ax4.bar(threshold_df['threshold'][::5], threshold_df['recall'][::5], width=0.04, alpha=0.7, color='coral')
    ax4.set_xlabel('Threshold')
    ax4.set_ylabel('Default Catch Rate (Recall)')
    ax4.set_title('Default Catch Rate at Different Thresholds')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the visualization
    viz_save_path = save_path / "threshold_analysis_plot.png"
    plt.savefig(viz_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Threshold analysis plot saved to: {viz_save_path}")


def get_optimal_thresholds(threshold_df: pd.DataFrame, model_type: str) -> dict:
    """
    Calculate and save optimal thresholds for different business objectives.
    
    Parameters:
    -----------
    threshold_df : pd.DataFrame
        DataFrame from find_optimal_threshold function
    model_type : str
        Type of model ('xgb' or 'lgbm')
        
    Returns:
    --------
    dict: Dictionary containing optimal thresholds for different objectives
    """
    save_path = get_model_save_path(model_type)
    save_path.mkdir(parents=True, exist_ok=True)
    
    optimal_thresholds = {}
    
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLDS FOR DIFFERENT BUSINESS OBJECTIVES")
    print("="*60)
    
    # Minimum cost threshold
    min_cost_row = threshold_df.loc[threshold_df['total_cost'].idxmin()]
    optimal_thresholds['min_cost'] = {
        'threshold': min_cost_row['threshold'],
        'recall': min_cost_row['recall'],
        'precision': min_cost_row['precision'],
        'f1': min_cost_row['f1'],
        'missed_defaults': int(min_cost_row['fn'])
    }
    print(f"\n1. MINIMUM BUSINESS COST (Cost_FN=10, Cost_FP=1):")
    print(f"   Threshold: {min_cost_row['threshold']:.2f}")
    print(f"   Recall (Default Catch Rate): {min_cost_row['recall']:.2%}")
    print(f"   Precision: {min_cost_row['precision']:.2%}")
    print(f"   Missed Defaults: {int(min_cost_row['fn'])}")
    
    # Maximum F1 threshold
    max_f1_row = threshold_df.loc[threshold_df['f1'].idxmax()]
    optimal_thresholds['max_f1'] = {
        'threshold': max_f1_row['threshold'],
        'recall': max_f1_row['recall'],
        'precision': max_f1_row['precision'],
        'f1': max_f1_row['f1']
    }
    print(f"\n2. MAXIMUM F1 SCORE (Balanced):")
    print(f"   Threshold: {max_f1_row['threshold']:.2f}")
    print(f"   F1 Score: {max_f1_row['f1']:.2%}")
    print(f"   Recall: {max_f1_row['recall']:.2%}")
    print(f"   Precision: {max_f1_row['precision']:.2%}")
    
    # High recall threshold (catch 80% of defaults)
    high_recall_df = threshold_df[threshold_df['recall'] >= 0.80]
    if not high_recall_df.empty:
        high_recall_row = high_recall_df.loc[high_recall_df['precision'].idxmax()]
        optimal_thresholds['high_recall_80'] = {
            'threshold': high_recall_row['threshold'],
            'recall': high_recall_row['recall'],
            'precision': high_recall_row['precision'],
            'f1': high_recall_row['f1']
        }
        print(f"\n3. HIGH RECALL (>=80% Default Catch Rate):")
        print(f"   Threshold: {high_recall_row['threshold']:.2f}")
        print(f"   Recall: {high_recall_row['recall']:.2%}")
        print(f"   Precision: {high_recall_row['precision']:.2%}")
    
    # Very high recall threshold (catch 90% of defaults)
    very_high_recall_df = threshold_df[threshold_df['recall'] >= 0.90]
    if not very_high_recall_df.empty:
        very_high_recall_row = very_high_recall_df.loc[very_high_recall_df['precision'].idxmax()]
        optimal_thresholds['high_recall_90'] = {
            'threshold': very_high_recall_row['threshold'],
            'recall': very_high_recall_row['recall'],
            'precision': very_high_recall_row['precision'],
            'f1': very_high_recall_row['f1']
        }
        print(f"\n4. VERY HIGH RECALL (>=90% Default Catch Rate):")
        print(f"   Threshold: {very_high_recall_row['threshold']:.2f}")
        print(f"   Recall: {very_high_recall_row['recall']:.2%}")
        print(f"   Precision: {very_high_recall_row['precision']:.2%}")
    
    print("="*60)
    
    # Save optimal thresholds to CSV
    optimal_df = pd.DataFrame([
        {'objective': k, **v} for k, v in optimal_thresholds.items()
    ])
    optimal_save_path = save_path / "optimal_thresholds.csv"
    optimal_df.to_csv(optimal_save_path, index=False)
    print(f"Optimal thresholds saved to: {optimal_save_path}")
    
    return optimal_thresholds


def compare_thresholds(model, X, y_true, model_type: str, optimized_threshold: float = None):
    """
    Compare default threshold (0.5) vs optimized threshold and save results.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict_proba method
    X : array-like
        Feature matrix
    y_true : array-like
        True labels
    model_type : str
        Type of model ('xgb' or 'lgbm')
    optimized_threshold : float, optional
        Optimized threshold to compare. If None, will calculate minimum cost threshold.
        
    Returns:
    --------
    dict: Comparison results including business impact
    """
    save_path = get_model_save_path(model_type)
    save_path.mkdir(parents=True, exist_ok=True)
    
    y_proba = model.predict_proba(X)[:, 1]
    
    # If no threshold provided, calculate optimal one
    if optimized_threshold is None:
        threshold_df = find_optimal_threshold(model, X, y_true, model_type)
        optimized_threshold = threshold_df.loc[threshold_df['total_cost'].idxmin(), 'threshold']
    
    y_pred_optimized = (y_proba >= optimized_threshold).astype(int)
    y_pred_default = (y_proba >= 0.5).astype(int)
    
    print(f"\nCOMPARISON: Default Threshold (0.5) vs Optimized Threshold ({optimized_threshold:.2f})")
    print("="*70)
    
    # Default threshold results
    print("\nDefault Threshold (0.5):")
    report_default = classification_report(y_true, y_pred_default, output_dict=True)
    print(classification_report(y_true, y_pred_default))
    cm_default = confusion_matrix(y_true, y_pred_default)
    print(f"Confusion Matrix:\n{cm_default}")
    
    # Optimized threshold results
    print(f"\nOptimized Threshold ({optimized_threshold:.2f}):")
    report_optimized = classification_report(y_true, y_pred_optimized, output_dict=True)
    print(classification_report(y_true, y_pred_optimized))
    cm_optimized = confusion_matrix(y_true, y_pred_optimized)
    print(f"Confusion Matrix:\n{cm_optimized}")
    
    # Business impact comparison
    fn_default = cm_default[1, 0]
    fn_optimized = cm_optimized[1, 0]
    defaults_saved = fn_default - fn_optimized
    
    print("\n" + "="*70)
    print("BUSINESS IMPACT")
    print("="*70)
    print(f"Missed defaults with 0.5 threshold: {fn_default}")
    print(f"Missed defaults with {optimized_threshold:.2f} threshold: {fn_optimized}")
    if fn_default > 0:
        improvement = defaults_saved / fn_default * 100
        print(f"Additional defaults caught: {defaults_saved} ({improvement:.1f}% improvement)")
    
    # Save comparison results
    comparison_results = {
        'default_threshold': {
            'threshold': 0.5,
            'confusion_matrix': cm_default.tolist(),
            'classification_report': report_default,
            'false_negatives': int(fn_default)
        },
        'optimized_threshold': {
            'threshold': optimized_threshold,
            'confusion_matrix': cm_optimized.tolist(),
            'classification_report': report_optimized,
            'false_negatives': int(fn_optimized)
        },
        'business_impact': {
            'defaults_saved': int(defaults_saved),
            'improvement_pct': float(defaults_saved / fn_default * 100) if fn_default > 0 else 0
        }
    }
    
    # Save comparison summary to CSV
    comparison_df = pd.DataFrame([
        {
            'metric': 'threshold',
            'default': 0.5,
            'optimized': optimized_threshold
        },
        {
            'metric': 'false_negatives',
            'default': fn_default,
            'optimized': fn_optimized
        },
        {
            'metric': 'false_positives',
            'default': cm_default[0, 1],
            'optimized': cm_optimized[0, 1]
        },
        {
            'metric': 'recall_class_1',
            'default': report_default.get('1', report_default.get(1, {})).get('recall', 0),
            'optimized': report_optimized.get('1', report_optimized.get(1, {})).get('recall', 0)
        },
        {
            'metric': 'precision_class_1',
            'default': report_default.get('1', report_default.get(1, {})).get('precision', 0),
            'optimized': report_optimized.get('1', report_optimized.get(1, {})).get('precision', 0)
        },
        {
            'metric': 'f1_class_1',
            'default': report_default.get('1', report_default.get(1, {})).get('f1-score', 0),
            'optimized': report_optimized.get('1', report_optimized.get(1, {})).get('f1-score', 0)
        }
    ])
    comparison_save_path = save_path / "threshold_comparison.csv"
    comparison_df.to_csv(comparison_save_path, index=False)
    print(f"\nThreshold comparison saved to: {comparison_save_path}")
    
    # Save confusion matrices
    cm_default_df = pd.DataFrame(cm_default, 
                                  index=['Actual_0', 'Actual_1'], 
                                  columns=['Predicted_0', 'Predicted_1'])
    cm_optimized_df = pd.DataFrame(cm_optimized, 
                                    index=['Actual_0', 'Actual_1'], 
                                    columns=['Predicted_0', 'Predicted_1'])
    
    cm_default_df.to_csv(save_path / "confusion_matrix_default.csv")
    cm_optimized_df.to_csv(save_path / "confusion_matrix_optimized.csv")
    print(f"Confusion matrices saved to: {save_path}")
    
    return comparison_results


def run_full_evaluation(model, X, y, model_type: str, dataset_name: str = "Test", 
                        cost_fn: float = 10, cost_fp: float = 1):
    """
    Run complete model evaluation pipeline and save all results.
    
    Parameters:
    -----------
    model : estimator
        Trained model with predict_proba method
    X : array-like
        Feature matrix
    y : array-like
        True labels
    model_type : str
        Type of model ('xgb' or 'lgbm')
    dataset_name : str
        Name of the dataset
    cost_fn : float
        Cost of False Negative
    cost_fp : float
        Cost of False Positive
        
    Returns:
    --------
    dict: Complete evaluation results
    """
    print(f"\n{'='*70}")
    print(f"FULL EVALUATION PIPELINE - {model_type.upper()} MODEL")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Step 1: Basic evaluation with ROC curve
    print("Step 1: Evaluating model performance...")
    results['auc'] = evaluate_model(model, X, y, model_type, dataset_name)
    
    # Step 2: Find optimal thresholds
    print("\nStep 2: Finding optimal thresholds...")
    threshold_df = find_optimal_threshold(model, X, y, model_type, cost_fn, cost_fp)
    results['threshold_analysis'] = threshold_df
    
    # Step 3: Visualize threshold analysis
    print("\nStep 3: Creating visualizations...")
    visualize_threshold_analysis(threshold_df, model_type)
    
    # Step 4: Get optimal thresholds for different objectives
    print("\nStep 4: Calculating optimal thresholds...")
    results['optimal_thresholds'] = get_optimal_thresholds(threshold_df, model_type)
    
    # Step 5: Compare thresholds
    print("\nStep 5: Comparing threshold performance...")
    optimal_thresh = results['optimal_thresholds']['min_cost']['threshold']
    results['comparison'] = compare_thresholds(model, X, y, model_type, optimal_thresh)
    
    save_path = get_model_save_path(model_type)
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE - All results saved to: {save_path}")
    print(f"{'='*70}\n")
    
    return results
