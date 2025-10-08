import optuna
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_auc_score, f1_score, precision_recall_curve,
                            precision_recall_fscore_support)
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import warnings
warnings.filterwarnings("ignore")

def tune_with_optuna(model_type, X_train, y_train, X_val, y_val, timeout=180):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        if model_type == 'decision_tree':
            params = {
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'class_weight': 'balanced',
                'random_state': 42
            }
            model = DecisionTreeClassifier(**params)

        elif model_type == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                'max_depth': trial.suggest_int('max_depth', 8, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'class_weight': 'balanced',
                'random_state': 42,
                'n_jobs': -1
            }
            model = RandomForestClassifier(**params)

        elif model_type == 'logistic':
            params = {
                'C': trial.suggest_loguniform('C', 1e-3, 10),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': 'liblinear',
                'class_weight': 'balanced',
                'max_iter': 1000,
                'random_state': 42
            }
            model = LogisticRegression(**params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Optimize for F1-score of minority class (class 0)
        f1_class0 = f1_score(y_val, y_pred, pos_label=0, average='binary')

        return f1_class0

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    print(f"  Best params: {best_params}")
    print(f"  Best F1 (Class 0): {study.best_value:.4f}")

    # Train final model with best params
    if model_type == 'decision_tree':
        best_model = DecisionTreeClassifier(**best_params, class_weight='balanced', random_state=42)
    elif model_type == 'random_forest':
        best_model = RandomForestClassifier(**best_params, class_weight='balanced',
                                           random_state=42, n_jobs=-1)
    elif model_type == 'logistic':
        best_model = LogisticRegression(**best_params, class_weight='balanced',
                                       max_iter=1000, random_state=42)

    best_model.fit(X_train, y_train)
    return best_model



def apply_smote(X_train, y_train):
    try:
        print(f"  Original distribution: Class 0={np.sum(y_train==0)}, Class 1={np.sum(y_train==1)}")

        smote = SMOTE(random_state=42, sampling_strategy='auto')
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"  After SMOTE: Class 0={np.sum(y_resampled==0)}, Class 1={np.sum(y_resampled==1)}")
        return X_resampled, y_resampled
    except ImportError:
        print("  imbalanced-learn not installed. Install with: pip install imbalanced-learn")
        return X_train, y_train


def get_class_weights(y_train):
    """Calculate balanced class weights"""
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))

    print(f"Calculated class weights: {class_weight_dict}")
    return class_weight_dict


def calibrate_probabilities(model, X_train, y_train, X_val, y_val, method='isotonic'):
    """
    Calibrate model probabilities using isotonic or sigmoid calibration
    """
    print(f"\nCalibrating probabilities using {method} regression...")

    # Check calibration before
    y_prob_before = model.predict_proba(X_val)[:, 1]

    # Calibrate
    calibrated_model = CalibratedClassifierCV(
        model,
        method=method,
        cv='prefit'
    )
    calibrated_model.fit(X_train, y_train)

    # Check calibration after
    y_prob_after = calibrated_model.predict_proba(X_val)[:, 1]

    # Plot calibration curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Before calibration
    prob_true_before, prob_pred_before = calibration_curve(
        y_val, y_prob_before, n_bins=10, strategy='uniform'
    )
    ax1.plot(prob_pred_before, prob_true_before, marker='o', label='Before')
    ax1.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('True Probability')
    ax1.set_title('Before Calibration')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # After calibration
    prob_true_after, prob_pred_after = calibration_curve(
        y_val, y_prob_after, n_bins=10, strategy='uniform'
    )
    ax2.plot(prob_pred_after, prob_true_after, marker='o', label='After', color='green')
    ax2.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('True Probability')
    ax2.set_title('After Calibration')
    ax2.legend()
    ax2.grid(alpha=0.3)

    project_root = Path(__file__).resolve().parent.parent
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    plot_path = outputs_dir / "calibration_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Calibration plot saved as '{plot_path}'")

    return calibrated_model