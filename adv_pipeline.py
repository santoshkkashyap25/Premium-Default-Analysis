
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_auc_score, f1_score, precision_recall_curve,
                            precision_recall_fscore_support)
import warnings
warnings.filterwarnings("ignore")
import sys
from src.preprocessing import ( load_and_clean_data , engineer_features, split_data , handle_outliers ,
    create_binned_features,preprocess_for_trees,preprocess_for_logreg, create_advanced_features,
    find_optimal_threshold)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from src.evaluation import evaluate_model , evaluate_with_business_metrics
from src.tuning import apply_smote , tune_with_optuna, calibrate_probabilities
from pathlib import Path

def train_cost_sensitive_model(X_train, y_train, cost_fp=10, cost_fn=500):
    """
    Train model with custom cost matrix

    Args:
        cost_fp: Cost of false positive (unnecessary intervention)
        cost_fn: Cost of false negative (missed non-payer)
    """
    print(f"\nTraining cost-sensitive model...")
    print(f"   Cost FP (false alarm): ${cost_fp}")
    print(f"   Cost FN (missed non-payer): ${cost_fn}")

    # Calculate class weights based on costs
    # For imbalanced data: weight = cost * (n_samples / (n_classes * n_class_samples))
    n_samples = len(y_train)
    n_class_0 = np.sum(y_train == 0)
    n_class_1 = np.sum(y_train == 1)

    # Cost-based weights
    weight_0 = cost_fn * (n_samples / (2 * n_class_0))
    weight_1 = cost_fp * (n_samples / (2 * n_class_1))

    # Normalize weights
    total_weight = weight_0 + weight_1
    weight_0 = weight_0 / total_weight * 2
    weight_1 = weight_1 / total_weight * 2

    print(f"   Class 0 weight: {weight_0:.2f}")
    print(f"   Class 1 weight: {weight_1:.2f}")

    try:
        import xgboost as xgb

        # XGBoost with scale_pos_weight
        scale_pos_weight = (cost_fn * n_class_0) / (cost_fp * n_class_1)

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        return model

    except ImportError:
        # Fallback to Random Forest with class_weight
        class_weight = {0: weight_0, 1: weight_1}

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        return model


def create_stacking_ensemble(X_train, y_train):
    """
    Create stacking ensemble with multiple base models
    """
    print(f"\nBuilding stacking ensemble...")

    # Base models
    base_models = []

    # Model 1: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    base_models.append(('rf', rf))

    # Model 2: Logistic Regression
    lr = LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    base_models.append(('lr', lr))

    # Model 3: XGBoost
    try:
        import xgboost as xgb
        scale_pos_weight = np.sum(y_train == 1) / np.sum(y_train == 0)
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        base_models.append(('xgb', xgb_model))
    except ImportError:
        print("XGBoost not available, using RF + LR only")

    # Model 4: LightGBM
    try:
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        base_models.append(('lgb', lgb_model))
    except ImportError:
        pass

    # Meta-learner: Logistic Regression
    meta_learner = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    # Create stacking ensemble
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False  # Don't pass original features to meta-learner
    )

    print(f"   Base models: {len(base_models)}")
    print(f"   Meta-learner: Logistic Regression")

    stacking.fit(X_train, y_train)

    return stacking

def deployment(tiers_df, results_df):
    print("\nRISK TIER ACTIONS:")
    tier_counts = tiers_df['risk_tier'].value_counts()
    for tier in ['High Risk', 'Medium Risk', 'Low-Medium Risk', 'Low Risk']:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            pct = count / len(tiers_df) * 100
            action = tiers_df[tiers_df['risk_tier']==tier]['intervention_action'].iloc[0]
            cost = tiers_df[tiers_df['risk_tier']==tier]['intervention_cost'].iloc[0]
            print(f"\n   {tier}: {count} customers ({pct:.1f}%)")
            print(f"      Action: {action}")
            print(f"      Cost per customer: ${cost}")
            print(f"      Total cost: ${count * cost:,.0f}")

    print("\nEXPECTED MONTHLY IMPACT:")
    best_model = results_df.iloc[0]
    print(f"   • Net Benefit: ${best_model['Net_Benefit']:,.0f}")
    print(f"   • ROI: {best_model['ROI']:.1f}%")
    print(f"   • Annual Benefit: ${best_model['Net_Benefit'] * 12:,.0f}")


def main_advanced(filepath):
    print("\nLoading data...")
    df = pd.read_csv(filepath)
    df = df.drop('id', axis=1)

    # Handle missing values
    late_cols = ['Count_3-6_months_late', 'Count_6-12_months_late',
                 'Count_more_than_12_months_late']
    for col in late_cols:
        df[col].fillna(0, inplace=True)
    df['application_underwriting_score'].fillna(
        df['application_underwriting_score'].median(), inplace=True
    )

    # feature engineering
    print("\nAdvanced feature engineering...")
    df = create_advanced_features(df)

    # Split data
    print("\nSplitting data...")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = handle_outliers(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    X_train, X_val, X_test = create_binned_features(X_train, X_val, X_test)

    # Preprocess
    print("\nPreprocessing...")
    X_train_tree, X_val_tree, X_test_tree = preprocess_for_trees(X_train, X_val, X_test)
    X_train_lr, X_val_lr, X_test_lr, scaler = preprocess_for_logreg(X_train, X_val, X_test)

    # Results storage
    results = []

    # ========== BASELINE FOR COMPARISON ==========
    try:
        import xgboost as xgb
        scale_pos_weight = np.sum(y_train == 1) / np.sum(y_train == 0)
        xgb_baseline = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        xgb_baseline.fit(X_train_tree, y_train)

        res, _ = evaluate_with_business_metrics(
            xgb_baseline, X_test_tree, y_test,
            "XGBoost (Baseline)", threshold=0.5
        )
        results.append(res)
    except ImportError:
        print("XGBoost not available")

    # ========== CALIBRATED XGBOOST ==========
    print("\n" + "="*75)
    print("PROBABILITY CALIBRATION")
    print("="*75)

    if 'xgb_baseline' in locals():
        xgb_calibrated = calibrate_probabilities(
            xgb_baseline, X_train_tree, y_train, X_val_tree, y_val,
            method='isotonic'
        )

        # Find optimal threshold
        optimal_thresh = find_optimal_threshold(xgb_calibrated, X_val_tree, y_val)

        res, _ = evaluate_with_business_metrics(
            xgb_calibrated, X_test_tree, y_test,
            "XGBoost (Calibrated)", threshold=optimal_thresh
        )
        results.append(res)

    # ========== COST-SENSITIVE MODEL ==========
    print("\n" + "="*75)
    print("COST-SENSITIVE LEARNING")
    print("="*75)

    cost_sensitive_model = train_cost_sensitive_model(
        X_train_tree, y_train,
        cost_fp=10,  # Cost of false alarm
        cost_fn=500  # Cost of missed non-payer
    )

    optimal_thresh = find_optimal_threshold(cost_sensitive_model, X_val_tree, y_val)
    res, _ = evaluate_with_business_metrics(
        cost_sensitive_model, X_test_tree, y_test,
        "Cost-Sensitive Model", threshold=optimal_thresh
    )
    results.append(res)

    # ========== STACKING ENSEMBLE ==========
    print("\n" + "="*75)
    print("STACKING ENSEMBLE")
    print("="*75)

    stacking_model = create_stacking_ensemble(X_train_tree, y_train)

    optimal_thresh = find_optimal_threshold(stacking_model, X_val_tree, y_val)
    res, tiers_final = evaluate_with_business_metrics(
        stacking_model, X_test_tree, y_test,
        "Stacking Ensemble", threshold=optimal_thresh
    )
    results.append(res)

    # ========== COMPARISON ==========
    print("\n" + "="*75)
    print("FINAL COMPARISON")
    print("="*75)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Net_Benefit', ascending=False)

    display_cols = ['Model', 'Test_F1_Class0', 'Test_Recall_Class0',
                   'Intervention_Cost', 'Net_Benefit', 'ROI']
    print("\n" + results_df[display_cols].to_string(index=False))

    # best Model
    best_model = results_df.iloc[0]
    print(f"\nBEST MODEL: {best_model['Model']}")
    print(f"   Net Benefit: ${best_model['Net_Benefit']:,.0f}")
    print(f"   ROI: {best_model['ROI']:.1f}%")
    print(f"   F1 (Class 0): {best_model['Test_F1_Class0']:.4f}")

    # Save
    project_root = Path(__file__).resolve().parent
    output_path = project_root / "outputs" / "advanced_model_results.csv"

    results_df.to_csv(output_path, index=False)
    print("\nResults saved to 'advanced_model_results.csv'")

    # Save best model
    import pickle
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f'best_model_advanced.pkl'
            
    with open(model_path, 'wb') as f:
        pickle.dump(stacking_model, f)
    print("Best model saved as 'best_model_advanced.pkl'")

    # Create deployment guide
    deployment(tiers_final, results_df)

    return results_df, stacking_model



if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "premium_dataset.csv"

    results_df, models_dict = main_advanced(data_path)
