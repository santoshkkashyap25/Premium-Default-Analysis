import pandas as pd
import numpy as np
import pickle
import warnings
import argparse
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

from src.preprocessing import (
    load_and_clean_data,
    create_advanced_features,
    split_data,
    handle_outliers,
    create_binned_features,
    preprocess_for_trees,
    preprocess_for_logreg,
    find_optimal_threshold
)
from src.evaluation import evaluate_with_business_metrics
from src.tuning import apply_smote, tune_with_optuna, calibrate_probabilities
from src.training import train_cost_sensitive_model, create_stacking_ensemble


def deployment_summary(tiers_df, results_df):
    """Log risk tier action plan and financial impact."""
    print("\n" + "="*75)
    print("RISK TIER ACTIONS & DEPLOYMENT SUMMARY")
    print("="*75)

    tier_counts = tiers_df['risk_tier'].value_counts()
    for tier in ['High Risk', 'Medium Risk', 'Low-Medium Risk', 'Low Risk']:
        if tier in tier_counts.index:
            count = tier_counts[tier]
            pct = count / len(tiers_df) * 100
            action = tiers_df[tiers_df['risk_tier'] == tier]['intervention_action'].iloc[0]
            cost = tiers_df[tiers_df['risk_tier'] == tier]['intervention_cost'].iloc[0]
            print(f"\n   {tier}: {count} customers ({pct:.1f}%)")
            print(f"      Action: {action}")
            print(f"      Cost per customer: ${cost}")
            print(f"      Total tier cost: ${count * cost:,.0f}")

    if not results_df.empty:
        best_model = results_df.iloc[0]
        print("\nEXPECTED MONTHLY FINANCIAL IMPACT:")
        print(f"   • Best Model: {best_model['Model']}")
        print(f"   • Net Benefit: ${best_model['Net_Benefit']:,.0f}")
        print(f"   • ROI: {best_model['ROI']:.1f}%")
        print(f"   • Projected Annual Benefit: ${best_model['Net_Benefit'] * 12:,.0f}")


def run_training(filepath, run_tuning=True, fast_mode=False, cost_fp=10, cost_fn=500):
    project_root = Path(__file__).resolve().parent

    print("\n" + "="*75)
    print("INSURANCE DEFAULT PREDICTION MODEL TRAINING")
    print("="*75)

    print("\n[Step 1/8] Loading & cleaning data...")
    df = load_and_clean_data(filepath)
    print(f"   Loaded {len(df)} records")

    print("\n[Step 2/8] Creating advanced interaction features...")
    df = create_advanced_features(df)

    print("\n[Step 3/8] Splitting data into Train, Validation, and Test sets...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\n[Step 4/8] Handling outliers & creating binned features...")
    X_train, X_val, X_test, y_train, y_val, y_test = handle_outliers(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    X_train, X_val, X_test = create_binned_features(X_train, X_val, X_test)

    print("\n[Step 5/8] Preprocessing for Tree & Linear Models...")
    X_train_tree, X_val_tree, X_test_tree = preprocess_for_trees(X_train, X_val, X_test)
    X_train_lr, X_val_lr, X_test_lr, scaler = preprocess_for_logreg(X_train, X_val, X_test)

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    with open(models_dir / "preprocessing_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print(f"\nClass Distribution (Training):")
    print(f"   Class 0 (Non-payers): {np.sum(y_train==0)} ({np.mean(y_train==0)*100:.1f}%)")
    print(f"   Class 1 (On-time):    {np.sum(y_train==1)} ({np.mean(y_train==1)*100:.1f}%)")

    results = []
    models_dict = {}

    print("\n" + "="*75)
    print("[Step 6/8] TRAINING & EVALUATING MODEL SUITE")
    print("="*75)

    # 1. Baseline Logistic Regression
    print("\n--- Logistic Regression (Baseline) ---")
    lr_base = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_base.fit(X_train_lr, y_train)
    models_dict['lr_base'] = lr_base
    res, _ = evaluate_with_business_metrics(lr_base, X_test_lr, y_test, "Logistic Regression (Baseline)")
    results.append(res)

    # 2. Baseline Random Forest
    print("\n--- Random Forest (Baseline) ---")
    rf_base = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                     random_state=42, class_weight='balanced', n_jobs=-1)
    rf_base.fit(X_train_tree, y_train)
    models_dict['rf_base'] = rf_base
    thresh_rf = find_optimal_threshold(rf_base, X_val_tree, y_val)
    res, _ = evaluate_with_business_metrics(rf_base, X_test_tree, y_test, "Random Forest (Threshold Opt)", threshold=thresh_rf)
    results.append(res)

    # 3. SMOTE Logistic Regression
    print("\n--- Logistic Regression + SMOTE ---")
    X_train_lr_smote, y_train_smote = apply_smote(X_train_lr, y_train)
    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_train_lr_smote, y_train_smote)
    models_dict['lr_smote'] = lr_smote
    res, _ = evaluate_with_business_metrics(lr_smote, X_test_lr, y_test, "Logistic Regression (SMOTE)")
    results.append(res)

    # 4. Optuna Hyperparameter Tuning (Optional)
    if run_tuning and not fast_mode:
        print("\n--- Optuna Hyperparameter Tuning ---")
        try:
            rf_tuned = tune_with_optuna('random_forest', X_train_tree, y_train, X_val_tree, y_val, timeout=90)
            models_dict['rf_tuned'] = rf_tuned
            thresh_rf_tuned = find_optimal_threshold(rf_tuned, X_val_tree, y_val)
            res, _ = evaluate_with_business_metrics(rf_tuned, X_test_tree, y_test, "Random Forest (Tuned)", threshold=thresh_rf_tuned)
            results.append(res)
        except Exception as e:
            print(f"Optuna tuning skipped: {e}")

    # 5. Balanced Random Forest
    try:
        print("\n--- Balanced Random Forest ---")
        from imblearn.ensemble import BalancedRandomForestClassifier
        brf = BalancedRandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=3, random_state=42, n_jobs=-1)
        brf.fit(X_train_tree, y_train)
        models_dict['brf'] = brf
        thresh_brf = find_optimal_threshold(brf, X_val_tree, y_val)
        res, _ = evaluate_with_business_metrics(brf, X_test_tree, y_test, "Balanced Random Forest", threshold=thresh_brf)
        results.append(res)
    except ImportError:
        pass

    # 6. XGBoost Classifier
    try:
        print("\n--- XGBoost Classifier ---")
        import xgboost as xgb
        scale_pos_weight = np.sum(y_train == 1) / max(np.sum(y_train == 0), 1)
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=scale_pos_weight, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, eval_metric='logloss'
        )
        xgb_model.fit(X_train_tree, y_train)
        models_dict['xgb'] = xgb_model

        thresh_xgb = find_optimal_threshold(xgb_model, X_val_tree, y_val)
        res, _ = evaluate_with_business_metrics(xgb_model, X_test_tree, y_test, "XGBoost", threshold=thresh_xgb)
        results.append(res)

        # 7. Calibrated XGBoost
        print("\n--- Calibrated XGBoost (Isotonic) ---")
        xgb_calibrated = calibrate_probabilities(xgb_model, X_train_tree, y_train, X_val_tree, y_val, method='isotonic')
        models_dict['xgb_calibrated'] = xgb_calibrated
        thresh_cal = find_optimal_threshold(xgb_calibrated, X_val_tree, y_val)
        res, _ = evaluate_with_business_metrics(xgb_calibrated, X_test_tree, y_test, "XGBoost (Calibrated)", threshold=thresh_cal)
        results.append(res)
    except ImportError:
        print("XGBoost not available")

    # 8. LightGBM Classifier
    try:
        print("\n--- LightGBM Classifier ---")
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, class_weight='balanced',
            random_state=42, n_jobs=-1, verbose=-1
        )
        lgb_model.fit(X_train_tree, y_train)
        models_dict['lgb'] = lgb_model
        thresh_lgb = find_optimal_threshold(lgb_model, X_val_tree, y_val)
        res, _ = evaluate_with_business_metrics(lgb_model, X_test_tree, y_test, "LightGBM", threshold=thresh_lgb)
        results.append(res)
    except ImportError:
        pass

    # 9. Cost-Sensitive Model
    print("\n--- Cost-Sensitive Model ---")
    cost_sensitive_model = train_cost_sensitive_model(X_train_tree, y_train, cost_fp=cost_fp, cost_fn=cost_fn)
    models_dict['cost_sensitive'] = cost_sensitive_model
    thresh_cost = find_optimal_threshold(cost_sensitive_model, X_val_tree, y_val)
    res, _ = evaluate_with_business_metrics(cost_sensitive_model, X_test_tree, y_test, "Cost-Sensitive Model", threshold=thresh_cost)
    results.append(res)

    # 10. Stacking Ensemble
    print("\n--- Stacking Ensemble ---")
    stacking_model = create_stacking_ensemble(X_train_tree, y_train)
    models_dict['stacking'] = stacking_model
    thresh_stacking = find_optimal_threshold(stacking_model, X_val_tree, y_val)
    res_stacking, tiers_final = evaluate_with_business_metrics(stacking_model, X_test_tree, y_test, "Stacking Ensemble", threshold=thresh_stacking)
    results.append(res_stacking)

    # Step 7: Ranking & Model Selection
    print("\n" + "="*75)
    print("[Step 7/8] FINAL COMPARISON & MODEL SELECTION")
    print("="*75)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Net_Benefit', ascending=False)

    display_cols = ['Model', 'Threshold', 'Test_F1_Class0', 'Test_Recall_Class0', 'Intervention_Cost', 'Net_Benefit', 'ROI']
    print("\n" + results_df[display_cols].to_string(index=False))

    best_row = results_df.iloc[0]
    best_model_name = best_row['Model']
    print(f"\n[BEST MODEL SELECTED]: {best_model_name}")
    print(f"   • Net Benefit: ${best_row['Net_Benefit']:,.0f}")
    print(f"   • ROI:         {best_row['ROI']:.1f}%")
    print(f"   • F1 (Class 0):{best_row['Test_F1_Class0']:.4f}")

    best_model_obj = stacking_model
    for key, model_obj in models_dict.items():
        if key in best_model_name.lower():
            best_model_obj = model_obj
            break

    # Save outputs & artifacts
    print("\n" + "="*75)
    print("[Step 8/8] SAVING ARTIFACTS & DEPLOYMENT SUMMARY")
    print("="*75)

    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    results_df.to_csv(outputs_dir / "model_results.csv", index=False)
    results_df.to_csv(outputs_dir / "advanced_model_results.csv", index=False)
    print(f"   • Comparative metrics saved to 'outputs/model_results.csv'")

    with open(models_dir / "best_model.pkl", "wb") as f:
        pickle.dump(best_model_obj, f)
    with open(models_dir / "best_model_advanced.pkl", "wb") as f:
        pickle.dump(best_model_obj, f)
    print(f"   • Best model artifact saved to 'models/best_model.pkl' & 'models/best_model_advanced.pkl'")

    deployment_summary(tiers_final, results_df)

    return results_df, best_model_obj


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insurance Premium Payment Default - Model Training Script")
    parser.add_argument("--data", type=str, default=None, help="Path to training dataset CSV file")
    parser.add_argument("--fast", action="store_true", help="Fast execution mode (skips Optuna hyperparameter search)")
    parser.add_argument("--tune", action="store_true", help="Force Optuna hyperparameter tuning")
    parser.add_argument("--cost-fp", type=float, default=10, help="Cost of false alarm (default: $10)")
    parser.add_argument("--cost-fn", type=float, default=500, help="Cost of missed default (default: $500)")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_path = Path(args.data) if args.data else project_root / "data" / "premium_dataset.csv"

    run_training(
        filepath=data_path,
        run_tuning=args.tune or (not args.fast),
        fast_mode=args.fast,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn
    )
