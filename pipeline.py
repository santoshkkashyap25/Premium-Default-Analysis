
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
    create_binned_features,preprocess_for_trees,preprocess_for_logreg)
from src.evaluation import evaluate_model
from src.tuning import apply_smote , tune_with_optuna
from pathlib import Path

def main(filepath):
    print("\nLoading and cleaning data...")
    df = load_and_clean_data(filepath)
    print(f"   Loaded {len(df)} records")

    print("\nEngineering features...")
    df = engineer_features(df)

    print("\nSplitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("\nHandling outliers...")
    X_train, X_val, X_test, y_train, y_val, y_test = handle_outliers(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    print("\nCreating binned features...")
    X_train, X_val, X_test = create_binned_features(X_train, X_val, X_test)

    print("\nPreprocessing for models...")
    X_train_tree, X_val_tree, X_test_tree = preprocess_for_trees(X_train, X_val, X_test)
    X_train_lr, X_val_lr, X_test_lr, scaler = preprocess_for_logreg(X_train, X_val, X_test)

    # Class distribution
    print(f"\nClass Distribution (Training):")
    print(f"   Class 0 (Non-payers): {np.sum(y_train==0)} ({np.mean(y_train==0)*100:.1f}%)")
    print(f"   Class 1 (On-time):    {np.sum(y_train==1)} ({np.mean(y_train==1)*100:.1f}%)")
    print(f"   Imbalance Ratio: 1:{np.sum(y_train==1)/np.sum(y_train==0):.1f}")

    # Train models
    print("\n" + "="*75)
    print("TRAINING MODELS")
    print("="*75)

    results = []
    models_dict = {}

    # ========== BASELINE MODELS ==========
    print("\n" + "-"*75)
    print("BASELINE MODELS")
    print("-"*75)

    # Logistic Regression (Baseline)
    print("\nLogistic Regression (Baseline)...")
    lr_base = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_base.fit(X_train_lr, y_train)
    models_dict['lr_base'] = lr_base
    results.append(evaluate_model(lr_base, X_val_lr, y_val, X_test_lr, y_test,
                                  "Logistic Regression (Baseline)"))

    # Random Forest (Baseline)
    print("\nRandom Forest (Baseline)...")
    rf_base = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                    random_state=42, class_weight='balanced', n_jobs=-1)
    rf_base.fit(X_train_tree, y_train)
    models_dict['rf_base'] = rf_base
    results.append(evaluate_model(rf_base, X_val_tree, y_val, X_test_tree, y_test,
                                  "Random Forest (Baseline)"))

    # ========== THRESHOLD OPTIMIZED ==========
    print("\n" + "-"*75)
    print("THRESHOLD OPTIMIZATION")
    print("-"*75)

    # Random Forest with Threshold
    print("\nRandom Forest (Threshold Optimized)...")
    results.append(evaluate_model(rf_base, X_val_tree, y_val, X_test_tree, y_test,
                                  "Random Forest", use_threshold_opt=True))

    # ========== SMOTE ==========
    print("\n" + "-"*75)
    print("SMOTE OVERSAMPLING")
    print("-"*75)

    # Logistic Regression + SMOTE
    print("\nLogistic Regression + SMOTE...")
    X_train_lr_smote, y_train_smote = apply_smote(X_train_lr, y_train)
    lr_smote = LogisticRegression(max_iter=1000, random_state=42)
    lr_smote.fit(X_train_lr_smote, y_train_smote)
    models_dict['lr_smote'] = lr_smote
    results.append(evaluate_model(lr_smote, X_val_lr, y_val, X_test_lr, y_test,
                                  "Logistic Regression (SMOTE)"))

    # ========== HYPERPARAMETER TUNING ==========
    print("\n" + "-"*75)
    print("HYPERPARAMETER TUNING (with Optuna)")
    print("-"*75)

    # Tuned Decision Tree
    print("\nDecision Tree (Tuned)...")
    dt_tuned = tune_with_optuna('decision_tree', X_train_tree, y_train,
                                X_val_tree, y_val, timeout=120)
    models_dict['dt_tuned'] = dt_tuned
    results.append(evaluate_model(dt_tuned, X_val_tree, y_val, X_test_tree, y_test,
                                  "Decision Tree (Tuned)"))

    # Tuned Random Forest
    print("\nRandom Forest (Tuned)...")
    rf_tuned = tune_with_optuna('random_forest', X_train_tree, y_train,
                                X_val_tree, y_val, timeout=180)
    models_dict['rf_tuned'] = rf_tuned
    results.append(evaluate_model(rf_tuned, X_val_tree, y_val, X_test_tree, y_test,
                                  "Random Forest (Tuned)"))

    # Tuned Random Forest + Threshold
    print("\nRandom Forest (Tuned + Threshold)...")
    results.append(evaluate_model(rf_tuned, X_val_tree, y_val, X_test_tree, y_test,
                                  "Random Forest (Tuned)", use_threshold_opt=True))

    # Balanced Random Forest
    try:
        print("\nBalanced Random Forest...")
        from imblearn.ensemble import BalancedRandomForestClassifier
        brf = BalancedRandomForestClassifier(n_estimators=200, max_depth=12,
                                            min_samples_leaf=3, random_state=42, n_jobs=-1)
        brf.fit(X_train_tree, y_train)
        models_dict['brf'] = brf
        results.append(evaluate_model(brf, X_val_tree, y_val, X_test_tree, y_test,
                                      "Balanced Random Forest"))
    except ImportError:
        print(" Balanced RF not available")

    # XGBoost
    try:
        print("\nXGBoost...")
        import xgboost as xgb
        scale_pos_weight = np.sum(y_train==1) / np.sum(y_train==0)
        xgb_model = xgb.XGBClassifier(
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
        xgb_model.fit(X_train_tree, y_train)
        models_dict['xgb'] = xgb_model
        results.append(evaluate_model(xgb_model, X_val_tree, y_val, X_test_tree, y_test,
                                      "XGBoost"))

        # XGBoost + Threshold
        print("\nXGBoost (Threshold Optimized)...")
        results.append(evaluate_model(xgb_model, X_val_tree, y_val, X_test_tree, y_test,
                                      "XGBoost", use_threshold_opt=True))
    except ImportError:
        print("  XGBoost not available")

    # LightGBM
    try:
        print("\nLightGBM...")
        import lightgbm as lgb
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train_tree, y_train)
        models_dict['lgb'] = lgb_model
        results.append(evaluate_model(lgb_model, X_val_tree, y_val, X_test_tree, y_test,
                                      "LightGBM"))
    except ImportError:
        print("  LightGBM not available")

    # ========== ENSEMBLE ==========
    print("\n" + "-"*75)
    print("ENSEMBLE MODELS")
    print("-"*75)

    # Voting Classifier (Top 3 models)
    print("\nVoting Ensemble (Top 3)...")
    estimators = []
    if 'rf_tuned' in models_dict:
        estimators.append(('rf', models_dict['rf_tuned']))
    if 'xgb' in models_dict:
        estimators.append(('xgb', models_dict['xgb']))
    if 'lgb' in models_dict:
        estimators.append(('lgb', models_dict['lgb']))
    elif 'brf' in models_dict:
        estimators.append(('brf', models_dict['brf']))

    if len(estimators) >= 2:
        voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        voting.fit(X_train_tree, y_train)
        models_dict['voting'] = voting
        results.append(evaluate_model(voting, X_val_tree, y_val, X_test_tree, y_test,
                                      "Voting Ensemble"))

        # Voting + Threshold
        print("\nVoting Ensemble (Threshold Optimized)...")
        results.append(evaluate_model(voting, X_val_tree, y_val, X_test_tree, y_test,
                                      "Voting Ensemble", use_threshold_opt=True))

    # ========== RESULTS COMPARISON ==========
    print("\n" + "="*75)
    print("MODEL COMPARISON")
    print("="*75)

    results_df = pd.DataFrame(results)

    # Sort by Test F1 (Class 0)
    results_df = results_df.sort_values('Test_F1_Class0', ascending=False)

    display_cols = ['Model', 'Threshold', 'Test_F1_Class0', 'Test_Precision_Class0',
                   'Test_Recall_Class0', 'Test_ROC_AUC', 'Test_Accuracy']

    print("\n" + results_df[display_cols].to_string(index=False))

    best_model_row = results_df.iloc[0]
    print(f"\nBest Model: {best_model_row['Model']}")
    print(f"\n   Key Metrics:")
    print(f"   • Test F1 (Class 0):        {best_model_row['Test_F1_Class0']:.4f}")
    print(f"   • Test Precision (Class 0): {best_model_row['Test_Precision_Class0']:.4f}")
    print(f"   • Test Recall (Class 0):    {best_model_row['Test_Recall_Class0']:.4f}")
    print(f"   • Test ROC-AUC:             {best_model_row['Test_ROC_AUC']:.4f}")
    print(f"   • Optimal Threshold:        {best_model_row['Threshold']:.3f}")

    recall = best_model_row['Test_Recall_Class0']
    precision = best_model_row['Test_Precision_Class0']
    print(f"   • Catches {recall*100:.1f}% of non-payers before they lapse")
    print(f"   • {precision*100:.1f}% of flagged customers are actual non-payers")
    print(f"   • {(1-precision)*100:.1f}% of interventions are false alarms")

    # ========== FEATURE IMPORTANCE ==========
    if 'rf_tuned' in models_dict:
        print("\n" + "="*75)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*75)

        feature_names = X_train_tree.columns
        importances = models_dict['rf_tuned'].feature_importances_

        feature_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_imp_df.head(10).to_string(index=False))

        # Visualize
        plt.figure(figsize=(10, 6))
        top_features = feature_imp_df.head(10)
        plt.barh(range(len(top_features)), top_features['Importance'])
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances (Random Forest Tuned)')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')


    # ========== SAVE RESULTS ==========
    project_root = Path(__file__).resolve().parent
    output_path = project_root / "outputs" / "model_comparison_results.csv"

    results_df.to_csv(output_path, index=False)

    # Save best model
    import pickle
    project_root = Path(__file__).resolve().parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    best_model_name = best_model_row['Model'].split('(')[0].strip().lower().replace(' ', '_')

    # Find the actual model object
    for key, model in models_dict.items():
        if key in best_model_name or best_model_name in key:            
            model_path = models_dir / f'best_model_{key}.pkl'
            scaler_path = models_dir / 'preprocessing_scaler.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Best model saved as 'best_model_{key}.pkl'")

            # Save preprocessing objects
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print("Scaler saved as 'preprocessing_scaler.pkl'")
            break

    return results_df, models_dict

if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "premium_dataset.csv"

    results_df, models_dict = main(data_path)
