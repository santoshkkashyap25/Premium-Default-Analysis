import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

def train_cost_sensitive_model(X_train, y_train, cost_fp=10, cost_fn=500):
    """
    Train model with custom cost matrix based on financial implications.
    """
    print(f"\nTraining cost-sensitive model...")
    print(f"   Cost FP (false alarm): ${cost_fp}")
    print(f"   Cost FN (missed non-payer): ${cost_fn}")

    n_samples = len(y_train)
    n_class_0 = np.sum(y_train == 0)
    n_class_1 = np.sum(y_train == 1)

    weight_0 = cost_fn * (n_samples / (2 * max(n_class_0, 1)))
    weight_1 = cost_fp * (n_samples / (2 * max(n_class_1, 1)))

    total_weight = weight_0 + weight_1
    weight_0 = weight_0 / total_weight * 2
    weight_1 = weight_1 / total_weight * 2

    print(f"   Class 0 weight: {weight_0:.2f}")
    print(f"   Class 1 weight: {weight_1:.2f}")

    try:
        import xgboost as xgb
        scale_pos_weight = (cost_fn * n_class_0) / (cost_fp * max(n_class_1, 1))

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
    Create stacking ensemble combining Random Forest, Logistic Regression, XGBoost, and LightGBM.
    """
    print(f"\nBuilding stacking ensemble...")

    base_models = []

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    base_models.append(('rf', rf))

    lr = LogisticRegression(
        C=0.1,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    base_models.append(('lr', lr))

    try:
        import xgboost as xgb
        scale_pos_weight = np.sum(y_train == 1) / max(np.sum(y_train == 0), 1)
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
        print("   XGBoost not available for ensemble")

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

    meta_learner = LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )

    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    print(f"   Base models: {len(base_models)}")
    print(f"   Meta-learner: Logistic Regression")

    stacking.fit(X_train, y_train)
    return stacking
