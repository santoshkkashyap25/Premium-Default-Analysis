import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_auc_score, f1_score, precision_recall_curve,
                            precision_recall_fscore_support)
import warnings
warnings.filterwarnings("ignore")


def load_and_clean_data(filepath):
    """Load data and perform initial cleaning"""
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

    return df


def engineer_features(df):
    """Create derived features"""
    df = df.copy()

    df['late_premium'] = (df['Count_3-6_months_late'] +
                          df['Count_6-12_months_late'] +
                          df['Count_more_than_12_months_late'])

    df['age'] = df['age_in_days'] // 365
    df['application_underwriting_score'] = df['application_underwriting_score'] / 100

    return df


def create_advanced_features(df):
    """Create interaction and derived features"""
    df = df.copy()

    # Basic features from original pipeline
    df['late_premium'] = (df['Count_3-6_months_late'] +
                          df['Count_6-12_months_late'] +
                          df['Count_more_than_12_months_late'])

    df['age'] = df['age_in_days'] // 365
    df['application_underwriting_score'] = df['application_underwriting_score'] / 100

    # 1. High cash payment + Late payments (risky combo)
    df['high_cash_late_combo'] = (
        (df['perc_premium_paid_by_cash_credit'] > 0.5) &
        (df['late_premium'] > 2)
    ).astype(int)

    # 2. Low income + High late payments (financial stress indicator)
    df['financial_stress'] = (
        (df['Income'] < df['Income'].quantile(0.25)) &
        (df['late_premium'] > 1)
    ).astype(int)

    # 3. Payment reliability score (ratio of on-time to total premiums)
    df['payment_reliability'] = df['no_of_premiums_paid'] / (
        df['no_of_premiums_paid'] + df['late_premium'] + 1e-5
    )

    # 4. Risk score combining underwriting + payment history
    df['composite_risk'] = (
        (1 - df['application_underwriting_score']) * 0.4 +
        (df['late_premium'] / (df['late_premium'].max() + 1)) * 0.6
    )

    # 5. Age-Income interaction (older high earners more stable)
    df['age_income_interaction'] = df['age'] * np.log1p(df['Income'])

    # 6. Recent late payment indicator (more weight to recent behavior)
    df['recent_late_weighted'] = (
        df['Count_3-6_months_late'] * 3 +
        df['Count_6-12_months_late'] * 2 +
        df['Count_more_than_12_months_late'] * 1
    )

    # 7. Income sufficiency (income relative to payment method)
    df['income_payment_ratio'] = df['Income'] / (
        df['perc_premium_paid_by_cash_credit'] * df['Income'].median() + 1
    )

    # 8. Binary flags for extreme cases
    df['zero_late_payments'] = (df['late_premium'] == 0).astype(int)
    df['chronic_late_payer'] = (df['late_premium'] >= 5).astype(int)
    df['new_customer'] = (df['no_of_premiums_paid'] <= 3).astype(int)

    return df

def split_data(df, test_size=0.2, val_size=0.25, random_state=42):
    """Split data with stratification"""
    df_full_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df['target']
    )

    df_train, df_val = train_test_split(
        df_full_train, test_size=val_size, random_state=random_state,
        stratify=df_full_train['target']
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train['target'].values
    y_val = df_val['target'].values
    y_test = df_test['target'].values

    X_train = df_train.drop('target', axis=1)
    X_val = df_val.drop('target', axis=1)
    X_test = df_test.drop('target', axis=1)

    return X_train, X_val, X_test, y_train, y_val, y_test


def handle_outliers(X_train, X_val, X_test, y_train, y_val, y_test):
    """Remove outliers"""
    lower_bound = X_train['Income'].quantile(0.10)
    upper_bound = X_train['Income'].quantile(0.95)

    train_mask = (X_train['Income'] >= lower_bound) & (X_train['Income'] <= upper_bound)
    X_train_filtered = X_train[train_mask].reset_index(drop=True)
    y_train_filtered = y_train[train_mask]

    X_val_clipped = X_val.copy()
    X_val_clipped['Income'] = X_val_clipped['Income'].clip(lower_bound, upper_bound)

    X_test_clipped = X_test.copy()
    X_test_clipped['Income'] = X_test_clipped['Income'].clip(lower_bound, upper_bound)

    print(f"Training samples: {len(X_train)} -> {len(X_train_filtered)}")
    print(f"Income bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")

    return X_train_filtered, X_val_clipped, X_test_clipped, y_train_filtered, y_val, y_test


def create_binned_features(X_train, X_val, X_test):
    """Create binned versions of continuous features"""
    # Income bins
    income_bins = pd.qcut(X_train['Income'], q=5, duplicates='drop', retbins=True)[1]
    income_labels = list(range(len(income_bins)-1))

    X_train['income_class'] = pd.cut(X_train['Income'], bins=income_bins,
                                     labels=income_labels, include_lowest=True).astype(int)
    X_val['income_class'] = pd.cut(X_val['Income'], bins=income_bins,
                                   labels=income_labels, include_lowest=True).astype(int)
    X_test['income_class'] = pd.cut(X_test['Income'], bins=income_bins,
                                    labels=income_labels, include_lowest=True).astype(int)

    # Age bins
    age_bins = [0, 37.2, 53.4, 69.6, 85.8, 102, float('inf')]
    age_labels = list(range(len(age_bins)-1))

    X_train['age_class'] = pd.cut(X_train['age'], bins=age_bins,
                                  labels=age_labels, include_lowest=True).astype(int)
    X_val['age_class'] = pd.cut(X_val['age'], bins=age_bins,
                                labels=age_labels, include_lowest=True).astype(int)
    X_test['age_class'] = pd.cut(X_test['age'], bins=age_bins,
                                 labels=age_labels, include_lowest=True).astype(int)

    return X_train, X_val, X_test



def preprocess_for_trees(X_train, X_val, X_test):
    """Prepare data for tree-based models"""
    drop_cols = ['Income', 'Count_3-6_months_late', 'Count_6-12_months_late',
                 'Count_more_than_12_months_late', 'age', 'age_in_days']

    res_area_map = {'Urban': 1, 'Rural': 0}
    sourcing_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    def transform(df):
        df = df.drop(drop_cols, axis=1, errors='ignore')
        df['residence_area_type'] = df['residence_area_type'].map(res_area_map)
        df['sourcing_channel'] = df['sourcing_channel'].map(sourcing_map)
        df['income_class'] = df['income_class'].astype(int)
        df['age_class'] = df['age_class'].astype(int)
        return df

    return transform(X_train.copy()), transform(X_val.copy()), transform(X_test.copy())

def preprocess_for_logreg(X_train, X_val, X_test):
    """Prepare data for logistic regression"""
    drop_cols = ['Income', 'Count_3-6_months_late', 'Count_6-12_months_late',
                 'Count_more_than_12_months_late', 'age', 'age_in_days']

    X_train_lr = X_train.drop(drop_cols, axis=1, errors='ignore')
    X_val_lr = X_val.drop(drop_cols, axis=1, errors='ignore')
    X_test_lr = X_test.drop(drop_cols, axis=1, errors='ignore')

    X_train_lr = pd.get_dummies(X_train_lr, columns=['sourcing_channel', 'residence_area_type'],
                                drop_first=True)
    X_val_lr = pd.get_dummies(X_val_lr, columns=['sourcing_channel', 'residence_area_type'],
                              drop_first=True)
    X_test_lr = pd.get_dummies(X_test_lr, columns=['sourcing_channel', 'residence_area_type'],
                               drop_first=True)

    X_train_lr['income_class'] = X_train_lr['income_class'].astype(int)
    X_val_lr['income_class'] = X_val_lr['income_class'].astype(int)
    X_test_lr['income_class'] = X_test_lr['income_class'].astype(int)

    X_train_lr['age_class'] = X_train_lr['age_class'].astype(int)
    X_val_lr['age_class'] = X_val_lr['age_class'].astype(int)
    X_test_lr['age_class'] = X_test_lr['age_class'].astype(int)

    X_val_lr = X_val_lr.reindex(columns=X_train_lr.columns, fill_value=0)
    X_test_lr = X_test_lr.reindex(columns=X_train_lr.columns, fill_value=0)

    numeric_features = ['perc_premium_paid_by_cash_credit',
                       'application_underwriting_score',
                       'no_of_premiums_paid',
                       'late_premium']

    scaler = StandardScaler()
    X_train_lr[numeric_features] = scaler.fit_transform(X_train_lr[numeric_features])
    X_val_lr[numeric_features] = scaler.transform(X_val_lr[numeric_features])
    X_test_lr[numeric_features] = scaler.transform(X_test_lr[numeric_features])

    return X_train_lr, X_val_lr, X_test_lr, scaler


def find_optimal_threshold(model, X_val, y_val, metric='f1_class0'):
    y_val_prob = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob, pos_label=1)

    y_val_prob_class0 = 1 - y_val_prob
    precisions_0, recalls_0, thresholds_0 = precision_recall_curve(
        1 - y_val, y_val_prob_class0, pos_label=1
    )

    if metric == 'f1_class0':
        f1_scores = 2 * precisions_0 * recalls_0 / (precisions_0 + recalls_0 + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = 1 - thresholds_0[optimal_idx]  # Invert back
    elif metric == 'recall_class0':
        # Find threshold that gives ~70% recall for class 0
        target_recall = 0.70
        idx = np.argmin(np.abs(recalls_0 - target_recall))
        optimal_threshold = 1 - thresholds_0[idx]
    else:  # balanced
        # Balance precision and recall for class 0
        diff = np.abs(precisions_0 - recalls_0)
        optimal_idx = np.argmin(diff)
        optimal_threshold = 1 - thresholds_0[optimal_idx]

    return optimal_threshold
