"""Preprocessing and feature engineering module for insurance default prediction.

This module provides metadata-driven feature transformation for inference,
ensuring strict parity with the verified notebook experimentation pipeline.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# Default metadata location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "experiment_metadata.json"

RES_AREA_MAP = {'Urban': 1, 'Rural': 0}
SOURCING_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

AGE_BINS = [-np.inf, 35, 50, 65, 80, np.inf]
AGE_LABELS = list(range(len(AGE_BINS) - 1))

INCOME_BINS = [-np.inf, 100000, 140000, 190000, 260000, np.inf]
INCOME_LABELS = list(range(len(INCOME_BINS) - 1))

DROP_COLS = [
    'age_in_days',
    'Count_3-6_months_late',
    'Count_6-12_months_late',
    'Count_more_than_12_months_late',
    'id',
    'target',
]


def load_metadata(metadata_path=None):
    """Load learned experiment statistics, feature columns, and decision thresholds."""
    path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment metadata not found at {path}. "
            "Please ensure models/experiment_metadata.json exists."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_for_inference(df: pd.DataFrame, metadata: dict = None) -> pd.DataFrame:
    """Transform raw customer input data into model-ready tree features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer record(s) containing demographic and payment history fields.
    metadata : dict, optional
        Metadata dictionary containing training statistics and feature definitions.
        If None, loaded from models/experiment_metadata.json.

    Returns
    -------
    pd.DataFrame
        Engineered feature matrix matching the champion model's feature space.
    """
    if metadata is None:
        metadata = load_metadata()

    stats = metadata.get('training_statistics', {})
    underwriting_median = stats.get('underwriting_median', 99.21)
    income_q25 = stats.get('income_q25', 106730.0)
    income_median = stats.get('income_median', 165140.0)
    late_max = stats.get('late_premium_max', 19.0)
    inc_clip_lower = stats.get('income_clip_lower', 70070.0)
    inc_clip_upper = stats.get('income_clip_upper', 450040.0)

    feature_names = metadata.get('feature_names_tree', None)

    d = df.copy()

    # 1. Zero-delinquency imputation for missing late count fields
    late_cols = [
        'Count_3-6_months_late',
        'Count_6-12_months_late',
        'Count_more_than_12_months_late',
    ]
    for col in late_cols:
        if col in d.columns:
            d[col] = d[col].fillna(0).astype(float)
        else:
            d[col] = 0.0

    # 2. Underwriting score imputation + missingness indicator
    underwriting_col = 'application_underwriting_score'
    if underwriting_col in d.columns:
        d['underwriting_score_missing'] = d[underwriting_col].isna().astype(int)
        d[underwriting_col] = pd.to_numeric(d[underwriting_col], errors='coerce').fillna(underwriting_median).astype(float)
    else:
        d['underwriting_score_missing'] = 1
        d[underwriting_col] = underwriting_median

    # 3. Row-level feature engineering
    d['late_premium'] = (
        d['Count_3-6_months_late'] +
        d['Count_6-12_months_late'] +
        d['Count_more_than_12_months_late']
    )

    # 3. Robust Age Calculation
    if 'age' in d.columns and d['age'].notna().any():
        d['age'] = pd.to_numeric(d['age'], errors='coerce').fillna(45.0).astype(float)
    elif 'age_in_days' in d.columns and d['age_in_days'].notna().any():
        d['age'] = (pd.to_numeric(d['age_in_days'], errors='coerce') // 365).fillna(45.0).astype(float)
    else:
        d['age'] = 45.0

    d['underwriting_score_norm'] = (d[underwriting_col] / 100.0).astype(float)

    d['recent_late_weighted'] = (
        d['Count_3-6_months_late'] * 3.0 +
        d['Count_6-12_months_late'] * 2.0 +
        d['Count_more_than_12_months_late'] * 1.0
    ).astype(float)

    if 'no_of_premiums_paid' not in d.columns:
        d['no_of_premiums_paid'] = 12.0
    d['no_of_premiums_paid'] = pd.to_numeric(d['no_of_premiums_paid'], errors='coerce').fillna(12.0).astype(float)
    d['payment_reliability'] = (d['no_of_premiums_paid'] / (d['no_of_premiums_paid'] + d['late_premium'] + 1e-5)).astype(float)

    if 'perc_premium_paid_by_cash_credit' not in d.columns:
        d['perc_premium_paid_by_cash_credit'] = 0.0
    d['perc_premium_paid_by_cash_credit'] = pd.to_numeric(d['perc_premium_paid_by_cash_credit'], errors='coerce').fillna(0.0).astype(float)
    d['high_cash_late_combo'] = ((d['perc_premium_paid_by_cash_credit'] > 0.5) & (d['late_premium'] > 2)).astype(int)

    d['zero_late_payments'] = (d['late_premium'] == 0).astype(int)
    d['chronic_late_payer'] = (d['late_premium'] >= 5).astype(int)
    d['new_customer'] = (d['no_of_premiums_paid'] <= 3).astype(int)

    # 4. Outlier clipping on Income using training bounds
    if 'Income' not in d.columns:
        d['Income'] = income_median
    d['Income'] = pd.to_numeric(d['Income'], errors='coerce').fillna(income_median).clip(inc_clip_lower, inc_clip_upper).astype(float)

    d['age_income_interaction'] = (d['age'] * np.log1p(d['Income'])).astype(float)

    # 5. Population-statistical features derived strictly from training statistics
    d['financial_stress'] = ((d['Income'] < income_q25) & (d['late_premium'] > 1)).astype(int)
    d['income_payment_ratio'] = (d['Income'] / (d['perc_premium_paid_by_cash_credit'] * income_median + 1.0)).astype(float)
    d['composite_risk'] = (
        (1.0 - d['underwriting_score_norm']) * 0.4 +
        (d['late_premium'] / (late_max + 1.0)) * 0.6
    ).astype(float)

    # 6. Binned features
    d['income_class'] = pd.cut(
        d['Income'], bins=INCOME_BINS, labels=INCOME_LABELS, include_lowest=True
    ).fillna(0).astype(int)

    d['age_class'] = pd.cut(
        d['age'], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True
    ).fillna(0).astype(int)

    # 7. Categorical encoding for Tree-based models
    if 'residence_area_type' in d.columns:
        d['residence_area_type'] = d['residence_area_type'].map(RES_AREA_MAP).fillna(0).astype(int)
    else:
        d['residence_area_type'] = 1

    if 'sourcing_channel' in d.columns:
        d['sourcing_channel'] = d['sourcing_channel'].map(SOURCING_MAP).fillna(0).astype(int)
    else:
        d['sourcing_channel'] = 0

    # 8. Drop metadata and raw columns
    X = d.drop(columns=DROP_COLS, errors='ignore')

    # 9. Align columns strictly to champion model feature names and cast to float
    if feature_names is not None:
        X = X.reindex(columns=feature_names, fill_value=0)

    return X.astype(float)
