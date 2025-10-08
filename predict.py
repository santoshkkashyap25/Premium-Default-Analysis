import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.impute import SimpleImputer
import argparse
from src.preprocessing import create_advanced_features

# ================== CONSTANTS ==================
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_advanced.pkl"

# Binned feature parameters
INCOME_BINS = [0, 25000, 40000, 60000, 90000, 150000]
AGE_BINS = [0, 37.2, 53.4, 69.6, 85.8, 102, float('inf')]

# Mapping for categorical features
RES_AREA_MAP = {'Urban': 1, 'Rural': 0}
SOURCING_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

# Columns to drop
DROP_COLS = ['Income', 'Count_3-6_months_late', 'Count_6-12_months_late',
             'Count_more_than_12_months_late', 'age', 'age_in_days']

# Columns for missing value handling
LATE_COLS = ['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late']
SCORE_COL = 'application_underwriting_score'

# Intervention costs
INTERVENTION_COSTS = {
    'high': 50,
    'medium': 10,
    'low': 2,
    'none': 0
}

# ================== INTERVENTION TIERS ==================
def create_risk_tiers(probs, thresholds=(0.8, 0.6, 0.4)):
    """
    Assign risk tiers based on probability thresholds.
    """
    risk_tier, intervention_action, intervention_cost = [], [], []

    for p in probs:
        if p >= thresholds[0]:
            risk_tier.append('High')
            intervention_action.append('Targeted Intervention')
            intervention_cost.append(INTERVENTION_COSTS['high'])
        elif p >= thresholds[1]:
            risk_tier.append('Medium')
            intervention_action.append('Moderate Intervention')
            intervention_cost.append(INTERVENTION_COSTS['medium'])
        elif p >= thresholds[2]:
            risk_tier.append('Low')
            intervention_action.append('Basic Intervention')
            intervention_cost.append(INTERVENTION_COSTS['low'])
        else:
            risk_tier.append('None')
            intervention_action.append('No Action')
            intervention_cost.append(INTERVENTION_COSTS['none'])

    return pd.DataFrame({
        'risk_tier': risk_tier,
        'intervention_action': intervention_action,
        'intervention_cost': intervention_cost
    })

# ================== MAIN SCRIPT ==================
def main(input_path, output_path):
    # Load model
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Loaded stacking ensemble model.")

    # Load new customer data
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} new customer records.")

    # Apply advanced feature engineering
    df = create_advanced_features(df)

    # Fill missing values
    for col in LATE_COLS:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    if SCORE_COL in df.columns:
        df[SCORE_COL].fillna(df[SCORE_COL].median(), inplace=True)

    # Create binned features (nullable integers)
    df['income_class'] = pd.cut(
        df['Income'], bins=INCOME_BINS,
        labels=range(len(INCOME_BINS)-1),
        include_lowest=True
    ).astype("Int64")

    df['age_class'] = pd.cut(
        df['age'], bins=AGE_BINS,
        labels=range(len(AGE_BINS)-1),
        include_lowest=True
    ).astype("Int64")

    # Map categorical features
    df['residence_area_type'] = df['residence_area_type'].map(RES_AREA_MAP)
    df['sourcing_channel'] = df['sourcing_channel'].map(SOURCING_MAP)

    # Drop unused columns
    X_new = df.drop(DROP_COLS, axis=1, errors='ignore')
    if 'id' in X_new.columns:
        X_new = X_new.drop('id', axis=1)

    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    X_new_imputed = pd.DataFrame(imputer.fit_transform(X_new), columns=X_new.columns)

    # Predict probabilities
    probs = model.predict_proba(X_new_imputed)[:, 1]

    # Assign risk tiers
    tiers_df = create_risk_tiers(probs)

    # Combine with original data
    output_df = pd.DataFrame({
        'customer_id': df['id'] if 'id' in df.columns else df.index,
        'non_payer_probability': probs
    })
    output_df = pd.concat([output_df, tiers_df], axis=1)

    # Save output
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_df.to_csv(output_path, index=False)
    print(f"Monthly interventions saved at: {output_path}")

# ================== CLI ==================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict customer non-payer probabilities and risk tiers")
    parser.add_argument("--data", type=str, required=True, help="Path to input customer CSV file")
    parser.add_argument("--output", type=str, default=PROJECT_ROOT / "outputs" / "monthly_interventions.csv",
                        help="Path to save output CSV file")
    args = parser.parse_args()
    main(args.data, args.output)
