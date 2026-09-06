import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Ensure ML libraries are imported for unpickling models
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.preprocessing import create_advanced_features


# Default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "preprocessing_scaler.pkl"

# Constants for preprocessing
INCOME_BINS = [-np.inf, 71200, 134000, 197000, 260000, 323000, np.inf]
AGE_BINS = [-np.inf, 37.2, 53.4, 69.6, 85.8, 102, np.inf]

RES_AREA_MAP = {'Urban': 1, 'Rural': 0}
SOURCING_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
DROP_COLS = ['Income', 'Count_3-6_months_late', 'Count_6-12_months_late',
             'Count_more_than_12_months_late', 'age', 'age_in_days']

INTERVENTION_COSTS = {'high': 50, 'medium': 10, 'low': 2, 'none': 0}


class InferencePipeline:
    """Modular Inference Pipeline for scoring insurance default risk from pre-trained model artifacts."""

    def __init__(self, model_path=None, scaler_path=None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            # Fallback to legacy model path
            fallback = PROJECT_ROOT / "models" / "best_model_advanced.pkl"
            if fallback.exists():
                self.model_path = fallback

        self.scaler_path = Path(scaler_path) if scaler_path else DEFAULT_SCALER_PATH

        self.model = self._load_model()
        self.scaler = self._load_scaler()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Trained model file not found at {self.model_path}. Please run train.py first.")
        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def _load_scaler(self):
        if self.scaler_path.exists():
            with open(self.scaler_path, 'rb') as f:
                return pickle.load(f)
        return None

    def transform(self, df):
        """Prepare raw input DataFrame for inference."""
        df_processed = df.copy()

        # Handle missing values in raw features
        late_cols = ['Count_3-6_months_late', 'Count_6-12_months_late', 'Count_more_than_12_months_late']
        for col in late_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].fillna(0)

        if 'application_underwriting_score' in df_processed.columns:
            df_processed['application_underwriting_score'] = df_processed['application_underwriting_score'].fillna(99.0)

        # Apply advanced feature engineering
        df_processed = create_advanced_features(df_processed)

        # Binned features
        if 'Income' in df_processed.columns:
            income_labels = list(range(len(INCOME_BINS)-1))
            df_processed['income_class'] = pd.cut(
                df_processed['Income'], bins=INCOME_BINS, labels=income_labels, include_lowest=True
            ).fillna(0).astype(int)

        if 'age' in df_processed.columns:
            age_labels = list(range(len(AGE_BINS)-1))
            df_processed['age_class'] = pd.cut(
                df_processed['age'], bins=AGE_BINS, labels=age_labels, include_lowest=True
            ).fillna(0).astype(int)

        # Categorical mappings
        if 'residence_area_type' in df_processed.columns:
            df_processed['residence_area_type'] = df_processed['residence_area_type'].map(RES_AREA_MAP).fillna(0)
        if 'sourcing_channel' in df_processed.columns:
            df_processed['sourcing_channel'] = df_processed['sourcing_channel'].map(SOURCING_MAP).fillna(0)

        # Drop unused metadata/raw columns
        drop_list = [c for c in DROP_COLS + ['id', 'target'] if c in df_processed.columns]
        X = df_processed.drop(columns=drop_list, errors='ignore')

        # Fill any remaining missing values
        X = X.fillna(0)

        # Match exact feature names expected by trained model
        if hasattr(self.model, 'feature_names_in_'):
            X = X.reindex(columns=self.model.feature_names_in_, fill_value=0)

        return X

    def assign_risk_tiers(self, probs):
        """Assign risk tiers and recommended intervention actions based on probabilities."""
        tiers, actions, costs = [], [], []

        for p in probs:
            if p >= 0.7:
                tiers.append('High Risk')
                actions.append('Personal call + Special offer')
                costs.append(INTERVENTION_COSTS['high'])
            elif p >= 0.4:
                tiers.append('Medium Risk')
                actions.append('Email + SMS reminder')
                costs.append(INTERVENTION_COSTS['medium'])
            elif p >= 0.2:
                tiers.append('Low-Medium Risk')
                actions.append('SMS reminder')
                costs.append(INTERVENTION_COSTS['low'])
            else:
                tiers.append('Low Risk')
                actions.append('Standard communication')
                costs.append(INTERVENTION_COSTS['none'])

        return pd.DataFrame({
            'risk_tier': tiers,
            'intervention_action': actions,
            'intervention_cost': costs
        })

    def predict(self, df):
        """Run complete inference workflow on new input data."""
        X_proc = self.transform(df)
        probs = self.model.predict_proba(X_proc)[:, 1]
        non_payer_probs = 1.0 - probs

        tiers_df = self.assign_risk_tiers(non_payer_probs)

        cust_ids = df['id'] if 'id' in df.columns else df.index

        results_df = pd.DataFrame({
            'customer_id': cust_ids,
            'on_time_probability': np.round(probs, 4),
            'non_payer_probability': np.round(non_payer_probs, 4)
        })

        results_df = pd.concat([results_df, tiers_df], axis=1)
        return results_df
