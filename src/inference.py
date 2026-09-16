"""Modular Inference Pipeline for Insurance Premium Default Risk Profiling.

Loads the calibrated champion model artifact and metadata to score policyholders,
route them into tiered operational interventions, and calculate cost efficiency.
"""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Import xgboost and lightgbm to ensure pickle unpickling succeeds
try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from src.preprocessing import preprocess_for_inference, load_metadata

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "champion_model.pkl"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "experiment_metadata.json"

INTERVENTION_ACTIONS = {
    'High Risk': 'Outbound concierge phone call + premium restructuring offer',
    'Medium Risk': 'Direct mail notice + priority SMS alert',
    'Low-Medium Risk': 'Automated SMS reminder',
    'Low Risk': 'Standard automated billing notice (no extra outreach)'
}

INTERVENTION_COSTS = {
    'High Risk': 50,
    'Medium Risk': 10,
    'Low-Medium Risk': 2,
    'Low Risk': 0
}


class InferencePipeline:
    """Production inference pipeline for insurance premium default prediction."""

    def __init__(self, model_path=None, metadata_path=None):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not self.model_path.exists():
            # Fallback to best_model.pkl if champion_model.pkl is missing
            fallback = PROJECT_ROOT / "models" / "best_model.pkl"
            if fallback.exists():
                self.model_path = fallback

        self.metadata_path = Path(metadata_path) if metadata_path else DEFAULT_METADATA_PATH
        self.metadata = self._load_metadata()
        self.model = self._load_model()
        self.optimal_threshold = float(self.metadata.get('optimal_threshold', 0.768))

    def _load_metadata(self):
        return load_metadata(self.metadata_path)

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Trained champion model not found at {self.model_path}. "
                "Please verify models/champion_model.pkl or models/best_model.pkl."
            )
        with open(self.model_path, "rb") as f:
            return pickle.load(f)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply leak-free feature engineering and align to model feature space."""
        return preprocess_for_inference(df, metadata=self.metadata)

    def assign_risk_tiers(self, default_probs: np.ndarray):
        """Map default probabilities to operational tiers, recommended actions, and costs."""
        tiers = np.where(
            default_probs > 0.70, 'High Risk',
            np.where(
                default_probs > 0.40, 'Medium Risk',
                np.where(default_probs > 0.20, 'Low-Medium Risk', 'Low Risk')
            )
        )
        actions = [INTERVENTION_ACTIONS[t] for t in tiers]
        costs = [INTERVENTION_COSTS[t] for t in tiers]

        return tiers, actions, costs

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run end-to-end inference on batch or single-row DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing policyholder features.

        Returns
        -------
        pd.DataFrame
            Enriched results DataFrame containing probabilities, status, risk tiers,
            recommended actions, and projected intervention costs.
        """
        X = self.transform(df)

        # Champion model returns [P(default), P(on-time)]
        # Class 1 is on-time; Class 0 is default
        on_time_probs = self.model.predict_proba(X)[:, 1]
        default_probs = 1.0 - on_time_probs

        # Decision thresholding optimized on Validation PR curve
        is_on_time = (on_time_probs >= self.optimal_threshold).astype(int)
        predicted_status = np.where(is_on_time == 1, 'On-Time', 'Default Risk')

        tiers, actions, costs = self.assign_risk_tiers(default_probs)

        # Retain customer ID if provided, otherwise generate clean string IDs
        if 'id' in df.columns:
            customer_ids = [
                str(val) if pd.notna(val) and val is not None else f"POL-{i+1:05d}"
                for i, val in enumerate(df['id'])
            ]
        else:
            customer_ids = [f"POL-{i+1:05d}" for i in range(len(df))]

        results = pd.DataFrame({
            'customer_id': customer_ids,
            'default_probability': np.round(default_probs, 4),
            'on_time_probability': np.round(on_time_probs, 4),
            'predicted_status': predicted_status,
            'risk_tier': tiers,
            'recommended_action': actions,
            'intervention_cost': costs
        })

        return results

    def predict_one(self, record: dict) -> dict:
        """Convenience method for scoring a single policyholder record dictionary."""
        df_single = pd.DataFrame([record])
        results_df = self.predict(df_single)
        return results_df.iloc[0].to_dict()
