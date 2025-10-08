from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_advanced.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "preprocessing_scaler.pkl"

# Load model and scaler
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    logger.info(f"   Expected features: {len(model.feature_names_in_)}")
    logger.info(f"   Features: {model.feature_names_in_.tolist()}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded successfully")
except:
    logger.warning("Scaler not found, will proceed without scaling")
    scaler = None

# Constants
INTERVENTION_COSTS = {'high': 50, 'medium': 10, 'low': 2, 'none': 0}

# ================== PREPROCESSING FUNCTIONS ==================

def create_advanced_features(df):
    """Create advanced features (same as training)"""
    df = df.copy()
    
    # Basic features
    df['late_premium'] = (
        df.get('Count_3-6_months_late', 0) +
        df.get('Count_6-12_months_late', 0) +
        df.get('Count_more_than_12_months_late', 0)
    )
    
    df['age'] = df['age_in_days'] // 365
    
    if 'application_underwriting_score' in df.columns:
        df['application_underwriting_score'] = df['application_underwriting_score'] / 100
    
    # Advanced interaction features
    df['high_cash_late_combo'] = (
        (df.get('perc_premium_paid_by_cash_credit', 0) > 0.5) & 
        (df['late_premium'] > 2)
    ).astype(int)
    
    df['financial_stress'] = (
        (df.get('Income', 0) < df.get('Income', 0) * 0.25) & 
        (df['late_premium'] > 1)
    ).astype(int)
    
    df['payment_reliability'] = df.get('no_of_premiums_paid', 0) / (
        df.get('no_of_premiums_paid', 0) + df['late_premium'] + 1e-5
    )
    
    if 'application_underwriting_score' in df.columns:
        df['composite_risk'] = (
            (1 - df['application_underwriting_score']) * 0.4 +
            (df['late_premium'] / (df['late_premium'].max() + 1)) * 0.6
        )
    else:
        df['composite_risk'] = 0
    
    df['age_income_interaction'] = df['age'] * np.log1p(df.get('Income', 0))
    
    df['recent_late_weighted'] = (
        df.get('Count_3-6_months_late', 0) * 3 +
        df.get('Count_6-12_months_late', 0) * 2 +
        df.get('Count_more_than_12_months_late', 0) * 1
    )
    
    df['income_payment_ratio'] = df.get('Income', 0) / (
        df.get('perc_premium_paid_by_cash_credit', 0) * df.get('Income', 0) + 1
    )
    
    df['zero_late_payments'] = (df['late_premium'] == 0).astype(int)
    df['chronic_late_payer'] = (df['late_premium'] >= 5).astype(int)
    df['new_customer'] = (df.get('no_of_premiums_paid', 0) <= 3).astype(int)
    
    return df

def preprocess_for_prediction(df, model):
    """
    Preprocess data to match model's expected features
    """
    # Step 1: Feature engineering
    df = create_advanced_features(df)
    
    # Step 2: Handle missing values
    late_cols = ['Count_3-6_months_late', 'Count_6-12_months_late', 
                 'Count_more_than_12_months_late']
    for col in late_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    if 'application_underwriting_score' in df.columns:
        median_score = df['application_underwriting_score'].median()
        if pd.isna(median_score):
            median_score = 0.99  # Default
        df['application_underwriting_score'].fillna(median_score, inplace=True)
    
    # Step 3: Create binned features
    # Income bins (same as training)
    income_bins = [0, 71200, 134000, 197000, 260000, 323000, float('inf')]
    income_labels = list(range(len(income_bins) - 1))
    
    if 'Income' in df.columns:
        df['income_class'] = pd.cut(
            df['Income'], 
            bins=income_bins,
            labels=income_labels, 
            include_lowest=True
        )
        df['income_class'] = df['income_class'].astype(float).fillna(0).astype(int)
    
    # Age bins (fixed)
    age_bins = [0, 37.2, 53.4, 69.6, 85.8, 102, float('inf')]
    age_labels = list(range(len(age_bins) - 1))
    
    if 'age' in df.columns:
        df['age_class'] = pd.cut(
            df['age'],
            bins=age_bins,
            labels=age_labels,
            include_lowest=True
        )
        df['age_class'] = df['age_class'].astype(float).fillna(0).astype(int)
    
    # Step 4: Encode categorical variables
    res_area_map = {'Urban': 1, 'Rural': 0}
    sourcing_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    if 'residence_area_type' in df.columns:
        df['residence_area_type'] = df['residence_area_type'].map(res_area_map)
        df['residence_area_type'].fillna(0, inplace=True)
    
    if 'sourcing_channel' in df.columns:
        df['sourcing_channel'] = df['sourcing_channel'].map(sourcing_map)
        df['sourcing_channel'].fillna(0, inplace=True)
    
    # Step 5: Drop columns not needed for prediction
    drop_cols = [
        'Income', 'Count_3-6_months_late', 'Count_6-12_months_late',
        'Count_more_than_12_months_late', 'age', 'age_in_days', 'id'
    ]
    
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(col, axis=1)
    
    # Step 6: Align with model's expected features
    expected_features = model.feature_names_in_
    
    # Add missing columns with 0
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Keep only expected features in correct order
    df = df[expected_features]
    
    # Step 7: Fill any remaining NaN
    df = df.fillna(0)
    
    return df

def create_risk_tiers(prob, thresholds=(0.7, 0.4, 0.2)):
    """
    Create risk tier based on non-payer probability
    Note: prob is probability of CLASS 1 (on-time payment)
    So non-payer prob = 1 - prob
    """
    non_payer_prob = 1 - prob
    
    if non_payer_prob >= thresholds[0]:
        return 'High Risk', 'Personal call + Special offer', INTERVENTION_COSTS['high']
    elif non_payer_prob >= thresholds[1]:
        return 'Medium Risk', 'Email + SMS reminder', INTERVENTION_COSTS['medium']
    elif non_payer_prob >= thresholds[2]:
        return 'Low-Medium Risk', 'SMS reminder', INTERVENTION_COSTS['low']
    else:
        return 'Low Risk', 'Standard communication', INTERVENTION_COSTS['none']

# ================== API ENDPOINTS ==================

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict premium payment default risk
    
    Expected JSON format:
    {
        "id": "CUST123",
        "perc_premium_paid_by_cash_credit": 0.5,
        "age_in_days": 18250,
        "Income": 50000,
        "Count_3-6_months_late": 0,
        "Count_6-12_months_late": 1,
        "Count_more_than_12_months_late": 0,
        "application_underwriting_score": 99.5,
        "no_of_premiums_paid": 10,
        "sourcing_channel": "A",
        "residence_area_type": "Urban"
    }
    """
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        logger.info(f"Received prediction request for customer: {data.get('id', 'unknown')}")
        
        df = pd.DataFrame([data])
        
        # Preprocess
        logger.info("Preprocessing data...")
        X_processed = preprocess_for_prediction(df, model)
        
        logger.info(f"Processed features shape: {X_processed.shape}")
        logger.info(f"Expected features: {len(model.feature_names_in_)}")
        
        # Predict probability
        prob = model.predict_proba(X_processed)[:, 1][0]
        
        logger.info(f"Prediction probability: {prob:.4f}")
        
        # Calculate non-payer probability (CLASS 0)
        non_payer_prob = 1 - prob
        
        # Determine risk tier
        tier, action, cost = create_risk_tiers(prob)
        
        # Build response
        response = {
            "customer_id": data.get("id", "unknown"),
            "on_time_probability": round(float(prob), 4),
            "non_payer_probability": round(float(non_payer_prob), 4),
            "risk_tier": tier,
            "recommended_action": action,
            "intervention_cost": cost,
            "model_confidence": "high" if abs(prob - 0.5) > 0.3 else "medium"
        }
        
        logger.info(f"Prediction successful: {tier}")
        
        return jsonify(response)
    
    except KeyError as e:
        error_msg = f"Missing required field: {str(e)}"
        logger.error(error_msg)
        return jsonify({
            "error": error_msg,
            "required_fields": [
                "perc_premium_paid_by_cash_credit", "age_in_days", "Income",
                "Count_3-6_months_late", "Count_6-12_months_late", 
                "Count_more_than_12_months_late", "application_underwriting_score",
                "no_of_premiums_paid", "sourcing_channel", "residence_area_type"
            ]
        }), 400
    
    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return jsonify({
            "error": error_msg,
            "traceback": traceback.format_exc()
        }), 500


# ================== ERROR HANDLERS ==================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

# ================== MAIN ==================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)