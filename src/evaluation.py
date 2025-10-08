import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_auc_score, f1_score, precision_recall_curve,
                            precision_recall_fscore_support)
import warnings
warnings.filterwarnings("ignore")
from src.preprocessing import find_optimal_threshold

def evaluate_model(model, X_val, y_val, X_test, y_test, model_name,
                  use_threshold_opt=False):
    # Get probabilities
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    # Determine threshold
    if use_threshold_opt:
        threshold = find_optimal_threshold(model, X_val, y_val, metric='f1_class0')
        y_val_pred = (y_val_prob >= threshold).astype(int)
        y_test_pred = (y_test_prob >= threshold).astype(int)
        model_name += f" (thresh={threshold:.3f})"
    else:
        threshold = 0.5
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

    # Calculate metrics
    val_metrics = precision_recall_fscore_support(y_val, y_val_pred, average=None, zero_division=0)
    test_metrics = precision_recall_fscore_support(y_test, y_test_pred, average=None, zero_division=0)

    results = {
        'Model': model_name,
        'Threshold': threshold,
        'Val_Accuracy': accuracy_score(y_val, y_val_pred),
        'Val_ROC_AUC': roc_auc_score(y_val, y_val_prob),
        'Val_F1_Macro': f1_score(y_val, y_val_pred, average='macro'),
        'Val_F1_Class0': val_metrics[2][0],
        'Val_Precision_Class0': val_metrics[0][0],
        'Val_Recall_Class0': val_metrics[1][0],
        'Test_Accuracy': accuracy_score(y_test, y_test_pred),
        'Test_ROC_AUC': roc_auc_score(y_test, y_test_prob),
        'Test_F1_Macro': f1_score(y_test, y_test_pred, average='macro'),
        'Test_F1_Class0': test_metrics[2][0],
        'Test_Precision_Class0': test_metrics[0][0],
        'Test_Recall_Class0': test_metrics[1][0]
    }

    print(f"\n{'='*75}")
    print(f"{model_name}")
    print(f"{'='*75}")
    print(f"Threshold: {threshold:.3f}")
    print(f"\nValidation:")
    print(f"  Overall    - Acc: {results['Val_Accuracy']:.4f}, ROC-AUC: {results['Val_ROC_AUC']:.4f}")
    print(f"  Class 0    - Precision: {results['Val_Precision_Class0']:.4f}, "
          f"Recall: {results['Val_Recall_Class0']:.4f}, F1: {results['Val_F1_Class0']:.4f}")

    print(f"\nTest:")
    print(f"  Overall    - Acc: {results['Test_Accuracy']:.4f}, ROC-AUC: {results['Test_ROC_AUC']:.4f}")
    print(f"  Class 0    - Precision: {results['Test_Precision_Class0']:.4f}, "
          f"Recall: {results['Test_Recall_Class0']:.4f}, F1: {results['Test_F1_Class0']:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nTest Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0        1")
    print(f"Actual  0    {cm[0,0]:5d}    {cm[0,1]:5d}  (Non-payers)")
    print(f"        1    {cm[1,0]:5d}    {cm[1,1]:5d}  (On-time payers)")

    total_class0 = cm[0,0] + cm[0,1]
    if total_class0 > 0:
        capture_rate = cm[0,0] / total_class0 * 100
        print(f"\nCaptured {cm[0,0]}/{total_class0} non-payers ({capture_rate:.1f}%)")

    return results


def create_intervention_tiers(probabilities, costs):
    # Convert to non-payer probabilities
    non_payer_probs = 1 - probabilities

    # Define tiers
    tiers = []
    actions = []
    tier_costs = []

    for prob in non_payer_probs:
        if prob > 0.7:
            tier = 'High Risk'
            action = 'Personal call + Special offer'
            cost = costs['high']
        elif prob > 0.4:
            tier = 'Medium Risk'
            action = 'Email + SMS reminder'
            cost = costs['medium']
        elif prob > 0.2:
            tier = 'Low-Medium Risk'
            action = 'SMS reminder'
            cost = costs['low']
        else:
            tier = 'Low Risk'
            action = 'Standard communication'
            cost = costs['none']

        tiers.append(tier)
        actions.append(action)
        tier_costs.append(cost)

    return pd.DataFrame({
        'non_payer_probability': non_payer_probs,
        'risk_tier': tiers,
        'intervention_action': actions,
        'intervention_cost': tier_costs
    })



def calculate_roi(y_true, tiers_df, policy_value=500):
    """
    Calculate ROI of intervention strategy
    """
    # Costs
    total_intervention_cost = tiers_df['intervention_cost'].sum()

    # Benefits
    actual_non_payers = np.sum(y_true == 0)

    # Identify which non-payers were caught (received intervention)
    non_payers_flagged = np.sum(
        (y_true == 0) & (tiers_df['risk_tier'] != 'Low Risk')
    )

    # Assume 60% of flagged non-payers are retained
    retention_rate = 0.60
    retained_policies = non_payers_flagged * retention_rate
    benefit = retained_policies * policy_value

    # Missed non-payers (cost)
    missed_non_payers = actual_non_payers - non_payers_flagged
    missed_cost = missed_non_payers * policy_value

    # Net benefit
    net_benefit = benefit - total_intervention_cost - missed_cost

    # ROI
    roi = (benefit - total_intervention_cost) / total_intervention_cost * 100

    results = {
        'total_intervention_cost': total_intervention_cost,
        'non_payers_flagged': non_payers_flagged,
        'retained_policies': retained_policies,
        'benefit': benefit,
        'missed_non_payers': missed_non_payers,
        'missed_cost': missed_cost,
        'net_benefit': net_benefit,
        'roi': roi
    }

    return results


def evaluate_with_business_metrics(model, X_test, y_test, model_name,
                                   threshold=0.5, intervention_costs=None):
    """Enhanced evaluation with business metrics"""

    if intervention_costs is None:
        intervention_costs = {
            'high': 50,
            'medium': 10,
            'low': 2,
            'none': 0
        }

    # Get predictions and probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Standard metrics
    test_metrics = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

    # Create intervention tiers
    tiers_df = create_intervention_tiers(y_prob, intervention_costs)

    # Calculate ROI
    roi_results = calculate_roi(y_test, tiers_df, policy_value=500)

    # Print results
    print(f"\n{'='*75}")
    print(f"{model_name}")
    print(f"{'='*75}")
    print(f"Threshold: {threshold:.3f}")

    print(f"\nClassification Metrics:")
    print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"   ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"   F1 (Class 0): {test_metrics[2][0]:.4f}")
    print(f"   Precision (Class 0): {test_metrics[0][0]:.4f}")
    print(f"   Recall (Class 0): {test_metrics[1][0]:.4f}")

    print(f"\nBusiness Metrics:")
    print(f"   Intervention Cost: ${roi_results['total_intervention_cost']:,.0f}")
    print(f"   Non-payers Flagged: {roi_results['non_payers_flagged']}")
    print(f"   Retained Policies: {roi_results['retained_policies']:.0f}")
    print(f"   Revenue Retained: ${roi_results['benefit']:,.0f}")
    print(f"   Missed Non-payers: {roi_results['missed_non_payers']}")
    print(f"   Missed Revenue: ${roi_results['missed_cost']:,.0f}")
    print(f"   Net Benefit: ${roi_results['net_benefit']:,.0f}")
    print(f"   ROI: {roi_results['roi']:.1f}%")

    # Tier distribution
    print(f"\nRisk Tier Distribution:")
    tier_counts = tiers_df['risk_tier'].value_counts()
    for tier, count in tier_counts.items():
        pct = count / len(tiers_df) * 100
        print(f"   {tier}: {count} ({pct:.1f}%)")

    results = {
        'Model': model_name,
        'Threshold': threshold,
        'Test_Accuracy': accuracy_score(y_test, y_pred),
        'Test_ROC_AUC': roc_auc_score(y_test, y_prob),
        'Test_F1_Class0': test_metrics[2][0],
        'Test_Precision_Class0': test_metrics[0][0],
        'Test_Recall_Class0': test_metrics[1][0],
        'Intervention_Cost': roi_results['total_intervention_cost'],
        'Net_Benefit': roi_results['net_benefit'],
        'ROI': roi_results['roi']
    }

    return results, tiers_df

