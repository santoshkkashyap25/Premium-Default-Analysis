# Insurance Premium Default Risk Profiler

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-Calibrated_XGBoost-EB5424?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Predicting insurance policy payment default under severe class imbalance through probability calibration and asymmetric cost optimization.**

---

## Problem Statement

When an insurance policy lapses due to non-payment, the insurer incurs an **asymmetric financial loss**:
- **Severe Capital Destruction**: The insurer forfeits customer acquisition cost, future recurring premium revenue, and underwriting margins (~$500+ lifetime policy value lost per lapse).
- **The Operational Dilemma**: Blanket outreach to the entire portfolio is economically wasteful and causes customer fatigue for on-time payers. Conversely, passive collection strategies fail to rescue savable accounts before their grace period expires.
- **The Objective**: Accurately predict policyholders with high probability of default in advance, routing each account into an economically rational, cost-tiered intervention workflow that maximizes net revenue preserved while minimizing operational expenditure.

---

## Navigating Severe Class Imbalance: Why Accuracy Fails

### 1. The Accuracy Illusion
In our 79,853 policyholder benchmark dataset:
- **93.7%** of policyholders pay on time (Class 1).
- Only **6.3%** default on their premium (Class 0).

A naive baseline classifier that predicts *every customer will pay on time* achieves **93.7% accuracy**. However, such a system detects zero defaulters, prevents zero lapses, and produces **$0 in preserved financial value**.

### 2. Ground Truth Evaluation Strategy
Because standard ROC-AUC and accuracy are artificially inflated by the majority class, this system evaluates performance strictly using **Precision-Recall AUC (PR-AUC)**, **Defaulter Recall**, and **Cost-Weighted Net Benefit**:

- **Baseline Random Guess Precision**: 6.3%
- **Model Defaulter Precision**: **35.9%** (**5.7x precision lift** over random baseline)
- **Defaulter Recall**: **38.5%** (identifies nearly 40% of all lapses in advance)
- **Precision-Recall AUC**: **0.2954** (vs. 0.063 baseline)
- **ROC-AUC**: **0.8426**

**Operational Translation**: Out of every 100 accounts flagged by our system, approximately **36 are verified defaulters**. This enables retention teams to allocate high-touch phone outreach and physical mail sequences strictly to accounts where default risk is real and intervention ROI is provably positive.

---

## The Probability Calibration Breakthrough

Tree-based ensemble models (XGBoost, Random Forest, LightGBM) produce reliable ranking scores, but their raw output probabilities are heavily distorted near the distribution tails.

In cost-sensitive retention, uncalibrated probabilities cause **severe capital overspending**: the model appears artificially overconfident on marginal false alarms, triggering expensive phone interventions for customers who would have paid anyway.

### Empirical Candidate Benchmark (Held-Out Test Set)

By applying **Isotonic Regression calibration** and optimizing decision thresholding on the validation Precision-Recall curve (**optimal threshold: 0.768 on on-time probability**), we achieved a massive economic turnaround:

| Model Candidate | Decision Threshold | Test Recall (Defaulters) | Test Precision | Intervention Budget | Net Benefit | ROI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost (Calibrated)** *(Champion)* | **0.768** | **38.5%** | **35.9%** | **$5,790** | **$186,710** | **3,224.7%** |
| XGBoost (Optuna Tuned) | 0.552 | 41.3% | 34.7% | $32,210 | $174,290 | 541.1% |
| Random Forest | 0.328 | 42.5% | 31.9% | $93,918 | $118,582 | 126.3% |
| Logistic Regression + SMOTE | 0.344 | 37.8% | 33.4% | $72,184 | $116,816 | 161.8% |
| Balanced Random Forest | 0.280 | 45.3% | 32.6% | $120,464 | $106,036 | 88.0% |
| Stacking Ensemble | 0.177 | 39.5% | 34.1% | $140,020 | $57,480 | 41.1% |
| XGBoost (Cost-Sensitive) | 0.073 | 35.4% | 36.0% | $280,938 | -$103,938 | -37.0% |

> **Empirical Insight**: Brute-force cost-sensitive weighting during training failed (-$103k net loss) because it flooded collection teams with false positives. Conversely, **well-calibrated probabilities paired with optimal post-hoc thresholding** compressed outreach expenditure from $140k+ to just $5,790 while delivering a verified **$186,710 net benefit (3,224.7% ROI)**.

---

## Operational Risk-Tier Matrix

Calibrated default probabilities feed directly into four operational intervention tiers:

```
Calibrated Default Probability
 │
 ├── > 70%  ────────►  High Risk ($50 outreach)
 │                     Outbound concierge call + payment restructuring
 │
 ├── 40% – 70%  ────►  Medium Risk ($10 outreach)
 │                     Direct mail statement + priority SMS alert sequence
 │
 ├── 20% – 40%  ────►  Low-Medium Risk ($2 outreach)
 │                     Automated 2-touch SMS payment link
 │
 └── ≤ 20%  ────────►  Low Risk ($0 outreach)
                       Standard automated digital billing invoice
```

### Financial Payoff Equation
$$\text{Net Benefit} = (\text{True Positives} \times \text{Lapse Value} \times \text{Recovery Rate}) - \text{Total Intervention Spend}$$
- **Lapse Value Preserved**: $500 per successfully retained policy
- **Recovery Rate**: 60% recovery assumption for contacted defaulters
- **Cost of False Negatives**: $500 lifetime value lost if a defaulter lapses undetected

---

## Behavioral Drivers & Feature Insights

Empirical feature importance reveals that dynamic payment behavior heavily outweighs static demographic markers:

1. **Delinquency Recency (`Count_3-6_months_late`)**: The single most dominant default indicator. Accounts with even one 3–6 month late payment are over 4x more likely to lapse.
2. **Payment Channel (`perc_premium_paid_by_cash_credit`)**: Policyholders paying via cash or credit card default at more than double the rate of those on automated bank ACH drafts.
3. **Payment Reliability Ratio**: Non-linear feature $\frac{\text{Premiums Paid}}{\text{Premiums Paid} + \text{Total Late Payments}}$ accurately captures recovery trajectory in tenured policyholders.
4. **Underwriting Score Missingness**: Unrecorded underwriting scores (`underwriting_score_missing = 1`) reflect distinct risk characteristics captured natively by the pipeline.


---

## Robust Handling of Incomplete Records

In production environments, policy records often lack complete data. The inference pipeline handles missing attributes silently using learned training statistics without data leakage:

- **Underwriting Score Missing**: Flags `underwriting_score_missing = 1` and imputes score with the training median (`99.21`).
- **Payment Method Unrecorded**: Imputes `perc_premium_paid_by_cash_credit` with the population median (`0.167`).
- **Sourcing Channel Unknown**: Maps unrecorded acquisition channels to Channel A (`0`, the mode comprising 54% of training records).

---

## Standalone Research Notebooks

The research and benchmarking pipeline is fully documented in two standalone, reproducible notebooks in `notebooks/`:
- `Default_Prediction_EDA.ipynb`: Exploratory data analysis, class imbalance diagnostics, and feature interaction studies.
- `Default_Prediction_Model.ipynb`: Candidate model evaluations, Bayesian hyperparameter optimization (Optuna), isotonic probability calibration, and financial ROI simulation.

---

## Technical Stack

- **Modeling**: XGBoost, Scikit-Learn (IsotonicRegression, CalibratedClassifierCV), LightGBM, Imbalanced-Learn
- **Serving & Web Application**: Streamlit, FastAPI, Pydantic, Uvicorn
- **Data Engineering**: Pandas, NumPy
- **Environment**: Python 3.11+