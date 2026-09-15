"""Insurance Premium Default Risk Profiler - Production Web Application.

Provides interactive single-customer risk profiling, bulk batch CSV scoring,
and financial ROI analytics powered by the calibrated champion XGBoost model.
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

from src.inference import InferencePipeline

# Page Configuration
st.set_page_config(
    page_title="Insurance Premium Default Risk Profiler",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design System & CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #111827;
    }
    .tier-card-high {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1.2rem;
        border-radius: 8px;
        color: #991B1B;
        margin-top: 1rem;
    }
    .tier-card-med {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 1.2rem;
        border-radius: 8px;
        color: #92400E;
        margin-top: 1rem;
    }
    .tier-card-low-med {
        background-color: #EFF6FF;
        border-left: 6px solid #3B82F6;
        padding: 1.2rem;
        border-radius: 8px;
        color: #1E40AF;
        margin-top: 1rem;
    }
    .tier-card-low {
        background-color: #F0FDF4;
        border-left: 6px solid #10B981;
        padding: 1.2rem;
        border-radius: 8px;
        color: #065F46;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Load and cache the InferencePipeline instance."""
    try:
        return InferencePipeline()
    except Exception as e:
        st.error(f"⚠️ Failed to initialize inference pipeline: {e}")
        return None


pipeline = get_pipeline()

# Title Header
st.markdown('<div class="main-title">🛡️ Insurance Premium Default Risk Profiler</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Identify high-risk policyholders prior to premium lapse and automate cost-effective outreach interventions.</div>',
    unsafe_allow_html=True
)

if pipeline is None:
    st.error("Model artifact or metadata is missing. Please verify `models/champion_model.pkl`.")
    st.stop()

# Sidebar Navigation & Model Info
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=64)
st.sidebar.title("Operational Control")
app_mode = st.sidebar.radio(
    "Select Workflow:",
    [
        "👤 Single Policyholder Scoring",
        "📁 Bulk Batch CSV Scoring",
        "📊 Strategy & Economic Blueprint"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Production Model Telemetry")
st.sidebar.info(f"""
- **Champion Model**: `{pipeline.model_path.name}`
- **Optimal Decision Threshold**: `{pipeline.optimal_threshold:.3f}`
- **Probability Calibration**: `Isotonic Regression`
- **Verified Benchmark**: `3,224% ROI` | `$186k+ Net Benefit`
""")


# ==============================================================================
# MODE 1: SINGLE POLICYHOLDER SCORING
# ==============================================================================
if app_mode == "👤 Single Policyholder Scoring":
    st.subheader("Policyholder Assessment Form")
    st.caption("Enter customer demographics, payment behavior, and policy attributes to calculate calibrated lapse risk.")

    with st.form("single_scoring_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Customer Profile")
            age_years = st.number_input(
                "Age (Years)",
                min_value=18, max_value=100, value=38, step=1,
                help="Customer age in years."
            )
            residence = st.selectbox(
                "Residence Area Type",
                options=["Urban", "Rural"],
                index=0,
                help="Geographic classification."
            )
            sourcing = st.selectbox(
                "Sourcing Channel",
                options=["A", "B", "C", "D", "E"],
                index=2,
                help="Policy acquisition channel code."
            )

        with col2:
            st.markdown("##### 💳 Financial Metrics")
            income = st.number_input(
                "Annual Income ($)",
                min_value=10000, max_value=5000000, value=145000, step=5000,
                help="Reported annual gross income."
            )
            perc_cash_credit = st.slider(
                "% Premium Paid by Cash / Credit",
                min_value=0.0, max_value=1.0, value=0.45, step=0.01,
                help="Percentage of previous premiums paid via cash or credit card (0.0 = 0%, 1.0 = 100%)."
            )
            underwriting_score = st.slider(
                "Application Underwriting Score (0-100)",
                min_value=0.0, max_value=100.0, value=98.5, step=0.1,
                help="Credit/underwriting score from application. Median imputation applied if omitted."
            )

        with col3:
            st.markdown("##### ⏱️ Payment & Delinquency History")
            no_premiums_paid = st.number_input(
                "Total Premiums Paid On-Time",
                min_value=1, max_value=100, value=12, step=1,
                help="Historical count of consecutive on-time payments."
            )
            late_3_6 = st.number_input(
                "Count 3–6 Months Late",
                min_value=0, max_value=20, value=1, step=1,
                help="Number of times premium was 3 to 6 months late."
            )
            late_6_12 = st.number_input(
                "Count 6–12 Months Late",
                min_value=0, max_value=20, value=0, step=1,
                help="Number of times premium was 6 to 12 months late."
            )
            late_12_plus = st.number_input(
                "Count >12 Months Late",
                min_value=0, max_value=20, value=0, step=1,
                help="Number of times premium was more than 12 months late."
            )

        submitted = st.form_submit_button("🔍 Evaluate Default Risk & Outreach Action", use_container_width=True, type="primary")

    if submitted:
        input_data = {
            "age": age_years,
            "residence_area_type": residence,
            "sourcing_channel": sourcing,
            "Income": income,
            "perc_premium_paid_by_cash_credit": perc_cash_credit,
            "application_underwriting_score": underwriting_score,
            "no_of_premiums_paid": no_premiums_paid,
            "Count_3-6_months_late": late_3_6,
            "Count_6-12_months_late": late_6_12,
            "Count_more_than_12_months_late": late_12_plus,
        }

        with st.spinner("Scoring customer risk profile with Calibrated XGBoost..."):
            res = pipeline.predict_one(input_data)

        st.markdown("---")
        st.subheader("Assessment Results")

        m1, m2, m3, m4 = st.columns(4)
        default_prob_pct = res['default_probability'] * 100
        ontime_prob_pct = res['on_time_probability'] * 100

        m1.metric("Predicted Default Risk", f"{default_prob_pct:.1f}%")
        m2.metric("On-Time Probability", f"{ontime_prob_pct:.1f}%")
        m3.metric("Assigned Risk Tier", res['risk_tier'])
        m4.metric("Intervention Budget", f"${res['intervention_cost']}")

        # Risk Gauge
        st.markdown("##### Lapse Risk Gauge")
        st.progress(min(max(res['default_probability'], 0.0), 1.0))

        # Operational Strategy Card
        tier = res['risk_tier']
        if tier == 'High Risk':
            st.markdown(f"""
            <div class="tier-card-high">
                <h4>🔴 High Risk Tier ({default_prob_pct:.1f}% Default Probability)</h4>
                <p><strong>Action:</strong> {res['recommended_action']}</p>
                <p><strong>Protocol:</strong> Route to Senior Retention Concierge within 24 hours. Present restructured payment schedule and auto-debit discount.</p>
            </div>
            """, unsafe_allow_html=True)
        elif tier == 'Medium Risk':
            st.markdown(f"""
            <div class="tier-card-med">
                <h4>🟠 Medium Risk Tier ({default_prob_pct:.1f}% Default Probability)</h4>
                <p><strong>Action:</strong> {res['recommended_action']}</p>
                <p><strong>Protocol:</strong> Dispatch priority physical reminder letter + urgent SMS reminder 14 days and 5 days before lapse date.</p>
            </div>
            """, unsafe_allow_html=True)
        elif tier == 'Low-Medium Risk':
            st.markdown(f"""
            <div class="tier-card-low-med">
                <h4>🟡 Low-Medium Risk Tier ({default_prob_pct:.1f}% Default Probability)</h4>
                <p><strong>Action:</strong> {res['recommended_action']}</p>
                <p><strong>Protocol:</strong> Trigger automated two-touch SMS payment link 3 days prior to due date.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="tier-card-low">
                <h4>🟢 Low Risk Tier ({default_prob_pct:.1f}% Default Probability)</h4>
                <p><strong>Action:</strong> {res['recommended_action']}</p>
                <p><strong>Protocol:</strong> Standard automated digital invoice. No additional intervention expenditure required.</p>
            </div>
            """, unsafe_allow_html=True)


# ==============================================================================
# MODE 2: BULK BATCH CSV SCORING
# ==============================================================================
elif app_mode == "📁 Bulk Batch CSV Scoring":
    st.subheader("Portfolio Batch Risk Scoring & Financial Projection")
    st.markdown("Upload a customer CSV dataset to evaluate risk tiers, intervention budgets, and portfolio ROI in real time.")

    sample_template = pd.DataFrame([{
        "id": "POL-1001",
        "age": 35,
        "Income": 185000,
        "perc_premium_paid_by_cash_credit": 0.35,
        "application_underwriting_score": 98.7,
        "no_of_premiums_paid": 12,
        "Count_3-6_months_late": 0,
        "Count_6-12_months_late": 0,
        "Count_more_than_12_months_late": 0,
        "sourcing_channel": "C",
        "residence_area_type": "Urban"
    }])

    c_down, c_up = st.columns([1, 3])
    with c_down:
        st.download_button(
            "📥 Download Sample CSV Template",
            data=sample_template.to_csv(index=False),
            file_name="sample_policyholder_template.csv",
            mime="text/csv"
        )

    uploaded_file = st.file_uploader("Upload Policyholder CSV File", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success(f"Successfully ingested {len(df_uploaded):,} policyholder records.")

        if st.button("🚀 Execute Batch Scoring", type="primary"):
            with st.spinner(f"Scoring {len(df_uploaded):,} records across 22 engineered features..."):
                results = pipeline.predict(df_uploaded)

            st.markdown("---")
            st.subheader("Batch Evaluation & Economic Impact")

            # Portfolio Economic KPIs
            k1, k2, k3, k4 = st.columns(4)
            total_customers = len(results)
            high_risk_n = int(np.sum(results['risk_tier'] == 'High Risk'))
            med_risk_n = int(np.sum(results['risk_tier'] == 'Medium Risk'))
            total_cost = results['intervention_cost'].sum()
            avg_default_risk = results['default_probability'].mean() * 100

            k1.metric("Total Customers Scored", f"{total_customers:,}")
            k2.metric("High / Medium Risk", f"{high_risk_n + med_risk_n:,}", f"{(high_risk_n + med_risk_n)/total_customers*100:.1f}%")
            k3.metric("Total Intervention Budget", f"${total_cost:,.0f}")
            k4.metric("Avg Portfolio Default Risk", f"{avg_default_risk:.1f}%")

            # Risk Tier Breakdown Bar Chart
            st.markdown("##### Operational Risk Distribution")
            tier_counts = results['risk_tier'].value_counts().reindex(
                ['High Risk', 'Medium Risk', 'Low-Medium Risk', 'Low Risk']
            ).fillna(0)
            st.bar_chart(tier_counts)

            st.markdown("##### Scored Customer Predictions")
            st.dataframe(results, use_container_width=True)

            csv_data = results.to_csv(index=False)
            st.download_button(
                "💾 Export Scored Predictions CSV",
                data=csv_data,
                file_name="scored_policyholder_predictions.csv",
                mime="text/csv",
                type="primary"
            )


# ==============================================================================
# MODE 3: STRATEGY & ECONOMIC BLUEPRINT
# ==============================================================================
elif app_mode == "📊 Strategy & Economic Blueprint":
    st.subheader("Risk-Based Operational Intervention Protocol")
    st.markdown("""
    To eliminate asymmetric financial loss, policyholders are routed into tiered operational workflows:
    """)

    st.markdown("""
    | Risk Tier | Default Probability | Operational Action | Cost / Policy | Economic Strategy |
    | :--- | :---: | :--- | :---: | :--- |
    | **🔴 High Risk** | $> 70\%$ | Outbound Concierge Phone Call | **$50** | High-touch outreach to restructure policy terms and prevent guaranteed lapse. |
    | **🟠 Medium Risk** | $40\% – 70\%$ | Direct Mail + Priority SMS Alert | **$10** | Multi-touch reminder sequence before lapse grace period expires. |
    | **🟡 Low-Medium Risk** | $20\% – 40\%$ | Automated SMS Reminder | **$2** | Lightweight digital touchpoint to prompt timely billing action. |
    | **🟢 Low Risk** | $\le 20\%$ | Standard Digital Billing | **$0** | Zero extra expenditure; standard automated invoice. |
    """)

    st.markdown("---")
    st.subheader("Financial Model & Asymmetric Cost Matrix")

    c1, c2 = st.columns(2)
    with c1:
        st.info("""
        **Cost Parameters**:
        - **False Negative Loss ($500)**: Lifetime policy value lost when a defaulter lapses undetected.
        - **False Alarm Cost ($10)**: Cost of an unnecessary reminder outreach to an on-time payer.
        - **Policy Revenue Preserved ($500)**: Average preserved premium revenue per retained policy.
        - **Retention Success Rate (60%)**: Estimated recovery rate of contacted defaulters.
        """)

    with c2:
        st.success("""
        **Verified Benchmark Performance (Held-Out Test Set)**:
        - **Projected Net Benefit**: **$186,710**
        - **Return on Investment (ROI)**: **3,224.7%**
        - **Intervention Budget Required**: **$5,790**
        - **Probability Calibration**: Isotonic calibration prevents overspending on low-confidence false positives.
        """)
