import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

from src.inference import InferencePipeline

# Page Config
st.set_page_config(
    page_title="Insurance Premium Default Risk Profiler",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-high {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        color: #991B1B;
    }
    .card-medium {
        background-color: #FFEDD5;
        border-left: 6px solid #F97316;
        padding: 1rem;
        border-radius: 8px;
        color: #9A3412;
    }
    .card-low-med {
        background-color: #FEF3C7;
        border-left: 6px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        color: #92400E;
    }
    .card-low {
        background-color: #D1FAE5;
        border-left: 6px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        color: #065F46;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    try:
        return InferencePipeline()
    except ModuleNotFoundError as me:
        st.error(f"⚠️ Missing dependency in environment: {me}")
        st.info("💡 **Tip**: Please run Streamlit using your virtual environment:\n```bash\n.\\venv\\Scripts\\streamlit run app.py\n```")
        return None
    except Exception as e:
        st.error(f"Failed to load inference pipeline: {e}")
        return None

pipeline = load_pipeline()


# Title Header
st.markdown('<div class="main-header">🛡️ Insurance Premium Default Risk Profiler</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict customer payment lapse risk and get instant risk-based intervention recommendations.</div>', unsafe_allow_html=True)

if pipeline is None:
    st.error("⚠️ Model artifact missing! Please run `python train.py` first to train and generate `models/best_model.pkl`.")
    st.stop()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/shield.png", width=70)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose Mode:",
    ["👤 Single Customer Scoring", "📁 Bulk Batch CSV Scoring", "📊 Risk Strategy Reference"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model Information")
st.sidebar.info(f"**Loaded Model Path:**\n`{pipeline.model_path.name}`")


if app_mode == "👤 Single Customer Scoring":
    st.subheader("Customer Risk Assessment Form")
    st.caption("Enter customer demographic, financial, and payment history metrics below. Mandatory and categorical fields include guidance tooltips.")

    with st.form("single_customer_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 👤 Customer Demographics")
            age_years = st.number_input(
                "Age (Years)",
                min_value=18, max_value=100, value=35, step=1,
                help="Customer age in years (calculated from age in days)."
            )
            residence_area_type = st.selectbox(
                "Residence Area Type",
                options=["Urban", "Rural"],
                index=0,
                help="Select customer residence classification."
            )
            sourcing_channel = st.selectbox(
                "Sourcing Channel",
                options=["A", "B", "C", "D", "E"],
                index=2,
                help="Select acquisition channel code (A, B, C, D, or E)."
            )

        with col2:
            st.markdown("##### 💳 Financial & Policy Metrics")
            income = st.number_input(
                "Annual Income ($)",
                min_value=10000, max_value=5000000, value=250000, step=5000,
                help="Customer reported annual income."
            )
            perc_cash_credit = st.slider(
                "% Premium Paid by Cash/Credit",
                min_value=0.0, max_value=1.0, value=0.40, step=0.01,
                help="Ratio of premium paid via cash or credit card (0.0 = 0%, 1.0 = 100%)."
            )
            underwriting_score = st.slider(
                "Application Underwriting Score (0-100)",
                min_value=0.0, max_value=100.0, value=98.5, step=0.1,
                help="Credit/underwriting risk score evaluated at application."
            )

        with col3:
            st.markdown("##### ⏱️ Premium Payment History")
            no_premiums_paid = st.number_input(
                "Total Premiums Paid On-Time",
                min_value=1, max_value=100, value=12, step=1,
                help="Total number of consecutive on-time premiums paid."
            )
            late_3_6m = st.number_input(
                "Count 3-6 Months Late",
                min_value=0, max_value=20, value=0, step=1,
                help="Number of times payments were 3 to 6 months late."
            )
            late_6_12m = st.number_input(
                "Count 6-12 Months Late",
                min_value=0, max_value=20, value=0, step=1,
                help="Number of times payments were 6 to 12 months late."
            )
            late_12m_plus = st.number_input(
                "Count >12 Months Late",
                min_value=0, max_value=20, value=0, step=1,
                help="Number of times payments were over 12 months late."
            )

        submitted = st.form_submit_button("🔍 Calculate Risk & Recommendation", use_container_width=True)

    if submitted:
        input_data = {
            "id": 99999,
            "age_in_days": age_years * 365,
            "perc_premium_paid_by_cash_credit": perc_cash_credit,
            "Income": income,
            "Count_3-6_months_late": late_3_6m,
            "Count_6-12_months_late": late_6_12m,
            "Count_more_than_12_months_late": late_12m_plus,
            "application_underwriting_score": underwriting_score,
            "no_of_premiums_paid": no_premiums_paid,
            "sourcing_channel": sourcing_channel,
            "residence_area_type": residence_area_type
        }

        with st.spinner("Analyzing risk profile..."):
            df_single = pd.DataFrame([input_data])
            res_df = pipeline.predict(df_single)
            result = res_df.iloc[0]

        st.markdown("---")
        st.subheader("📊 Assessment Results")

        res_col1, res_col2, res_col3, res_col4 = st.columns(4)
        res_col1.metric("Non-Payer Risk Probability", f"{result['non_payer_probability']*100:.1f}%")
        res_col2.metric("On-Time Probability", f"{result['on_time_probability']*100:.1f}%")
        res_col3.metric("Intervention Action Cost", f"${result['intervention_cost']}")
        res_col4.metric("Assigned Risk Tier", result['risk_tier'])

        # Risk badge display
        tier = result['risk_tier']
        if tier == 'High Risk':
            st.markdown(f"""
            <div class="card-high">
                <h3>🔴 High Risk Customer (Non-Payer Prob: {result['non_payer_probability']*100:.1f}%)</h3>
                <p><strong>Recommended Action:</strong> {result['intervention_action']}</p>
                <p><strong>Strategy:</strong> Immediate personal phone outreach by senior customer success manager + tailored payment plan options.</p>
            </div>
            """, unsafe_allow_html=True)
        elif tier == 'Medium Risk':
            st.markdown(f"""
            <div class="card-medium">
                <h3>🟠 Medium Risk Customer (Non-Payer Prob: {result['non_payer_probability']*100:.1f}%)</h3>
                <p><strong>Recommended Action:</strong> {result['intervention_action']}</p>
                <p><strong>Strategy:</strong> Automated email sequence + SMS reminder 14 days and 7 days prior to premium due date.</p>
            </div>
            """, unsafe_allow_html=True)
        elif tier == 'Low-Medium Risk':
            st.markdown(f"""
            <div class="card-low-med">
                <h3>🟡 Low-Medium Risk Customer (Non-Payer Prob: {result['non_payer_probability']*100:.1f}%)</h3>
                <p><strong>Recommended Action:</strong> {result['intervention_action']}</p>
                <p><strong>Strategy:</strong> Standard SMS reminder 3 days prior to payment due date.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card-low">
                <h3>🟢 Low Risk Customer (Non-Payer Prob: {result['non_payer_probability']*100:.1f}%)</h3>
                <p><strong>Recommended Action:</strong> {result['intervention_action']}</p>
                <p><strong>Strategy:</strong> Standard digital invoice; no extra intervention required.</p>
            </div>
            """, unsafe_allow_html=True)


elif app_mode == "📁 Bulk Batch CSV Scoring":
    st.subheader("Bulk Batch CSV Customer Risk Scoring")
    st.markdown("Upload a CSV dataset containing customer policy records to calculate risk scores for all rows at once.")

    sample_template = pd.DataFrame([{
        "id": 1001,
        "age_in_days": 14000,
        "perc_premium_paid_by_cash_credit": 0.35,
        "Income": 180000,
        "Count_3-6_months_late": 0,
        "Count_6-12_months_late": 0,
        "Count_more_than_12_months_late": 0,
        "application_underwriting_score": 98.5,
        "no_of_premiums_paid": 10,
        "sourcing_channel": "A",
        "residence_area_type": "Urban"
    }])

    st.download_button(
        "📥 Download Sample CSV Template",
        data=sample_template.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_uploaded = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded {len(df_uploaded)} records.")

        if st.button("🚀 Run Batch Prediction", type="primary"):
            with st.spinner("Processing customer records..."):
                results = pipeline.predict(df_uploaded)

            st.markdown("---")
            st.subheader("Batch Prediction Results")

            # Summary metrics
            t_col1, t_col2, t_col3 = st.columns(3)
            high_risk_count = np.sum(results['risk_tier'] == 'High Risk')
            total_cost = results['intervention_cost'].sum()
            avg_risk = results['non_payer_probability'].mean() * 100

            t_col1.metric("High Risk Customers Flagged", f"{high_risk_count} ({high_risk_count/len(results)*100:.1f}%)")
            t_col2.metric("Total Intervention Budget Required", f"${total_cost:,.0f}")
            t_col3.metric("Average Portfolio Non-Payer Prob", f"{avg_risk:.1f}%")

            st.dataframe(results, use_container_width=True)

            csv_data = results.to_csv(index=False)
            st.download_button(
                "💾 Download Scored CSV Predictions",
                data=csv_data,
                file_name="scored_customer_predictions.csv",
                mime="text/csv",
                type="primary"
            )


elif app_mode == "📊 Risk Strategy Reference":
    st.subheader("Intervention Matrix & Risk Tier Definitions")
    st.markdown("""
    | Risk Tier | Non-Payer Probability | Intervention Action | Cost per Policy | Strategy |
    | :--- | :--- | :--- | :--- | :--- |
    | **🔴 High Risk** | > 70% | Personal Call + Special Offer | $50 | High-touch outreach to prevent high-value policy lapses. |
    | **🟠 Medium Risk** | 40% – 70% | Email + SMS Reminder | $10 | Multi-channel automated reminders prior to due date. |
    | **🟡 Low-Medium Risk** | 20% – 40% | SMS Reminder | $2 | Light-touch automated SMS notification. |
    | **🟢 Low Risk** | < 20% | Standard Communication | $0 | Regular digital billing communication. |
    """)

    st.markdown("---")
    st.markdown("### Financial ROI & Model Parameters")
    st.info("""
    - **False Negative Cost ($500)**: Net revenue lost when a non-paying customer lapses without intervention.
    - **False Alarm Cost ($10)**: Expense incurred for unnecessary intervention on an on-time payer.
    - **Expected Retention Rate (60%)**: Estimated proportion of flagged non-payers saved through proactive intervention.
    """)
