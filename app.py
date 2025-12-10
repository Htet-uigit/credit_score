import streamlit as st
import pandas as pd
import numpy as np
import joblib 
import os
from sklearn.preprocessing import LabelEncoder

MODEL_FILE = 'model.pkl' 

IMAGE_PATHS = {
    'Good': 'good.jpg',     
    'Poor': 'poor.jpg',     
    'Standard': 'standard.png' 
}

# --- 1. Model Loading (Using joblib) ---

@st.cache_resource
def load_model():
    """Loads the model and all transformers using joblib."""
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the root directory.")
        return None
    
    try:
        with open(MODEL_FILE, 'rb') as file:
            # Use joblib.load() to deserialize the file
            data = joblib.load(file) 
        
        st.success("Model artifacts loaded successfully (using joblib).")
        
        required_keys = ['model', 'encoders', 'target_encoder', 'scaler', 'feature_names']
        for key in required_keys:
            if key not in data:
                st.error(f"Model artifact is missing required key: '{key}'. Cannot run prediction.")
                return None
        
        return data

    except Exception as e:
        # Corrected error message points to joblib
        st.error(f"Error loading model using joblib: {e}. Check if the model file '{MODEL_FILE}' was saved correctly.")
        return None

ARTIFACTS = load_model()

if ARTIFACTS is None:
    st.stop()

model = ARTIFACTS['model']
encoders = ARTIFACTS['encoders']
target_encoder = ARTIFACTS['target_encoder']
scaler = ARTIFACTS['scaler'] 
FEATURE_NAMES = ARTIFACTS['feature_names']

# --- 2. Helper Functions & Mappings ---

def get_encoder_map(encoder):
    """Creates a user-friendly map from the LabelEncoder's classes_ array."""
    try:
        if hasattr(encoder, 'classes_'):
            return {cls: i for i, cls in enumerate(encoder.classes_)}
        else:
            return {}
    except:
        return {}

OCCUPATION_MAP = get_encoder_map(encoders.get('Occupation', LabelEncoder()))
CREDIT_MIX_MAP = get_encoder_map(encoders.get('Credit_Mix', LabelEncoder()))
PAYMENT_MIN_MAP = get_encoder_map(encoders.get('Payment_of_Min_Amount', LabelEncoder()))
PAYMENT_BEHAVIOUR_MAP = get_encoder_map(encoders.get('Payment_Behaviour', LabelEncoder()))

# --- Prediction Display Function (Image Display & Recommendations) ---

def display_prediction_result(prediction_label):
    """Displays prediction result using native Streamlit components, an image, and styled recommendations."""
    
    # Nested function to create the styled HTML box
    def create_styled_box(title, items_markdown, color):
        """Generates a styled HTML box for recommendations (DO or DON'T)."""
        if color == 'green':
            bg_color = '#e6ffe6'
            text_color = '#006400'
            border_color = '#006400'
        elif color == 'red':
            bg_color = '#ffebe6'
            text_color = '#8b0000'
            border_color = '#8b0000'
        else:
             bg_color = '#f0f2f6'
             text_color = 'black'
             border_color = '#ccc'
             
        # Convert markdown list to HTML <ul> with bold formatting for key words
        html_items = "".join([
            f"<li>{item.strip().replace('**', '<b>').replace('**', '</b>')}</li>" 
            for item in items_markdown.strip().split('* ')[1:] if item.strip()
        ])
        
        return f"""
        <div style="padding: 15px; border-radius: 10px; background-color: {bg_color}; border: 2px solid {border_color};">
            <h4 style="color: {text_color}; margin-top: 0; margin-bottom: 10px;">{title}</h4>
            <ul style="margin: 0; padding-left: 20px; color: {text_color}; font-size: 14px;">
                {html_items}
            </ul>
        </div>
        """
    
    image_path = IMAGE_PATHS.get(prediction_label)
    
    result_col, image_col = st.columns([1, 1])

    with result_col:
        if prediction_label == 'Good':
            st.success(f"## ✅ Excellent News! Predicted Score: {prediction_label}", icon="💰") 
            st.markdown("Your financial habits are strong!")
        elif prediction_label == 'Standard':
            st.warning(f"## 🌟 Notice: Predicted Score: {prediction_label}", icon="🟡")
            st.markdown("Your credit is acceptable, but there is room for improvement.")
        else: # Poor
            st.error(f"## ⚠️ Urgent: Predicted Score: {prediction_label}", icon="🛑")
            st.markdown("Immediate attention required to improve your score.")
        
        st.markdown("---")
    
    with image_col:
        # Checks if the image file exists before trying to display it
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption=f"Result: {prediction_label}", use_column_width=True)
        else:
            # This is the warning if files are not found (location/name mismatch)
            st.warning(f"Image not found at: {image_path}. Please check your file names and ensure they are in the project folder.")
            
    st.markdown("---") 
    
    st.header("📈 Actionable Recommendations")
    
    if prediction_label == 'Good':
        st.subheader("MAINTENANCE MODE: Protect Your Excellent Score")
        dos = """
        * Keep your **Credit Utilization Ratio** below **10%** (ideally 1-5%).
        * Continue making **all payments on time**, every time.
        * Review your credit report annually for errors.
        """
        donts = """
        * Co-sign loans for others, as their mistakes become yours.
        * Close old, established credit accounts, as this lowers your average credit history age.
        """
        
    elif prediction_label == 'Standard':
        st.subheader("IMPROVEMENT FOCUS: Move to Excellent")
        dos = """
        * Prioritize reducing your **Outstanding Debt**. Aim to get your **Credit Utilization Ratio below 30%**.
        * Pay more than the minimum amount due whenever possible.
        * Set up **automatic payments** to eliminate late payment risk.
        """
        donts = """
        * Apply for multiple new lines of credit simultaneously; this generates hard inquiries.
        * Let your balance on any card get close to its limit.
        """
        
    else: # Poor
        st.subheader("⚠️ URGENT ACTION NEEDED: Rebuild Your Score")
        dos = """
        * Immediately focus on clearing **delayed payments** and paying down high-interest debt.
        * Contact your creditors or lenders to discuss potential **payment plans** for overdue accounts.
        * Check your credit report for collection accounts and prioritize resolving them.
        """
        donts = """
        * Use credit cards for day-to-day purchases while carrying high debt.
        * Ignore calls or letters from creditors, as communication is key to negotiation.
        """
        
    # Display DO'S and DON'TS in styled boxes 
    do_col, dont_col = st.columns(2)

    with do_col:
        st.markdown(create_styled_box("✅ DO'S (Focus Areas)", dos, 'green'), unsafe_allow_html=True)
        
    with dont_col:
        st.markdown(create_styled_box("🛑 DON'TS (Avoid These)", donts, 'red'), unsafe_allow_html=True)
        
    st.markdown("---") # Final separator


st.set_page_config(
    page_title="Credit Score Predictor",
    layout="wide", 
    initial_sidebar_state="auto"
)

# --- Enhanced Header ---
st.markdown("<h1>🏦 Credit Score Prediction Application</h1>", unsafe_allow_html=True)
st.caption("A Machine Learning model to predict credit score based on key financial indicators. Fill out the sections below and press 'Predict'.")
st.markdown("---")

# Initialize session state for interactive metrics
if 'utilization' not in st.session_state:
    st.session_state['utilization'] = 35.0
if 'delay' not in st.session_state:
    st.session_state['delay'] = 15
    
input_data = {}

# --- INTERACTIVE METRICS SECTION ---
st.markdown("### 📊 Interactive Health Indicators")
st.markdown("These indicators update as you adjust the input fields related to utilization and payment delay.")
metric_col1, metric_col2, _, _ = st.columns(4)

current_utilization = st.session_state['utilization']
current_delay = st.session_state['delay']

with metric_col1:
    if current_utilization <= 30:
        risk_caption = "Optimal Utilization (Recommended)"
        risk_delta = "LOW RISK"
        risk_color = "normal"
    elif current_utilization <= 50:
        risk_caption = "Monitor Utilization"
        risk_delta = "MODERATE RISK"
        risk_color = "inverse"
    else:
        risk_caption = "High Utilization"
        risk_delta = "HIGH RISK"
        risk_color = "inverse"

    st.metric(
        label="Credit Utilization Risk",
        value=f"{current_utilization:.1f}%",
        delta=risk_delta,
        delta_color=risk_color
    )
    st.caption(risk_caption)

with metric_col2:
    if current_delay <= 5:
        delay_caption = "Excellent Payment History"
        delay_delta = "ON TRACK"
        delay_color = "normal"
    elif current_delay <= 15:
        delay_caption = "Acceptable Payment History"
        delay_delta = "MINOR ISSUE"
        delay_color = "inverse"
    else:
        delay_caption = "Frequent Delays"
        delay_delta = "MAJOR ISSUE"
        delay_color = "inverse"

    st.metric(
        label="Payment Health (Avg Delay)",
        value=f"{current_delay} Days",
        delta=delay_delta,
        delta_color=delay_color
    )
    st.caption(delay_caption)

st.markdown("---")


# --- Input Fields using Expanders for better UI ---

with st.form("prediction_form"):
    
    with st.expander("👤 Personal & Financial Information", expanded=True):
        st.markdown("Enter basic demographic and income details.")
        col1, col2 = st.columns(2)

        with col1:
            input_data['Age'] = st.number_input("Age", min_value=18, max_value=90, value=30, key='age')
            input_data['Annual_Income'] = st.number_input("Annual Income ($)", min_value=10000.0, max_value=500000.0, value=75000.0, step=1000.0, key='income')
            input_data['Monthly_Inhand_Salary'] = st.number_input("Monthly In-hand Salary ($)", min_value=500.0, max_value=50000.0, value=6250.0, step=100.0, key='salary')
            
        with col2:
            occupation_selection = st.selectbox("Occupation", options=list(OCCUPATION_MAP.keys()), key='occupation')
            input_data['Occupation'] = OCCUPATION_MAP.get(occupation_selection, 0)
            input_data['Num_Bank_Accounts'] = st.number_input("Number of Bank Accounts", min_value=0, max_value=10, value=3, key='bank_acct')
            input_data['Num_Credit_Card'] = st.number_input("Number of Credit Cards", min_value=0, max_value=15, value=4, key='cc_num')
    
    # 2. Credit History & Debt Expander
    with st.expander("💳 Credit History & Debt", expanded=False):
        st.markdown("Provide details about your current debt load and credit usage.")
        col3, col4 = st.columns(2)

        with col3:
            input_data['Interest_Rate'] = st.slider("Interest Rate (%)", min_value=1, max_value=35, value=15, key='interest')
            input_data['Num_of_Loan'] = st.number_input("Number of Loans", min_value=0, max_value=20, value=5, key='loan_num')
            input_data['Delay_from_due_date'] = st.number_input("Average Delay (Days)", min_value=0, max_value=100, value=15, key='delay')
            input_data['Num_of_Delayed_Payment'] = st.number_input("Number of Delayed Payments", min_value=0, max_value=30, value=6, key='delayed_pay')
            input_data['Changed_Credit_Limit'] = st.slider("Change in Credit Limit (%)", min_value=-30.0, max_value=30.0, value=10.0, step=1.0, key='limit_change')
            
        with col4:
            input_data['Num_Credit_Inquiries'] = st.number_input("Credit Inquiries (Last 6 months)", min_value=0, max_value=25, value=3, key='inquiries')
            credit_mix_selection = st.selectbox("Credit Mix", options=list(CREDIT_MIX_MAP.keys()), key='credit_mix')
            input_data['Credit_Mix'] = CREDIT_MIX_MAP.get(credit_mix_selection, 0)
            input_data['Outstanding_Debt'] = st.number_input("Outstanding Debt ($)", min_value=0.0, max_value=50000.0, value=8000.0, step=100.0, key='debt')
            input_data['Credit_Utilization_Ratio'] = st.slider("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=35.0, step=0.1, key='utilization')
            input_data['Credit_History_Age'] = st.number_input("Credit History Age (Months)", min_value=0, value=100, key='history_age')

    with st.expander("🕰️ Payment Behavior & Balances", expanded=False):
        st.markdown("Detail your monthly payments, investments, and account behavior.")
        col5, col6 = st.columns(2)

        with col5:
            min_pay_selection = st.selectbox("Payment of Minimum Amount", options=list(PAYMENT_MIN_MAP.keys()), key='min_pay')
            input_data['Payment_of_Min_Amount'] = PAYMENT_MIN_MAP.get(min_pay_selection, 0)
            input_data['Total_EMI_per_month'] = st.number_input("Total EMI per Month ($)", min_value=0.0, max_value=5000.0, value=500.0, step=10.0, key='emi')
            
        with col6:
            pay_behavior_selection = st.selectbox("Payment Behaviour", options=list(PAYMENT_BEHAVIOUR_MAP.keys()), key='pay_behavior')
            input_data['Payment_Behaviour'] = PAYMENT_BEHAVIOUR_MAP.get(pay_behavior_selection, 0)
            input_data['Amount_invested_monthly'] = st.number_input("Amount Invested Monthly ($)", min_value=0.0, max_value=5000.0, value=200.0, step=10.0, key='invested')
            input_data['Monthly_Balance'] = st.number_input("Monthly Balance ($)", min_value=0.0, max_value=50000.0, value=3000.0, step=100.0, key='balance')
            
    for feature in FEATURE_NAMES:
        if feature not in input_data:
            input_data[feature] = 0.0

    st.markdown("---")
    submit_button = st.form_submit_button("💰 Predict Credit Score", type="primary")

# --- Prediction Logic ---
if submit_button:
    
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_NAMES] 

        X_scaled = scaler.transform(input_df)

        with st.spinner('Calculating prediction...'):
            prediction_encoded = model.predict(X_scaled)[0]

        prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

        st.subheader("Prediction Result")
        
        display_prediction_result(prediction_label) 

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")


st.sidebar.subheader("Owner's Information")
st.sidebar.write(f"*Name:* Htet Myat Phone Naing")
st.sidebar.write(f"*Student ID:* PIUS20230054")

st.sidebar.markdown(

    """
    ## 🛠️ Deployment Details
    
    * **Model:** Random Forest Classifier
    * **Model File:** `{MODEL_FILE}`
    * **Loader:** `joblib`
    """
)

