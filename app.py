import streamlit as st
import pandas as pd
import numpy as np
import pickle # <-- Confirmed use of pickle
import os

# --- Configuration ---
# This file is expected to be created by the optimization script: 
# model_rf_optimized.pkl
MODEL_FILE = 'model.pkl'

# --- 1. Model Loading (Using Python's built-in pickle) ---

# Use st.cache_resource for heavy objects like models and transformers
@st.cache_resource
def load_model():
    """Loads the model and all transformers using pickle."""
    if not os.path.exists(MODEL_FILE):
        # A clear instruction if the required model file is missing
        st.error(f"Error: Model file '{MODEL_FILE}' not found. Please ensure it is in the root directory.")
        return None
    
    try:
        # Standard Python file opening and pickle loading
        with open(MODEL_FILE, 'rb') as file:
            data = pickle.load(file)
        
        st.success("Model artifacts loaded successfully (using pickle).")
        
        # Validation check for required components
        required_keys = ['model', 'encoders', 'target_encoder', 'scaler', 'feature_names']
        for key in required_keys:
            if key not in data:
                st.error(f"Model artifact is missing required key: '{key}'. Cannot run prediction.")
                return None
        
        return data

    except Exception as e:
        st.error(f"Error loading model using pickle: {e}. If the model was saved with joblib, you should use joblib.load().")
        return None

# Load all artifacts immediately
ARTIFACTS = load_model()

if ARTIFACTS is None:
    st.stop()

# Extract components from the loaded dictionary
model = ARTIFACTS['model']
encoders = ARTIFACTS['encoders']
target_encoder = ARTIFACTS['target_encoder']
scaler = ARTIFACTS['scaler']
FEATURE_NAMES = ARTIFACTS['feature_names']

# --- 2. Streamlit UI and Prediction Logic ---

st.set_page_config(
    page_title="Credit Score Predictor",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Credit Score Predictor (Random Forest)")
st.markdown("Enter the customer details below to predict their **Credit Score**.")

# --- Define Categorical Options (from encoders) ---

def get_encoder_map(encoder):
    """Creates a user-friendly map from the LabelEncoder's classes_ array."""
    try:
        # Check if the object has the classes_ attribute (standard for sklearn encoders)
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

# --- Input Fields ---

input_data = {}

# Use st.form to group inputs and prevent unnecessary recalculations
with st.form("prediction_form"):
    
    st.header("Personal & Financial Information")
    col1, col2 = st.columns(2)

    with col1:
        input_data['Age'] = st.number_input("Age", min_value=18, max_value=90, value=30, key='age')
        input_data['Annual_Income'] = st.number_input("Annual Income ($)", min_value=10000.0, value=75000.0, step=1000.0, key='income')
        input_data['Monthly_Inhand_Salary'] = st.number_input("Monthly In-hand Salary ($)", min_value=500.0, value=6250.0, step=100.0, key='salary')
        
    with col2:
        occupation_selection = st.selectbox("Occupation", options=list(OCCUPATION_MAP.keys()), key='occupation')
        # Use .get() for safe lookup, defaulting to 0 if key is missing (should not happen if maps are correct)
        input_data['Occupation'] = OCCUPATION_MAP.get(occupation_selection, 0)
        input_data['Num_Bank_Accounts'] = st.number_input("Number of Bank Accounts", min_value=0, max_value=10, value=3, key='bank_acct')
        input_data['Num_Credit_Card'] = st.number_input("Number of Credit Cards", min_value=0, max_value=15, value=4, key='cc_num')


    st.header("Credit History & Debt")
    col3, col4 = st.columns(2)

    with col3:
        input_data['Interest_Rate'] = st.slider("Interest Rate (%)", min_value=1, max_value=35, value=15, key='interest')
        input_data['Num_of_Loan'] = st.number_input("Number of Loans", min_value=0, max_value=20, value=5, key='loan_num')
        input_data['Delay_from_due_date'] = st.number_input("Average Delay (Days)", min_value=0, max_value=100, value=15, key='delay')
        input_data['Num_of_Delayed_Payment'] = st.number_input("Number of Delayed Payments", min_value=0, max_value=30, value=6, key='delayed_pay')
        input_data['Changed_Credit_Limit'] = st.number_input("Change in Credit Limit (%)", min_value=-50.0, max_value=100.0, value=10.0, key='limit_change')
        
    with col4:
        input_data['Num_Credit_Inquiries'] = st.number_input("Credit Inquiries (Last 6 months)", min_value=0, max_value=25, value=3, key='inquiries')
        credit_mix_selection = st.selectbox("Credit Mix", options=list(CREDIT_MIX_MAP.keys()), key='credit_mix')
        input_data['Credit_Mix'] = CREDIT_MIX_MAP.get(credit_mix_selection, 0)
        input_data['Outstanding_Debt'] = st.number_input("Outstanding Debt ($)", min_value=0.0, value=8000.0, step=100.0, key='debt')
        input_data['Credit_Utilization_Ratio'] = st.slider("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=35.0, step=0.1, key='utilization')
        input_data['Credit_History_Age'] = st.number_input("Credit History Age (Months)", min_value=0, value=100, key='history_age')


    st.header("Payment Behavior")
    col5, col6 = st.columns(2)

    with col5:
        min_pay_selection = st.selectbox("Payment of Minimum Amount", options=list(PAYMENT_MIN_MAP.keys()), key='min_pay')
        input_data['Payment_of_Min_Amount'] = PAYMENT_MIN_MAP.get(min_pay_selection, 0)
        input_data['Total_EMI_per_month'] = st.number_input("Total EMI per Month ($)", min_value=0.0, value=500.0, step=10.0, key='emi')
        
    with col6:
        pay_behavior_selection = st.selectbox("Payment Behaviour", options=list(PAYMENT_BEHAVIOUR_MAP.keys()), key='pay_behavior')
        input_data['Payment_Behaviour'] = PAYMENT_BEHAVIOUR_MAP.get(pay_behavior_selection, 0)
        input_data['Amount_invested_monthly'] = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=200.0, step=10.0, key='invested')
        input_data['Monthly_Balance'] = st.number_input("Monthly Balance ($)", min_value=0.0, value=3000.0, step=100.0, key='balance')
        
    # Ensure all features expected by the model are present, setting missing ones to a safe default (like 0.0)
    # This prevents the model from failing if a feature was included in training but is missing from the simplified UI
    for feature in FEATURE_NAMES:
        if feature not in input_data:
            input_data[feature] = 0.0

    submit_button = st.form_submit_button("Predict Credit Score", type="primary")

# --- Prediction Logic ---
if submit_button:
    
    try:
        # 1. Convert input data to a DataFrame in the correct feature order
        input_df = pd.DataFrame([input_data])
        # IMPORTANT: Select and order columns according to FEATURE_NAMES from the artifact
        input_df = input_df[FEATURE_NAMES] 

        # 2. Scale the input features
        X_scaled = scaler.transform(input_df)

        # 3. Predict the encoded score (0, 1, or 2)
        with st.spinner('Calculating prediction...'):
            prediction_encoded = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)

        # 4. Inverse transform to get the human-readable score
        prediction_label = target_encoder.inverse_transform([prediction_encoded])[0]

        st.subheader("Prediction Result")
        
        score_color = ""
        if prediction_label == 'Good':
            score_color = "green"
        elif prediction_label == 'Standard':
            score_color = "orange"
        else: # Poor
            score_color = "red"

        st.markdown(
            f"""
            <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
                <p style="font-size: 16px; margin: 0;">The Predicted Credit Score is:</p>
                <h1 style="color: {score_color}; margin: 5px 0 0 0;">{prediction_label}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display Probabilities
        st.markdown("---")
        st.subheader("Confidence Scores")
        
        prob_df = pd.DataFrame(probability.T, index=target_encoder.classes_, columns=['Probability'])
        
        st.dataframe(
            prob_df.sort_values(by='Probability', ascending=False)
            .style.format({'Probability': "{:.2%}"})
        )
        st.bar_chart(prob_df.sort_values(by='Probability', ascending=False))

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.sidebar.markdown(
    """
    ## Deployment Details
    
    * **Model:** Random Forest Classifier
    * **Model File:** `{MODEL_FILE}`
    * **Loader:** `pickle`
    """

)

