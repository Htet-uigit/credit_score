import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="Credit Score Predictor", layout="wide")

# --- Model Loading ---

# Load the saved model and artifacts
@st.cache_resource
def load_model():
    """Loads the model and artifacts from the 'model.pkl' file."""
    try:
        data = joblib.load('model.pkl')
        
        # Critical check to ensure all expected keys are present
        required_keys = ['model', 'encoders', 'target_encoder', 'scaler', 'loan_mlb', 'feature_names']
        for key in required_keys:
            if key not in data:
                st.error(f"Model artifact is missing required key: '{key}'. Please check your training script.")
                return None
        return data
    except FileNotFoundError:
        st.error("Model file 'model.pkl' not found. Please ensure the file is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None

# --- Prediction Function ---

def predict_score(input_data, model, encoders, target_encoder, scaler, loan_mlb, feature_names):
    """
    Processes user inputs, scales them, and performs prediction using the loaded model.
    """
    try:
        # --- 1. Extract Raw Inputs ---
        age = input_data['age']
        annual_income = input_data['annual_income']
        monthly_salary = input_data['monthly_salary']
        occupation = input_data['occupation']
        num_bank_accounts = input_data['num_bank_accounts']
        num_credit_cards = input_data['num_credit_cards']
        interest_rate = input_data['interest_rate']
        num_loans = input_data['num_loans']
        outstanding_debt = input_data['outstanding_debt']
        credit_utilization = input_data['credit_utilization']
        credit_history_years = input_data['credit_history_years']
        credit_mix = input_data['credit_mix']
        delay_from_due = input_data['delay_from_due']
        num_delayed_payment = input_data['num_delayed_payment']
        changed_credit_limit = input_data['changed_credit_limit']
        num_credit_inquiries = input_data['num_credit_inquiries']
        payment_min = input_data['payment_min']
        payment_behaviour = input_data['payment_behaviour']
        selected_loans = input_data['selected_loans']

        # --- 2. Encode Categoricals ---
        occ_enc = encoders['Occupation'].transform([occupation])[0]
        mix_enc = encoders['Credit_Mix'].transform([credit_mix])[0]
        pay_min_enc = encoders['Payment_of_Min_Amount'].transform([payment_min])[0]
        pay_beh_enc = encoders['Payment_Behaviour'].transform([payment_behaviour])[0]
        
        # --- 3. Feature Engineering ---
        history_months = credit_history_years * 12
        
        # --- 4. Loan Feature Vector (MultiLabelBinarizer) ---
        loan_feature_vector = [0] * len(loan_mlb.classes_)
        for loan in selected_loans:
            # Find the index of the selected loan type in the MLB classes
            idx = np.where(loan_mlb.classes_ == loan)[0][0]
            loan_feature_vector[idx] = 1
            
        loan_feature_columns = [f for f in feature_names if f.startswith('Has_')]

        # --- 5. Create Final Input Dictionary (CRITICAL: Must match feature_names) ---
        input_dict = {
             'Age': age, 
             'Occupation': occ_enc, 
             'Annual_Income': annual_income, 
             'Monthly_Inhand_Salary': monthly_salary, 
             'Num_Bank_Accounts': num_bank_accounts, 
             'Num_Credit_Card': num_credit_cards, 
             'Interest_Rate': interest_rate, 
             'Num_of_Loan': num_loans, 
             'Delay_from_due_date': delay_from_due, 
             'Num_of_Delayed_Payment': num_delayed_payment, 
             'Changed_Credit_Limit': changed_credit_limit, 
             'Num_Credit_Inquiries': num_credit_inquiries, 
             'Credit_Mix': mix_enc, 
             'Outstanding_Debt': outstanding_debt, 
             'Credit_Utilization_Ratio': credit_utilization, 
             'Credit_History_Age': history_months, 
             'Payment_of_Min_Amount': pay_min_enc, 
             'Payment_Behaviour': pay_beh_enc,
             
             # Defaulted features (missing from app UI, assumed imputed to 0.0 in training)
             'Total_EMI_per_month': 0.0, 
             'Amount_invested_monthly': 0.0, 
             'Monthly_Balance': 0.0,
        }
        
        # Merge loan features into the dictionary
        for col_name, value in zip(loan_feature_columns, loan_feature_vector):
            input_dict[col_name] = value

        # --- 6. Create DataFrame (Order is guaranteed by feature_names) ---
        input_df = pd.DataFrame([input_dict])[feature_names]
        
        # --- 7. Scale and Predict ---
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        class_name = target_encoder.inverse_transform(prediction)[0]
        
        return class_name, probability, target_encoder.classes_

    except ValueError as e:
        st.error(f"Prediction Error: A categorical value was not recognized by the model. Please check the inputs. Detail: {e}")
        return None, None, None
    except KeyError as e:
        st.error(f"Feature Mismatch Error: Missing feature in input dictionary: {e}. Check your feature names.")
        return None, None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during prediction: {e}")
        return None, None, None


# --- Streamlit UI ---

data = load_model()

if data is not None:
    # Unpack artifacts
    model = data['model']
    encoders = data['encoders']
    target_encoder = data['target_encoder']
    scaler = data['scaler']
    loan_mlb = data['loan_mlb']
    feature_names = data['feature_names']
    
    st.title("💳 Credit Score Classification App (Loan Types Included)")
    st.markdown("Use the form below to input customer data and predict their credit score (Good, Standard, or Poor).")
    st.markdown("---")

    # Safely extract options for select boxes
    occupation_options = encoders['Occupation'].classes_
    credit_mix_options = encoders['Credit_Mix'].classes_
    payment_min_options = encoders['Payment_of_Min_Amount'].classes_
    payment_behaviour_options = encoders['Payment_Behaviour'].classes_
    loan_types = loan_mlb.classes_


    # Create a form for input
    with st.form("prediction_form"):
        
        # --- INPUT SECTION 1: NUMERIC & BASIC CATEGORICAL ---
        st.subheader("Financial & Demographic Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Age", min_value=18, max_value=100, value=30)
            annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
            monthly_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0, value=4000.0, step=100.0)
            occupation = st.selectbox("Occupation", occupation_options)
            
        with col2:
            num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, value=2)
            num_credit_cards = st.number_input("Num Credit Cards", min_value=0, value=3)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0, value=5)
            num_loans = st.number_input("Num of Loans", min_value=0, value=1)
            
        with col3:
            outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1000.0)
            credit_utilization = st.slider("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
            credit_history_years = st.number_input("Credit History (Years)", min_value=0, value=5)
            credit_mix = st.selectbox("Credit Mix", credit_mix_options)
            
        # --- INPUT SECTION 2: PAYMENT BEHAVIOR ---
        st.subheader("Payment and Debt Behavior")
        pcol1, pcol2, pcol3 = st.columns(3)
        
        with pcol1:
            delay_from_due = st.number_input("Avg Delay from Due Date (Days)", min_value=0, value=5)
            num_delayed_payment = st.number_input("Num Delayed Payments", min_value=0, value=2)
        with pcol2:
            changed_credit_limit = st.number_input("Changed Credit Limit ($)", value=0.0)
            num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, value=1)
        with pcol3:
            payment_min = st.selectbox("Payment of Min Amount", payment_min_options)
            payment_behaviour = st.selectbox("Payment Behaviour", payment_behaviour_options)
        
        # --- INPUT SECTION 3: LOAN TYPES ---
        st.subheader("Types of Active Loans")
        selected_loans = st.multiselect(
            "Select all active loan types:",
            options=loan_types,
            default=[]
        )
            
        submit = st.form_submit_button("Predict Credit Score", type="primary")

    if submit:
        # Collect all inputs into a single dictionary
        input_data = {
            'age': age, 'annual_income': annual_income, 'monthly_salary': monthly_salary, 'occupation': occupation,
            'num_bank_accounts': num_bank_accounts, 'num_credit_cards': num_credit_cards, 'interest_rate': interest_rate,
            'num_loans': num_loans, 'outstanding_debt': outstanding_debt, 'credit_utilization': credit_utilization,
            'credit_history_years': credit_history_years, 'credit_mix': credit_mix, 'delay_from_due': delay_from_due,
            'num_delayed_payment': num_delayed_payment, 'changed_credit_limit': changed_credit_limit,
            'num_credit_inquiries': num_credit_inquiries, 'payment_min': payment_min, 'payment_behaviour': payment_behaviour,
            'selected_loans': selected_loans, # Pass the list of selected loans
        }

        with st.spinner('Calculating prediction...'):
            class_name, probability, classes = predict_score(input_data, model, encoders, target_encoder, scaler, loan_mlb, feature_names)

        if class_name:
            # --- DISPLAY RESULTS ---
            st.success(f"Predicted Credit Score: **{class_name}**")
            
            # Visualize Probabilities
            st.subheader("Prediction Probabilities")
            
            # Flatten the probability array for DataFrame creation
            if probability.shape[0] == 1:
                prob_data = probability[0]
            else:
                prob_data = probability.flatten()

            # Create DataFrame for display
            prob_df = pd.DataFrame(prob_data, index=classes, columns=['Probability'])
            
            # Display table with formatting
            st.dataframe(
                prob_df.sort_values(by='Probability', ascending=False)
                .style.format({'Probability': "{:.2%}"})
            )
            
            # Display bar chart
            st.bar_chart(prob_df.sort_values(by='Probability', ascending=False), color="#008080")

