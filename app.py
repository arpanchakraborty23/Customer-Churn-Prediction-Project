import streamlit as st
import pandas as pd
import pickle

# Load the Random Forest model from the pickle file
model = pickle.load(open(r'NoteBook\model\model.pkl', 'rb'))

# Define the columns for user input
columns = ['tenure', 'PhoneService', 'Contract',
           'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

# Create a function to preprocess user input and make predictions
def predict_churn(input_data):
    # Preprocess the input data
    input_df = pd.DataFrame([input_data], columns=columns)

    # Make predictions using the loaded model
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1][0]  # Extract single value

    return prediction[0], probability

# Create the Streamlit app
def main():
    st.title("Telecom Churn Prediction")
    st.write("Enter the customer details below to predict churn.")

    # Create input fields for user input
    tenure = st.slider("Tenure (months)", 0, 100, 1)

    phone_service = st.selectbox("Phone Service", [0, 1])
    st.write("0: No, 1: Yes")
    contract = st.selectbox("Contract", [0, 1, 2])
    st.write("0: Month-to-month, 1: One year, 2: Two year")
    paperless_billing = st.selectbox("Paperless Billing", [0, 1])
    st.write("0: No, 1: Yes")
    payment_method = st.selectbox("Payment Method", [0, 1, 2, 3])
    st.write("0: Bank transfer (automatic), 1: Credit card (automatic), 2: Electronic check, 3: Mailed check")
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
    total_charges = st.number_input("Total Charges", min_value=0.0, step=0.1)

    # Create a dictionary to store the user input
    input_data = {
        'tenure': tenure,
        'PhoneService': phone_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Add Predict Button
    if st.button("Predict Churn"):
        # Predict churn based on user input
        churn_prediction, churn_probability = predict_churn(input_data)
        print(churn_prediction)

        # Display the prediction
        st.subheader("Churn Prediction")
        if churn_probability >= 0.4:
            st.write("The customer is **likely** to churn.")
        else:
            st.write("The customer is **unlikely** to churn.")

        # Display the churn probability
        st.subheader("Churn Probability")
        st.write(f"The probability of churn is: **{churn_probability:.2f}**")

# Run the Streamlit app
if __name__ == '__main__':
    main()
