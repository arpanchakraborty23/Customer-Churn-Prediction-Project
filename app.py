import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load the pre-trained model
model_path = r"NoteBook\model\model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Title of the Streamlit app
st.title("Customer Churn Prediction")

# Sidebar for dataset upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# If file is uploaded, display dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.write(df.head())

    # Show dataset description
    st.write("### Dataset Info")
    st.write(df.describe())

    # Churn Distribution Visualization
    st.write("### Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Churn", data=df, palette="coolwarm", ax=ax)
    st.pyplot(fig)

    # Monthly Charges Distribution
    st.write("### Monthly Charges Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["MonthlyCharges"], kde=True, bins=30, color="blue", ax=ax)
    st.pyplot(fig)

# Prediction Section
st.sidebar.header("Customer Data for Prediction")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, step=1)
phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.sidebar.number_input("Monthly Charges", min_value=0.0, step=0.1)
total_charges = st.sidebar.number_input("Total Charges", min_value=0.0, step=0.1)

# Create DataFrame for prediction
user_data = pd.DataFrame({
    "Gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "Tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Predict Churn
if st.sidebar.button("Predict Churn"):
    prediction = model.predict(user_data)
    churn_result = "Yes" if prediction[0] == 1 else "No"
    st.write(f"### Predicted Churn: **{churn_result}**")

