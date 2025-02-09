# Customer-Churn-Prediction-Project


## Overview
This project predicts customer churn for a telecom company using  **Machine Learning Models**. The model takes customer details as input and determines the likelihood of churn.

## Features
- **User-friendly Streamlit Web App** for predictions.
- **Random Forest Model** trained on  customer data.
- **Preprocessing & Feature Engineering** to handle categorical and numerical variables.
- **Interactive Visualizations** for EDA and insights.

## Installation
### Prerequisites
Ensure you have **Python 3.9+** installed. Then, install the required dependencies:
```bash
pip install -r requirements.txt
```

### Install Streamlit
```bash
pip install streamlit
```

## Usage
### Running the Streamlit App
```bash
streamlit run app.py
```

### Inputs Required
- Tenure (months)
- Phone Service (Yes/No)
- Contract Type (Month-to-month, One-year, Two-year)
- Paperless Billing (Yes/No)
- Payment Method
- Monthly Charges
- Total Charges

### Output
- **Churn Prediction** (Yes/No)


## Model Details
- **Algorithms:** Random Forest Classifier, Logistic Regression, KNN Classifier, Bagging Classifier
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score
- **Feature Importance Analysis** performed

## Folder Structure
```
|-- app.py                      # Streamlit Web App
|-- model/
|   |-- model.pkl               # Trained Model
|-- notebooks/ 
|   |-- EDA.ipynb               #  EDA    
|   |-- model_building.ipynb    #  Model Building    
|-- data/
|   |-- WA_TelcoChurn.csv       # Dataset
|-- README.md                   # Project Documentation
|-- requirements.txt            # Required Python Libraries
```


## License
This project is licensed under the MIT License.

