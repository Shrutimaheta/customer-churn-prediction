"""
predict_new_customer.py
========================
Demonstrates the whole POINT of saving a model: using it later on a brand
new customer without retraining anything.

THEORY: This is what "deploying" a model conceptually means (before you
add a web app/API around it). You take the exact same encoding + scaling
steps used during training and apply them to new raw data, then call
model.predict().

IMPORTANT: You must preprocess new data EXACTLY the same way as training
data (same columns, same order, same encoding) or the model's numbers
will mean something completely different -- this is a very common bug.
"""

import os
import sys

import pandas as pd
import joblib

# The model files are created by churn_prediction.py -- train first.
for f in ("churn_model.pkl", "model_columns.pkl"):
    if not os.path.exists(f):
        sys.exit(f"Missing {f}. Run 'python churn_prediction.py' first to train and save the model.")

model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# A single new customer, as raw data (same feature columns used in
# training -- i.e. the Telco dataset minus ID/leakage columns and Churn)
new_customer = pd.DataFrame([{
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 2,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "Yes",
    "Streaming Movies": "Yes",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 95.0,
    "Total Charges": 190.0,
}])

# One-hot encode the same way as training
new_encoded = pd.get_dummies(new_customer)

# Align columns: add any missing columns (fill with 0), drop extras,
# and put them in the SAME order the model was trained on.
new_encoded = new_encoded.reindex(columns=model_columns, fill_value=0)

prediction = model.predict(new_encoded)[0]
probability = model.predict_proba(new_encoded)[0][1]

print("Prediction:", "WILL CHURN" if prediction == 1 else "Will stay")
print(f"Churn probability: {probability:.1%}")
