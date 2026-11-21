"""
app.py
======
Live demo: enter a customer's details and get a churn prediction.

Run locally:  streamlit run app.py

The model is trained by churn_core.py (the same steps as
churn_prediction.py) when the app starts, then cached -- so the demo
works straight from the repo without any saved .pkl files.

The hosted demo is the static web/ version (see export_model.py); this
Streamlit app is the local, all-Python alternative.
"""

import altair as alt
import pandas as pd
import streamlit as st

import churn_core

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

data = st.cache_data(churn_core.load_data)()
trained = st.cache_resource(churn_core.train_model)(data)
model, model_columns = trained["model"], trained["columns"]
importances = trained["importances"]
metrics = {
    "ROC-AUC": trained["metrics"]["roc_auc"],
    "Recall": trained["metrics"]["recall"],
    "Precision": trained["metrics"]["precision"],
}


def options(col):
    return sorted(data[col].dropna().unique().tolist())


# ---------------------------------------------------------------- header
st.title("📉 Customer Churn Predictor")
st.write(
    "Enter a telecom customer's details to see how likely they are to cancel. "
    "The model is a Random Forest trained on the real IBM Telco dataset "
    f"({len(data):,} customers, {data['Churn'].mean():.1%} churned)."
)

m1, m2, m3 = st.columns(3)
m1.metric("Test ROC-AUC", f"{metrics['ROC-AUC']:.3f}")
m2.metric("Test recall", f"{metrics['Recall']:.2f}")
m3.metric("Test precision", f"{metrics['Precision']:.2f}")

# ---------------------------------------------------------------- inputs
st.subheader("Customer details")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Account**")
    tenure = st.slider("Tenure (months)", 0, 72, 2)
    contract = st.selectbox("Contract", options("Contract"))
    payment = st.selectbox("Payment method", options("Payment Method"), index=options("Payment Method").index("Electronic check"))
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    monthly = st.number_input("Monthly charges ($)", 0.0, 200.0, 95.0, step=5.0)
    total = st.number_input(
        "Total charges ($)", 0.0, 10000.0, float(round(monthly * tenure, 2)), step=50.0,
        help="Defaults to monthly charges × tenure.",
    )

with c2:
    st.markdown("**Services**")
    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
    multiple = "No phone service" if phone == "No" else st.radio("Multiple lines", ["No", "Yes"], horizontal=True)
    internet = st.selectbox("Internet service", options("Internet Service"), index=options("Internet Service").index("Fiber optic"))
    addons = {}
    for col in ["Online Security", "Online Backup", "Device Protection",
                "Tech Support", "Streaming TV", "Streaming Movies"]:
        if internet == "No":
            addons[col] = "No internet service"
        else:
            default = 1 if col.startswith("Streaming") else 0
            addons[col] = st.radio(col, ["No", "Yes"], index=default, horizontal=True)

with c3:
    st.markdown("**Demographics**")
    gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
    senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)

customer = pd.DataFrame([{
    "Gender": gender, "Senior Citizen": senior, "Partner": partner,
    "Dependents": dependents, "Tenure Months": tenure, "Phone Service": phone,
    "Multiple Lines": multiple, "Internet Service": internet, **addons,
    "Contract": contract, "Paperless Billing": paperless,
    "Payment Method": payment, "Monthly Charges": monthly, "Total Charges": total,
}])

# Same encoding as training, aligned to the model's exact columns.
encoded = churn_core.encode_customer(customer, model_columns)
prob = model.predict_proba(encoded)[0][1]

# ---------------------------------------------------------------- result
st.divider()
st.subheader("Prediction")
threshold = st.slider(
    "Decision threshold", 0.1, 0.9, 0.5, 0.05,
    help="Customers at or above this churn probability are flagged. Lower it to catch "
         "more churners (higher recall) at the cost of more false alarms.",
)
r1, r2 = st.columns([1, 2])
with r1:
    st.metric("Churn probability", f"{prob:.1%}")
    if prob >= threshold:
        st.error("⚠️ Likely to churn — consider a retention offer.")
    else:
        st.success("✅ Likely to stay.")
    st.progress(float(prob))
with r2:
    st.markdown("**What drives churn overall** (top 10 features in the model)")
    imp_df = importances.rename("Importance").rename_axis("Feature").reset_index()
    st.altair_chart(
        alt.Chart(imp_df)
        .mark_bar()
        .encode(
            x=alt.X("Importance:Q"),
            y=alt.Y("Feature:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")],
        )
        .properties(height=320),
        width="stretch",
    )

st.caption(
    "Source code and full explanation: "
    "[github.com/Shrutimaheta/customer-churn-prediction]"
    "(https://github.com/Shrutimaheta/customer-churn-prediction)"
)
