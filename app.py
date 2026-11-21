"""
app.py
======
Live demo: enter a customer's details and get a churn prediction.

Run locally:  streamlit run app.py

The model is trained the same way as in churn_prediction.py (same
cleaning, same dropped columns, same encoding, same Random Forest
settings) when the app starts, then cached -- so the demo works
straight from the repo without any saved .pkl files.
"""

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Value", "Churn Score", "Churn Reason", "CLTV",
]


@st.cache_data
def load_data():
    df = pd.read_excel("Telco_customer_churn.xlsx")
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)
    data = df.drop(columns=DROP_COLS)
    data["Churn"] = data.pop("Churn Label").map({"Yes": 1, "No": 0})
    return data


@st.cache_resource
def train_model(data):
    categorical_cols = data.select_dtypes(include=["object", "string"]).columns.tolist()
    encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    X = encoded.drop(columns=["Churn"])
    y = encoded["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_prob = rf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
    }
    importances = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    return rf, list(X.columns), categorical_cols, metrics, importances


data = load_data()
model, model_columns, categorical_cols, metrics, importances = train_model(data)


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
encoded = pd.get_dummies(customer).reindex(columns=model_columns, fill_value=0)
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
