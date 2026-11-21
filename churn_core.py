"""
churn_core.py
=============
Shared data-loading and training code for the demos (app.py and
export_model.py). It follows exactly the same steps as
churn_prediction.py -- same cleaning, same dropped columns, same
one-hot encoding, same split and same Random Forest settings -- so the
demos show the same model the main script evaluates.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_FILE = "Telco_customer_churn.xlsx"

DROP_COLS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Value", "Churn Score", "Churn Reason", "CLTV",
]


def load_data(path=DATA_FILE):
    """Load the raw Excel file and return the cleaned feature table + 'Churn' (0/1)."""
    df = pd.read_excel(path)
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce").fillna(0)
    data = df.drop(columns=DROP_COLS)
    data["Churn"] = data.pop("Churn Label").map({"Yes": 1, "No": 0})
    return data


def train_model(data):
    """Encode, split, train the Random Forest and score it on the test set."""
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
        "roc_auc": roc_auc_score(y_test, y_prob),
        "recall": recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
    }
    importances = (
        pd.Series(rf.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    return {
        "model": rf,
        "columns": list(X.columns),
        "categorical_cols": categorical_cols,
        "metrics": metrics,
        "importances": importances,
        "X_test": X_test,
    }


def encode_customer(customer, columns):
    """One-hot encode raw customer rows and align them to the model's columns."""
    return pd.get_dummies(customer).reindex(columns=columns, fill_value=0)
