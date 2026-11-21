"""
churn_prediction.py
====================
CUSTOMER CHURN PREDICTION -- Full ML Pipeline
Now using the REAL IBM Telco Customer Churn dataset (Telco_customer_churn.xlsx)
-- 7,043 real customers, 26.5% churn rate.

Run: python churn_prediction.py
(Reads the Excel file directly; run it from the project folder.)

Every section below has a THEORY block above the code, explaining not just
WHAT the code does but WHY that step exists in any ML project.
Read README.md alongside this file for the full deep-dive theory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import joblib

# ======================================================================
# STEP 1: LOAD THE DATA
# ======================================================================
df = pd.read_excel("Telco_customer_churn.xlsx")
print("Dataset shape:", df.shape)
print(df.head())

# ======================================================================
# STEP 2: CLEANING -- fix the "Total Charges" column
# ----------------------------------------------------------------------
# THEORY: Real data is messy. Here, "Total Charges" is stored as TEXT
# instead of a number, because 11 brand-new customers (Tenure = 0) have
# a BLANK value instead of 0 -- Excel/pandas reads the whole column as
# text/object type the moment even one entry isn't numeric. We convert
# it to numeric and fill those blanks with 0, since a customer with 0
# months of tenure has logically been charged $0 total so far.
# ======================================================================
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df["Total Charges"] = df["Total Charges"].fillna(0)
print(f"\nFixed 'Total Charges' -- {df['Total Charges'].isnull().sum()} nulls remain")

# ======================================================================
# STEP 3: REMOVE LEAKAGE & IDENTIFIER COLUMNS -- important real-world lesson
# ----------------------------------------------------------------------
# THEORY (DATA LEAKAGE): This raw dataset actually contains columns that
# would let a model "cheat" by seeing information that WOULDN'T exist
# yet at real prediction time:
#   - "Churn Value"  -> literally a 0/1 duplicate of the target itself.
#   - "Churn Score"  -> IBM's own pre-computed churn-risk score. Using it
#     as a FEATURE means our model would just be learning to copy
#     another model's already-known answer -- not learning real patterns.
#   - "Churn Reason" -> only filled in for customers who ALREADY churned
#     (e.g. "Competitor made better offer") -- this is explaining an
#     event that has already happened, so it can never be known in
#     advance for a customer we're trying to predict for.
# We also drop pure IDENTIFIERS with no predictive meaning (CustomerID),
# and columns that are CONSTANT across the whole dataset (Country, State
# are both single-valued here, so they carry zero signal). "City" has
# 1,129 unique values -- too high-cardinality to one-hot encode cleanly
# for this project, so we drop it too (a good "next step" extension is
# grouping cities into regions instead of dropping entirely).
# Zip Code / Lat Long / Latitude / Longitude are location detail at the
# same too-fine granularity as City, so they go as well.
# "CLTV" (Customer Lifetime Value) isn't strictly leakage -- it's another
# PRE-COMPUTED business score, like Churn Score. We drop it so the model
# learns only from raw customer attributes, not from someone else's model.
# ======================================================================
leakage_and_id_cols = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude",
    "Churn Value", "Churn Score", "Churn Reason", "CLTV",
]
data = df.drop(columns=leakage_and_id_cols).copy()
print(f"\nDropped {len(leakage_and_id_cols)} leakage/identifier/constant columns")
print("Remaining columns:", data.columns.tolist())

# ======================================================================
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------
# THEORY: EDA means looking at your data BEFORE modeling -- checking
# distributions, class balance, and relationships between features and
# the target. This is where you find real business insight, and where
# you'd catch a problem like the leakage columns above if you hadn't
# already known to look for them.
# ======================================================================
print("\nChurn distribution:\n", data["Churn Label"].value_counts(normalize=True))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
data["Churn Label"].value_counts().plot(kind="bar", ax=axes[0], title="Churn Count")
data.groupby("Contract")["Churn Label"].apply(
    lambda x: (x == "Yes").mean()
).plot(kind="bar", ax=axes[1], title="Churn Rate by Contract Type")
plt.tight_layout()
plt.savefig("eda_overview.png")
print("Saved eda_overview.png")

# ======================================================================
# STEP 5: FEATURE ENGINEERING & ENCODING
# ----------------------------------------------------------------------
# THEORY: ML models only understand numbers. Text columns like
# "Contract" (Month-to-month/One year/Two year) must be ENCODED into
# numeric form. We use ONE-HOT ENCODING here (via pd.get_dummies) --
# creating a separate 0/1 column per category -- because these are
# NOMINAL categories with no natural order (e.g. "DSL" isn't "less than"
# "Fiber optic"). drop_first=True avoids the "dummy variable trap"
# (redundant, perfectly-correlated columns).
# ======================================================================
data["Churn"] = data["Churn Label"].map({"Yes": 1, "No": 0})
data = data.drop(columns=["Churn Label"])

# "object" catches text columns in pandas 2; pandas 3 stores text as the
# new "str" dtype instead, so we ask for both to work on either version.
categorical_cols = data.select_dtypes(include=["object", "string"]).columns.tolist()
print("\nCategorical columns to encode:", categorical_cols)

data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
print("Shape after encoding:", data_encoded.shape)

# ======================================================================
# STEP 6: TRAIN/TEST SPLIT
# ----------------------------------------------------------------------
# THEORY: This is THE most important concept in ML. We split data into:
#   - TRAINING set (used to teach the model patterns)
#   - TEST set (held back, used ONLY to check how well it generalizes)
#
# A model can achieve near-100% accuracy on data it has already seen
# just by memorizing it (OVERFITTING) -- but that tells you NOTHING
# about how it performs on new, unseen customers. The test set simulates
# "future, unseen data" so evaluation is honest.
#
# stratify=y ensures both train and test sets keep the same churn/no-churn
# RATIO as the full dataset -- important since only ~26.5% of customers
# churned here.
# ======================================================================
X = data_encoded.drop(columns=["Churn"])
y = data_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ======================================================================
# STEP 7: FEATURE SCALING
# ----------------------------------------------------------------------
# THEORY: Features are on very different scales -- "Tenure Months" ranges
# 0-72, "Monthly Charges" ranges ~18-120, "Total Charges" can be in the
# thousands. Models like Logistic Regression (which use weighted sums)
# treat large-scale features as "more important" purely due to scale, not
# actual signal. StandardScaler fixes this: mean=0, std=1 per feature.
#
# We fit the scaler ONLY on training data, then apply it to test data --
# fitting on the full dataset first would leak test-set statistics into
# training (DATA LEAKAGE, same concept as Step 3, different flavor).
# Tree models (Random Forest) don't need scaling since they split on
# thresholds, not distances.
# ======================================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================================================================
# STEP 8: MODEL 1 -- LOGISTIC REGRESSION (the baseline)
# ----------------------------------------------------------------------
# THEORY: Logistic Regression computes a weighted sum of features, then
# squashes it through a sigmoid function into a probability (0 to 1):
#   sigmoid(z) = 1 / (1 + e^-z)
# Simple, fast, and highly INTERPRETABLE -- always build this as your
# baseline before trying anything fancier.
#
# class_weight="balanced": only ~26.5% of customers churn here, so
# without this, the model would lean toward always predicting "No churn"
# to rack up accuracy. This penalizes mistakes on the minority (churn)
# class more heavily, trading some precision for much better recall.
# ======================================================================
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# ======================================================================
# STEP 9: MODEL 2 -- RANDOM FOREST (an ensemble model)
# ----------------------------------------------------------------------
# THEORY: Builds many decision trees (here, 200), each trained on a
# random subset of rows/features, then averages their votes ("bagging").
# A single tree easily overfits; a forest of diverse trees smooths this
# out. Captures non-linear relationships and feature INTERACTIONS
# automatically (e.g. "short tenure AND month-to-month" combined) --
# something plain Logistic Regression can't do without manual work.
# ======================================================================
rf = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
)
rf.fit(X_train, y_train)  # tree models don't need scaled data
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# ======================================================================
# STEP 10: EVALUATION METRICS
# ----------------------------------------------------------------------
# THEORY: Accuracy alone is MISLEADING for imbalanced problems like this
# (only ~26.5% churn). A model predicting "No churn" for everyone would
# still get ~73% accuracy while being useless. So we also look at:
#   - PRECISION: of everyone predicted to churn, how many really did?
#   - RECALL: of everyone who really churned, how many did we catch?
#   - F1: balance of precision & recall.
#   - CONFUSION MATRIX: the raw TP/TN/FP/FN counts behind every metric.
#   - ROC-AUC: how well the model RANKS churners above non-churners
#     across ALL thresholds, not just the default 0.5 cutoff.
# ======================================================================

def evaluate(name, y_true, y_pred, y_prob):
    print(f"\n===== {name} =====")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.3f}")
    print(f"Precision: {precision_score(y_true, y_pred):.3f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.3f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.3f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.3f}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nFull report:\n", classification_report(y_true, y_pred))

evaluate("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
evaluate("Random Forest", y_test, y_pred_rf, y_prob_rf)

# ======================================================================
# STEP 11: CROSS-VALIDATION (checking the score isn't a fluke)
# ----------------------------------------------------------------------
# THEORY: A single train/test split gives you ONE score -- what if that
# split happened to be easy or hard by chance? K-FOLD CROSS-VALIDATION
# splits training data into K folds, trains K times (holding out a
# different fold each time), and averages the scores -- a far more
# reliable performance estimate, and its spread shows stability.
#
# We run it on the TRAINING set only. The test set must stay untouched
# until final evaluation -- otherwise it stops being "unseen" data.
# ======================================================================
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring="roc_auc")
print(f"\n5-Fold CV ROC-AUC scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f}  Std: {cv_scores.std():.3f}")

# ======================================================================
# STEP 12: FEATURE IMPORTANCE (explainability)
# ----------------------------------------------------------------------
# THEORY: A prediction alone isn't useful to a business without knowing
# WHY, so it can act (e.g. targeted retention offers). Random Forest
# gives a built-in importance score per feature based on how much it
# reduces uncertainty across all trees -- turning the model from a black
# box into an actionable business insight.
# ======================================================================
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False).head(10)
print("\nTop 10 most important features:\n", importances)

plt.figure(figsize=(8, 5))
importances.sort_values().plot(kind="barh", title="Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved feature_importance.png")

plt.figure(figsize=(6, 6))
for name, y_prob in [("Logistic Regression", y_prob_lr), ("Random Forest", y_prob_rf)]:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png")
print("Saved roc_curve.png")

# ======================================================================
# STEP 13: THRESHOLD TUNING (matching the cutoff to business cost)
# ----------------------------------------------------------------------
# THEORY: predict() uses a fixed 0.5 probability cutoff, but nothing
# makes 0.5 special. Lowering it flags more customers (higher recall,
# lower precision); raising it does the opposite. The right cutoff
# depends on what each mistake COSTS the business.
#
# Illustrative costs (swap in real numbers if you have them):
#   - Missing a churner (false negative) loses ~$500 of future revenue.
#   - A wasted retention offer (false positive) costs ~$50.
# We pick the cutoff with the lowest total cost. NOTE: in a stricter
# setup you'd choose this on a separate validation set, not the test set.
# ======================================================================
COST_FN, COST_FP = 500, 50
print("\nThreshold | Precision | Recall | Total cost")
threshold_costs = {}
for t in np.arange(0.1, 0.91, 0.1):
    y_pred_t = (y_prob_rf >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    cost = fn * COST_FN + fp * COST_FP
    threshold_costs[round(t, 1)] = cost
    print(f"   {t:.1f}    |   {precision_score(y_test, y_pred_t):.3f}   | "
          f"{recall_score(y_test, y_pred_t):.3f}  | ${cost:,}")
best_t = min(threshold_costs, key=threshold_costs.get)
print(f"Lowest-cost threshold: {best_t} (vs. default 0.5)")

# ======================================================================
# STEP 14: SAVE THE MODEL (so it can be reused without retraining)
# ----------------------------------------------------------------------
# We save the Random Forest (the stronger model here). It was trained on
# UNSCALED data, so no scaler needs to be saved -- only the model and
# the exact column list/order it expects.
# ======================================================================
joblib.dump(rf, "churn_model.pkl")
joblib.dump(list(X.columns), "model_columns.pkl")
print("\nSaved churn_model.pkl, model_columns.pkl")

print("\nPipeline complete.")
