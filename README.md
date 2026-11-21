# Customer Churn Prediction — Full ML Project + Theory Guide

A complete, ready-to-run machine learning project that predicts whether a
telecom customer will cancel their subscription ("churn"). This README
teaches the **theory behind every step**, in the order you'll actually
use it — not as a separate textbook, but wired directly to the code in
`churn_prediction.py`.

### 🔴 Live demo: [customer-churn-prediction-gamma-eight.vercel.app](https://customer-churn-prediction-gamma-eight.vercel.app)

Enter a customer's details and get their churn probability instantly,
adjust the decision threshold, and see what drives churn. The trained
Random Forest runs entirely in the browser (see
[Live demo: how it works](#live-demo-how-it-works)).

---

## 1. What is Machine Learning, really?

Traditional programming: you write explicit rules (`if monthly_charges >
100: flag_risky_customer`). Machine Learning: you show the computer many
**examples** (past customers, and whether they churned or not), and it
**learns the rules itself** by finding statistical patterns in the data.

This project is **Supervised Learning** — "supervised" because every
training example has a known correct answer (`Churn: Yes/No`), and the
model's whole job is to learn the mapping from customer features →
that answer, well enough to predict it for customers we haven't seen yet.

Supervised learning splits into two types:
- **Regression** → predicting a number (e.g. house price, temperature)
- **Classification** → predicting a category (e.g. Churn: Yes/No)

This project is **binary classification** (only two possible classes).

---

## 2. The problem we're solving

**Business question:** Which customers are likely to cancel their
subscription, so the company can proactively offer them a retention deal?

This is one of the most commonly asked-about ML projects in interviews
because it's realistic, uses tabular/structured data (not images/audio),
and directly ties to business value — perfect for data analyst / data
scientist roles.

---

## 3. The dataset

This project uses the **real IBM Telco Customer Churn dataset**
(`Telco_customer_churn.xlsx`) — 7,043 actual telecom customers, ~26.5%
of whom churned. It's one of the most widely used public datasets for
practicing churn prediction, so this project transfers directly to any
similar real-world dataset or interview take-home exercise.

Raw columns include demographics, account details, subscribed services,
billing info, and the churn outcome — but it also comes with a **very
instructive trap**: three columns (`Churn Score`, `Churn Reason`,
`Churn Value`) leak the answer or duplicate the target outright, and
`churn_prediction.py` explicitly drops them with an explanation of why
(see Section 5 below — this is a real skill interviewers test for).

Key columns used as features:

| Column | Meaning |
|---|---|
| Tenure Months | months as a customer |
| Contract | Month-to-month / One year / Two year |
| Internet Service | DSL / Fiber optic / No |
| Monthly Charges, Total Charges | billing amounts |
| Online Security, Tech Support, etc. | add-on services (Yes/No) |
| Churn Label | **target variable** — did they leave? |

Dropped columns (identifiers, constants, or leakage — explained in the
code): `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`,
`Lat Long`, `Latitude`, `Longitude`, `Churn Value`, `Churn Score`,
`Churn Reason`, `CLTV`.

---

## 4. Exploratory Data Analysis (EDA)

**Why it exists:** Before touching any model, you must understand your
data — otherwise you can't tell if a weird result is a real insight or a
bug. EDA answers questions like:
- Is the data balanced? (Here: ~26.5% churn, ~73.5% no-churn — **imbalanced**)
- Are there missing values?
- Which features visibly relate to the target?

**Key takeaway from this dataset:** churn rate is much higher for
month-to-month contracts than annual ones — a real pattern the model
should discover on its own.

---

## 5. Feature Engineering & Encoding

Models only understand numbers. Text categories like `"Month-to-month"`
must be converted — this is **encoding**.

- **One-Hot Encoding** (used here via `pd.get_dummies`): turns one
  categorical column into several 0/1 columns, one per category. Used
  for **nominal** data (no natural order) — e.g., `Internet Service` has
  no "greater than" relationship between DSL and Fiber.
- **Label Encoding**: assigns each category a single number (0, 1, 2…).
  Fine for **ordinal** data (has order) or tree models, but risky for
  linear models since it implies false ordering/magnitude.

We drop `CustomerID`, `Count`, `Country`, `State`, `City`, `Zip Code`,
`Lat Long`, `Latitude`, `Longitude` — pure identifiers or constants with
no predictive signal. We also drop three **leakage columns** unique to
this real dataset: `Churn Value` (a literal duplicate of the target),
`Churn Score` (IBM's own pre-computed churn-risk score — using it as a
feature would let the model just copy another model's answer instead of
learning real patterns), and `Churn Reason` (only filled in *after* a
customer already churned, e.g. "Competitor made better offer" — info
that can't exist yet for a customer you're trying to predict for).
Including any of these would inflate your metrics in testing while
being completely useless (or actively wrong) in production. Including
ID-like columns is a related, classic beginner mistake that causes
**overfitting** (the model "memorizes" rows instead of learning
transferable patterns).

---

## 6. Train/Test Split — the most important concept in ML

We split data into:
- **Training set** (80%) — used to teach the model
- **Test set** (20%) — held back, used ONLY to evaluate

**Why this matters so much:** A model can reach 99% accuracy on data it
already memorized — that tells you nothing about real-world performance.
The test set simulates genuinely new, unseen customers. Any ML result
you see WITHOUT a train/test split (or cross-validation) should be
treated with suspicion.

**Overfitting** = model learns the training data too specifically
(including its noise/quirks) and fails to generalize.
**Underfitting** = model is too simple to capture real patterns, and
performs poorly even on training data.
Good models sit in between — this balance is called the
**bias-variance tradeoff**.

`stratify=y` in the code keeps the churn/no-churn ratio identical in
both train and test sets — crucial for imbalanced problems like this one.

---

## 7. Feature Scaling

`Tenure Months` ranges 0–72, `Monthly Charges` ranges ~18–120. Models like
Logistic Regression compute weighted sums of features — without scaling,
large-range features would dominate purely due to scale, not real
importance.

**StandardScaler** transforms each feature to mean=0, std=1.

**Critical detail:** the scaler is `fit` only on training data, then
`transform`-only on test data. Fitting on the full dataset first would
leak test-set statistics into training — called **data leakage**, and it
silently makes your evaluation look better than reality.

Tree-based models (Random Forest) don't need scaling — they split on
thresholds, not distances — but it doesn't hurt to have it available.

---

## 8. Model 1 — Logistic Regression (the baseline)

Despite the name, this is a **classification** algorithm. It computes:

```
z = w1*x1 + w2*x2 + ... + b        (weighted sum, like linear regression)
probability = 1 / (1 + e^-z)        (sigmoid function — squashes to 0-1)
```

If probability > 0.5 → predict "will churn".

**Why start here:** it's simple, fast, and **interpretable** — each
feature gets a coefficient (weight) telling you its direction and
strength of effect on churn. Always build this first as your baseline:
if a more complex model can't beat it, the complexity isn't earning its
keep.

---

## 9. Model 2 — Random Forest (ensemble learning)

A single **Decision Tree** asks a sequence of yes/no questions ("is
tenure < 6? is contract month-to-month?") to reach a prediction. Trees
are easy to overfit — a deep enough tree can carve out a rule for every
individual training row.

A **Random Forest** builds many trees (here, 200), each trained on a
random subset of rows AND features, then averages their votes. This is
called **bagging** (bootstrap aggregating) — many "weak," diverse
learners combine into one strong, stable model. This generally beats a
single tree and captures **non-linear relationships** and **feature
interactions** automatically (e.g. "short tenure AND month-to-month"
might matter more together than either alone) — something plain
Logistic Regression can't do without manually engineering those
interaction features.

---

## 10. Handling Imbalanced Classes

Only ~26.5% of customers churn. A lazy model could predict "No churn" for
everyone and still be ~73.5% "accurate" — while being completely useless.

**Fix used here:** `class_weight="balanced"` — tells the model to
penalize mistakes on the minority class (churners) more heavily during
training, forcing it to actually learn to catch them, at some cost to
precision. This is a genuine **business tradeoff**, not just a technical
knob — see the next section for why.

---

## 11. Evaluation Metrics — why "accuracy" isn't enough

| Metric | Question it answers | When it matters most |
|---|---|---|
| **Accuracy** | Overall, how often is the model right? | Only meaningful for balanced classes |
| **Precision** | Of everyone flagged as "will churn," how many really will? | Matters when acting on a false alarm is costly (wasted discount) |
| **Recall** | Of everyone who really churns, how many did we catch? | Matters when missing a churner is costly (lost customer) |
| **F1 Score** | Balance of precision & recall | When both matter and classes are imbalanced |
| **ROC-AUC** | How well the model ranks churners above non-churners across all thresholds | Good overall model-quality measure, independent of chosen cutoff |

**Confusion Matrix** — the raw counts behind every metric above:

```
                 Predicted: No    Predicted: Yes
Actual: No       True Negative     False Positive
Actual: Yes      False Negative    True Positive
```

In this project, since **losing a customer costs more than one wasted
retention offer**, recall usually matters more than precision — which is
exactly why `class_weight="balanced"` is the right call here.

---

## 12. Cross-Validation

A single train/test split gives ONE score — but what if that particular
split was unusually easy or hard by chance? **K-Fold Cross-Validation**
splits the **training set** into K parts (folds, here K=5), trains K times
(each time holding out a different fold to validate on), and averages
the results. The test set is never touched here — it stays reserved for
the final, honest evaluation. This gives a far more **reliable** estimate of real-world
performance, and its standard deviation tells you how *stable* that
performance is across different slices of data.

---

## 13. Feature Importance — making the model explainable

A prediction alone ("this customer will churn") isn't very actionable. A
business needs to know **why**, so it can act. Random Forest gives a
built-in importance score per feature (based on how much each feature
reduces uncertainty across all trees). In this project, `Tenure Months`,
`Contract` (Two year), `Total Charges`, and `Dependents` come out as the
strongest drivers — a genuinely useful business insight, not just a
prediction.

---

## 14. Saving & Reusing the Model

Training is slow; predicting with an already-trained model is instant.
In real systems, you **train once**, **save ("serialize") the model to
disk** (`joblib.dump`), and **load it later** inside an app/API to score
new customers in real time — no retraining needed. `predict_new_customer.py`
demonstrates exactly this: load the saved model, feed it one new
customer's raw data (preprocessed the *exact* same way as training data),
and get an instant prediction + probability.

---

## How to run this project

```bash
pip install -r requirements.txt

python churn_prediction.py       # trains, evaluates, saves plots + model (run this first)
python predict_new_customer.py   # demo: predict on one new customer
```

Run both scripts from the project folder. The model files (`*.pkl`) are
not committed to git — `churn_prediction.py` regenerates them.

Input: `Telco_customer_churn.xlsx` — the real dataset (7,043 customers)

Outputs produced:
- `eda_overview.png` — churn distribution + churn rate by contract type
- `feature_importance.png` — top 10 drivers of churn
- `roc_curve.png` — model comparison
- `churn_model.pkl`, `model_columns.pkl` — saved Random Forest + the column order it expects

## Live demo: how it works

The hosted demo is a static website on Vercel. There's no Python server:
the trained model runs in the visitor's browser.

- **`export_model.py`** trains the same Random Forest (via `churn_core.py`)
  and writes every tree to `web/model.json`. Each tree is just a list of
  "is feature ≤ threshold? go left : go right" questions ending in a
  churn probability, so the browser can replay them and average the 200
  results. Before writing the file, the script checks that this replay
  gives the same probability as scikit-learn for every test customer.
- **`web/index.html`** is the page: it loads `model.json`, encodes the
  form exactly like training (one-hot columns), and predicts on every change.
- Vercel serves the `web/` folder and redeploys automatically on every
  push to `main`.

To update the demo after changing the model:

```bash
python export_model.py           # rewrites web/model.json (with the parity check)
git add web/model.json && git commit -m "Update demo model" && git push
```

To preview it locally: `python -m http.server 8600 --directory web`, then
open http://localhost:8600.

There's also a Streamlit version of the demo (`app.py`, all Python) for
running locally: `streamlit run app.py`.

## Results summary (this run, on real data)

- Logistic Regression: ROC-AUC ≈ 0.849, Recall ≈ 0.78, Precision ≈ 0.51
- Random Forest: ROC-AUC ≈ 0.853, Recall ≈ 0.80, Precision ≈ 0.53
- 5-fold CV ROC-AUC (training set): 0.859 ± 0.011 (stable across folds — good sign)
- Threshold tuning: with illustrative costs ($500 per missed churner, $50
  per wasted offer), a 0.2 cutoff costs $35,000 on the test set vs. $50,450
  at the default 0.5 — recall rises to 0.96, precision drops to 0.39
- Top churn drivers: **Tenure Months, Contract (Two year), Total Charges,
  Dependents, Internet Service (Fiber optic), Monthly Charges, Payment
  Method (Electronic check)**

These numbers are noticeably stronger than a synthetic dataset would give
you — real data has real, learnable structure. Notice precision is lower
than recall here: with `class_weight="balanced"`, the model is tuned to
catch more real churners (higher recall) at the cost of more false
alarms (lower precision) — a deliberate business tradeoff explained in
Section 11, not a flaw.

## Next steps to extend this project (great for a resume/portfolio)

1. **Hyperparameter tuning** — use `GridSearchCV` or `RandomizedSearchCV`
   to systematically search for the best `max_depth`, `n_estimators`, etc.
   instead of guessing values by hand.
2. **Try Gradient Boosting** (XGBoost / LightGBM) — usually outperforms
   Random Forest on tabular data like this.
3. **SHAP values** — a more rigorous explainability technique than basic
   feature importance, showing exactly how each feature pushed an
   *individual* prediction up or down.
4. **Better threshold tuning** — the script already sweeps cutoffs with
   illustrative costs; next, choose the threshold on a separate validation
   set (not the test set) and use real business costs.
5. **Wrap it in a Flask/FastAPI endpoint** — turn `predict_new_customer.py`
   into a real API a front-end could call. (Great pairing with your
   Django experience.)
6. ~~**Deploy a live demo**~~ — done: see the
   [live demo](https://customer-churn-prediction-gamma-eight.vercel.app).
