"""
export_model.py
===============
Exports the trained Random Forest to web/model.json so the live demo
(web/index.html) can make predictions directly in the browser -- no
Python server needed, which is what lets it run as a static site on
Vercel.

Run:  python export_model.py      (re-run whenever the model changes)

HOW A RANDOM FOREST CAN RUN IN JAVASCRIPT
Each of the 200 trees is just a set of yes/no questions: "is feature F
<= threshold T? go left, otherwise go right", until a leaf that holds a
churn probability. The forest's prediction is the AVERAGE of the leaf
probabilities across all trees. So we only need to export, per tree,
the node arrays scikit-learn already stores (left child, right child,
feature, threshold, leaf value) -- and replay the same walk in JS.

Before writing the file, this script re-implements that walk in plain
Python and checks it against scikit-learn's own predict_proba on every
test customer, so the browser version is proven to match.
"""

import json
import os

import numpy as np
import pandas as pd

import churn_core

OUT_FILE = os.path.join("web", "model.json")


def export_tree(tree):
    t = tree.tree_
    # value has shape (n_nodes, 1, n_classes); normalise to P(churn) per node.
    v = t.value[:, 0, :]
    p_churn = v[:, 1] / v.sum(axis=1)
    is_leaf = t.children_left == -1
    return {
        "l": t.children_left.tolist(),
        "r": t.children_right.tolist(),
        "f": t.feature.tolist(),
        # Thresholds are kept at full precision -- rounding could flip a split.
        "t": [float(x) if not leaf else 0 for x, leaf in zip(t.threshold, is_leaf)],
        "v": [round(float(p), 6) if leaf else 0 for p, leaf in zip(p_churn, is_leaf)],
    }


def feature_spec(columns, categorical_cols):
    """How to build each model column from a raw customer record."""
    spec = []
    for col in columns:
        source = next((c for c in categorical_cols if col.startswith(c + "_")), None)
        if source is None:
            spec.append({"name": col, "source": col, "value": None})  # numeric
        else:
            spec.append({"name": col, "source": source, "value": col[len(source) + 1:]})
    return spec


def encode_like_js(record, spec):
    row = []
    for s in spec:
        if s["value"] is None:
            row.append(float(record[s["source"]]))
        else:
            row.append(1.0 if record[s["source"]] == s["value"] else 0.0)
    return row


def predict_like_js(row, trees):
    # scikit-learn first rounds each input value to float32, then compares it
    # with the (float64) threshold. JS does the same with Math.fround(x) <= t.
    # (.tolist() turns the float32 values back into plain Python floats, so the
    # comparison happens in full precision, exactly like scikit-learn.)
    x = np.asarray(row, dtype=np.float32).tolist()
    total = 0.0
    for tr in trees:
        node = 0
        while tr["l"][node] != -1:
            node = tr["l"][node] if x[tr["f"][node]] <= tr["t"][node] else tr["r"][node]
        total += tr["v"][node]
    return total / len(trees)


def main():
    data = churn_core.load_data()
    trained = churn_core.train_model(data)
    rf, columns = trained["model"], trained["columns"]
    categorical_cols = trained["categorical_cols"]

    trees = [export_tree(est) for est in rf.estimators_]
    spec = feature_spec(columns, categorical_cols)

    # ---- Parity check: JSON logic vs scikit-learn on every test customer.
    X_test = trained["X_test"]
    raw_test = data.loc[X_test.index]
    sk_probs = rf.predict_proba(X_test)[:, 1]
    max_diff = 0.0
    for (_, record), (_, enc_row), sk_p in zip(raw_test.iterrows(), X_test.iterrows(), sk_probs):
        row = encode_like_js(record, spec)
        assert np.allclose(row, enc_row.astype(float).values), "encoding mismatch"
        max_diff = max(max_diff, abs(predict_like_js(row, trees) - sk_p))
    print(f"Parity check on {len(X_test)} test customers: max probability difference = {max_diff:.2e}")
    assert max_diff < 1e-4, "exported model does not match scikit-learn"

    options = {c: sorted(data[c].dropna().unique().tolist()) for c in categorical_cols}
    payload = {
        "n_customers": int(len(data)),
        "churn_rate": round(float(data["Churn"].mean()), 4),
        "metrics": {k: round(float(v), 3) for k, v in trained["metrics"].items()},
        "importances": [
            {"feature": k, "importance": round(float(v), 4)}
            for k, v in trained["importances"].items()
        ],
        "options": options,
        "features": spec,
        "trees": trees,
    }
    os.makedirs("web", exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote {OUT_FILE} ({os.path.getsize(OUT_FILE) / 1e6:.1f} MB, {len(trees)} trees)")


if __name__ == "__main__":
    main()
