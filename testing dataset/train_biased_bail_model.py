"""Train a biased criminal bail decision model on a synthetic dataset.

This script expects biased_bail_dataset.csv in the same directory.
The synthetic data intentionally encodes racial bias so it can be used
for fairness and explainability experiments. It is NOT real case data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

RACES = ["White", "Black", "Hispanic", "Asian", "Other"]


def main():
    df = pd.read_csv("biased_bail_dataset.csv")

    X = df[["age", "prior_convictions", "charge_severity", "bail_amount_requested"]].copy()
    for r in RACES:
        X[f"race_{r}"] = (df["race"] == r).astype(int)
    X["has_job_Y"] = (df["has_job"] == "Y").astype(int)

    y = df["bail_granted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipe.fit(X_train, y_train)

    print("Train accuracy:", pipe.score(X_train, y_train))
    print("Test accuracy:", pipe.score(X_test, y_test))

    joblib.dump(pipe, "biased_bail_model.pkl")


if __name__ == "__main__":
    main()
