"""Train a biased loan approval model on a synthetic dataset.

This script expects biased_loan_dataset.csv in the same directory.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib


def main():
    df = pd.read_csv("biased_loan_dataset.csv")

    X = df[["age", "income", "credit_score", "loan_amount"]].copy()
    X["gender_F"] = (df["gender"] == "F").astype(int)
    y = df["approved"]

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

    joblib.dump(pipe, "biased_loan_model.pkl")


if __name__ == "__main__":
    main()
