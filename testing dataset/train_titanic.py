"""Sample training script for Titanic survival prediction."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("/Users/ganesh/dev/Cephus-new/testing dataset/titanic.csv")

# Drop non-useful columns
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], errors="ignore")
df = df.dropna()

# Encode categoricals
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Split
X = df.drop(columns=["Survived"])
y = df["Survived"]

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

accuracy = accuracy_score(y, model.predict(X))
print(f"Training accuracy: {accuracy:.4f}")

# Save
joblib.dump(model, "/Users/ganesh/dev/Cephus-new/testing dataset/titanic_model.pkl")
print("Model saved.")
