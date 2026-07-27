import os
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv("dataset/dresses.csv")

X = df["description"]

attributes = [
    "silhouette",
    "fabric",
    "neckline",
    "sleeve",
    "length",
    "embellishment",
    "color",
    "category"
]

encoders = {}

Y = pd.DataFrame()

for col in attributes:
    le = LabelEncoder()
    Y[col] = le.fit_transform(df[col])
    encoders[col] = le

vectorizer = TfidfVectorizer()

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    Y,
    test_size=0.2,
    random_state=42
)

model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=500
        )
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nEvaluation Results\n")

accuracies = []
f1_scores = []

for i, col in enumerate(attributes):

    acc = accuracy_score(y_test.iloc[:, i], pred[:, i])
    f1 = f1_score(y_test.iloc[:, i], pred[:, i], average="weighted")

    accuracies.append(acc)
    f1_scores.append(f1)

    print(f"{col:15} Accuracy: {acc:.2f}   F1 Score: {f1:.2f}")

print("\nOverall Results")
print("Average Accuracy :", round(sum(accuracies) / len(accuracies), 2))
print("Average F1 Score :", round(sum(f1_scores) / len(f1_scores), 2))

# Create model directory
os.makedirs("model", exist_ok=True)

# Save files
joblib.dump(model, "model/model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")
joblib.dump(encoders, "model/label_encoders.pkl")

print("Model saved successfully!")
