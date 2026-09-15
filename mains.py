from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load saved files
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")
encoders = joblib.load("model/label_encoders.pkl")

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

app = FastAPI(
    title="Product Attribute Extraction API",
    version="1.0.0"
)

class Product(BaseModel):
    description: str


@app.post("/extract")
def extract(product: Product):

    # Convert text to TF-IDF features
    vec = vectorizer.transform([product.description])

    # Predict labels
    prediction = model.predict(vec)[0]

    # Try to get probabilities
    probabilities = None

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(vec)
        except Exception:
            probabilities = None

    result = {}
    confidence = {}

    for i, col in enumerate(attributes):

        # Convert encoded label back to text
        label = encoders[col].inverse_transform([prediction[i]])[0]
        result[col] = label

        # Confidence
        if probabilities is not None:
            confidence[col] = round(float(probabilities[i][0].max()), 4)
        else:
            confidence[col] = None

    return {
        "input": product.description,
        "attributes": result,
        "confidence": confidence
    }
