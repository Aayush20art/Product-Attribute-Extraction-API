# 👗 Product Attribute Extraction API

An ML-powered API that automatically extracts structured fashion attributes — **silhouette, fabric, neckline, sleeve, length, embellishment, color, and category** — from raw product descriptions.

Simply send a text description like *"A red silk dress with a V-neck and long sleeves"*, and the API returns clean, structured attribute labels with confidence scores — no manual tagging required.

---

## 🚀 Features

- 🔍 **Multi-attribute extraction** — predicts 8 fashion attributes from a single description in one API call
- 🌲 **Random Forest based multi-output classification** — one unified model handles all attributes together
- 📊 **Confidence scores** — returns prediction confidence for each attribute
- ⚡ **FastAPI backend** — lightweight, fast, and easy to integrate into any application
- 💾 **Pre-trained & reusable** — model, vectorizer, and encoders are saved and loaded instantly (no retraining needed)

---

## 🧠 How It Works

```
dresses.csv  ──▶  train.py  ──▶  model.pkl / vectorizer.pkl / label_encoders.pkl
                                              │
                                              ▼
                                       mains.py (FastAPI)
                                              │
                                              ▼
                    User sends description → API returns attributes + confidence
```

1. **Training (`train.py`)**
   - Loads labeled dress descriptions from `dataset/dresses.csv`
   - Converts text → numeric features using **TF-IDF Vectorization**
   - Encodes each attribute's labels using **LabelEncoder**
   - Trains a **`MultiOutputClassifier`** wrapped around a **`RandomForestClassifier`** (500 trees) to predict all 8 attributes simultaneously
   - Evaluates performance using **Accuracy** and **F1 Score** for each attribute
   - Saves the trained model, vectorizer, and encoders to the `model/` directory

2. **Inference API (`mains.py`)**
   - Loads the pre-trained model, vectorizer, and encoders on startup
   - Exposes a single `POST /extract` endpoint
   - Transforms input text using the same vectorizer used in training
   - Predicts attribute labels and decodes them back to human-readable text
   - Returns predictions along with per-attribute confidence scores

---

## 📁 Project Structure

```
├── dataset/
│   └── dresses.csv          # Training data (description + labeled attributes)
├── model/
│   ├── model.pkl             # Trained MultiOutputClassifier
│   ├── vectorizer.pkl         # Fitted TF-IDF vectorizer
│   └── label_encoders.pkl     # Label encoders for each attribute
├── train.py                  # Model training & evaluation script
├── mains.py                  # FastAPI inference server
└── README.md
```

---

## 🛠️ Tech Stack

- **Python**
- **scikit-learn** — TF-IDF, Label Encoding, Random Forest, MultiOutputClassifier
- **FastAPI** — REST API framework
- **Pydantic** — request validation
- **pandas** — data handling
- **joblib** — model persistence

---

## ⚙️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/Aayush20art/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install -r requirements.txt
```

### 1. Train the model

```bash
python train.py
```

This will train the model on `dataset/dresses.csv`, print evaluation metrics, and save the trained artifacts into the `model/` directory.

### 2. Run the API

```bash
uvicorn mains:app --reload
```

The API will be available at `http://127.0.0.1:8000`

---

## 📡 API Usage

### Endpoint: `POST /extract`

**Request body:**
```json
{
  "description": "A red silk dress with a V-neck and long sleeves"
}
```

**Response:**
```json
{
  "input": "A red silk dress with a V-neck and long sleeves",
  "attributes": {
    "silhouette": "A-line",
    "fabric": "Silk",
    "neckline": "V-neck",
    "sleeve": "Long sleeve",
    "length": "Midi",
    "embellishment": "None",
    "color": "Red",
    "category": "Dress"
  },
  "confidence": {
    "silhouette": 0.81,
    "fabric": 0.77,
    "neckline": 0.89,
    "sleeve": 0.85,
    "length": 0.72,
    "embellishment": 0.94,
    "color": 0.91,
    "category": 0.97
  }
}
```

You can also test it directly using the interactive Swagger docs at:
```
http://127.0.0.1:8000/docs
```

---

## 📊 Model Evaluation

The model is evaluated on a held-out 20% test split, reporting **Accuracy** and **weighted F1 Score** per attribute, along with overall average performance across all 8 attributes.

| Metric | Description |
|---|---|
| Accuracy | % of exact correct predictions per attribute |
| F1 Score (weighted) | Balances precision & recall, accounting for class imbalance |

---

## 🔮 Future Improvements

- Replace TF-IDF with transformer-based embeddings (e.g., Sentence-BERT) for richer text understanding
- Add batch prediction endpoint for processing multiple descriptions at once
- Deploy as a live demo on Streamlit Cloud / Render / HuggingFace Spaces
- Expand dataset to cover more fashion categories beyond dresses

---

## 👤 Author

**Aayush Sharma**
🔗 [GitHub](https://github.com/Aayush20art) | [LinkedIn](https://linkedin.com/in/aayush-sharma-b108a93b0)
