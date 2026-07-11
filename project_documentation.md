# Product Attribute Extraction System — Documentation

## 1. What does this project do?

Imagine you have a text description of a dress, like:

> "A red silk dress with a V-neck and long sleeves"

This project **reads that sentence and automatically figures out** 8 details about the dress:

| Attribute | Example Value |
|---|---|
| Silhouette | A-line |
| Fabric | Silk |
| Neckline | V-neck |
| Sleeve | Long sleeve |
| Length | Midi |
| Embellishment | None |
| Color | Red |
| Category | Dress |

Instead of a human manually tagging thousands of product descriptions, the system does it automatically using Machine Learning.

---

## 2. Our Approach (`train.py`) — Teaching the Model

Think of this like teaching a student using flashcards.

**Step 1: Collect examples**
We use a dataset (`dresses.csv`) that already has descriptions *and* their correct labels (silhouette, fabric, etc.) — like an answer key.

**Step 2: Convert words into numbers**
Computers don't understand English directly, so we use a technique called **TF-IDF** (`TfidfVectorizer`). It converts each description into a set of numbers based on which words appear and how important/rare those words are. Think of it as turning a sentence into a "fingerprint" the computer can compare.

**Step 3: Convert labels into numbers**
Similarly, text labels like "Red", "Blue", "Silk" are converted into numbers using **Label Encoding**, since the model can only work with numbers, not words. We do this separately for each of the 8 attributes.

**Step 4: Train the model**
We use a **Random Forest** — think of it as a large group of decision-makers (500 small decision trees) who each vote on the answer, and the majority vote wins. This tends to be accurate and doesn't overfit easily.

Since we need to predict **8 things at once** (silhouette, fabric, neckline, etc.), we use a wrapper called **MultiOutputClassifier**, which basically trains one mini-model per attribute, all managed together.

**Step 5: Split data for a fair test**
We don't test the model on data it already saw. 80% of the data is used for training ("studying"), and 20% is kept aside as a "surprise test" to check how well it actually learned.

**Step 6: Save the trained model**
Once trained, we save 3 things to disk so they can be reused later without retraining:
- `model.pkl` → the trained brain
- `vectorizer.pkl` → the word-to-number converter
- `label_encoders.pkl` → the label-to-number converter (and back)

---

## 3. Evaluation — How do we know it's good?

After training, we test the model on the "surprise test" data (the 20% it never saw) and measure two things for **each attribute**:

- **Accuracy** → Out of 100 predictions, how many were exactly correct?
- **F1 Score** → A balanced score that accounts for cases where some labels are rare (e.g., very few dresses have "embellishment = sequins"). It punishes the model more fairly if it's only good at predicting common labels and bad at rare ones.

The script prints a report like:

```
Attribute        Accuracy   F1 Score
silhouette        0.85        0.84
fabric            0.79        0.78
neckline          0.88        0.87
...
```

Then it calculates the **average accuracy and F1 score across all 8 attributes**, giving one overall "health score" for the entire model.

*(In simple words: it's basically a report card with 8 subjects and one overall GPA.)*

---

## 4. API Implementation (`mains.py`) — Making it Usable

Training a model is only half the job — someone needs to actually **use** it. That's what this file does: it wraps the trained model inside a **web API** using **FastAPI**, so any app (website, mobile app, another program) can send a dress description and instantly get back the predicted attributes.

**How it works, step by step:**

1. **Load the saved brain** — When the API starts, it loads the 3 saved files (`model.pkl`, `vectorizer.pkl`, `label_encoders.pkl`) from step 2 above, so it doesn't need to retrain anything.

2. **Define what input looks like** — Using Pydantic (`Product` class), we tell the API: "Expect a JSON request with one field called `description`."

3. **Create an endpoint: `/extract`** — This is the "door" other apps knock on. They send a POST request like:
   ```json
   { "description": "A red silk dress with a V-neck" }
   ```

4. **Convert the input the same way as training** — The description is converted into numbers using the *same* TF-IDF vectorizer used during training (very important — otherwise the model won't understand the input).

5. **Predict** — The model predicts a number for each of the 8 attributes.

6. **Translate numbers back into words** — Using the saved label encoders, "3" becomes "Red," "1" becomes "V-neck," and so on — so the response is human-readable.

7. **Add confidence scores** — If the model supports it, the API also returns how *confident* it is about each prediction (e.g., 92% sure the color is red).

8. **Return the final answer** as a clean JSON response:
   ```json
   {
     "input": "A red silk dress with a V-neck",
     "attributes": {
       "silhouette": "A-line",
       "fabric": "Silk",
       "neckline": "V-neck",
       ...
     },
     "confidence": {
       "silhouette": 0.81,
       "fabric": 0.77,
       ...
     }
   }
   ```

---

## 5. Overall Flow (Big Picture)

```
dresses.csv  ──▶  train.py  ──▶  model.pkl / vectorizer.pkl / encoders.pkl
                                              │
                                              ▼
                                       mains.py (FastAPI)
                                              │
                                              ▼
                        User sends description → API returns attributes
```

**In one line:** `train.py` teaches the model using past examples and grades how well it learned; `mains.py` puts that trained model behind a simple API so anyone can send a dress description and instantly get structured, labeled results back.
