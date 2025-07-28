# ✈️ ATIS Intent Classifier — BERT + Flask

This project is an **intent classification service** built with **DistilBERT** embeddings and a **Logistic Regression** classifier, wrapped in a simple **Flask API**.

It uses the **ATIS dataset** as an example, a classic benchmark for flight-related natural language understanding.

---

## 🚀 Features

- ✨ Pretrained **DistilBERT** for high-quality text embeddings
- 🧠 **Logistic Regression** for simple, interpretable intent prediction
- 🌐 **Flask API** with a clean web UI
- 📦 Organized, modular Python project structure
- 🧪 Easy to extend, test, and deploy

---

## 📂 Project Structure
```
atis-intent_classifier/
├── app.py                # Flask server for prediction
├── model/
│   ├── feature_extractor.py   # BERT-based embedding generator
│   ├── cls_train.py           # Training script for the classifier
│   ├── intent_model.pkl       # Saved scikit-learn model
├── data/
│   └── atis_intents.csv   # Training data (phrases + labels)
├── templates/
│   └── index.html        # Frontend UI
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 📋 Prerequisites

- Python 3.8+
- PyTorch
- Transformers (`huggingface/transformers`)
- Flask
- scikit-learn
- pandas

---

## ⚙️ Installation

1️⃣ Clone this repository  
```bash
git clone https://github.com/MohamedAhmed35/natural-language-processing.git
cd natural-language-processing/text-classification/atis-intent-classifier
```

2️⃣ Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
# .\venv\Scripts\activate  # Windows
```
3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
---
## Train the classifier
```bash
python model/train.py
```
This:
- Generates BERT embeddings for eahc phrase
- Trains a logistic regression model
- Saving it to model/intent-model.pkl

## Run the Web app
Start the Flask server:
```bash
python app.py
```

--- 
## ✅ Example Phrases
| Example Phrase                                    | Expected Intent       |
| ------------------------------------------------- | ----------------------|
| Show me flights from Boston to New York tomorrow. | `atis_flight`         |
| How much is a round-trip to Miami?                | `atis_airfare`        |
| What's the weather like in Chicago next week?.    | `atis_ground_service` |
| What is the arrival time for flight UA202?        | `atis_flight_time`    |
