# ✈️ Intent Classifier — BERT + Flask

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
atis-intent_classifier/

│
├── app.py # Flask server for prediction
├── model/
│ ├── feature_extractor.py # BERT-based embedding generator
│ ├── cls_train.py # Training script for the classifier
│ ├── intent_model.pkl # Saved scikit-learn model
│
├── data/
│ └── atis_intents.csv # Training data (phrases + labels)
│
├── templates/
│ └── index.html # Frontend UI
│
├── static/ # (Optional static assets: CSS, JS)
│
├── requirements.txt # Python dependencies
└── README.md # This file
