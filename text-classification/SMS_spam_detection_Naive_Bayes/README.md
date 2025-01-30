# SMS Spam Classifier using Naive Bayes

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

A machine learning model to classify SMS messages as **spam** or **ham** (non-spam) using **Multinomial Naive Bayes**. Built with Python, pandas, and scikit-learn. The project is implemented in a Jupyter Notebook for transparency and reproducibility.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Implementation Steps](#implementation-steps)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)

---

## Overview
This project trains a **Multinomial Naive Bayes classifier** to detect spam SMS messages. The workflow includes:
1. **Data Preprocessing**: Label conversion, train-test splitting.
2. **Feature Extraction**: Bag-of-words representation using `CountVectorizer`.
3. **Model Training**: Multinomial Naive Bayes classifier.
4. **Evaluation**: Metrics like accuracy, precision, recall, and F1-score.

---

## Project Structure

SMS_spam_detection_Naive_Bayes

├── smsSpamCollection/ # Dataset folder (manually added)
│ └── SMSSpamCollection # Raw dataset file
├── SMS_spam_detection.ipynb # Jupyter Notebook with code
├── README.md # This file

---

## Installation
**Clone the repository**:
   ```bash
   git clone https://github.com/MohamedAhmed35/SMS_spam_detection_Naive_Bayes.git
   cd SMS_spam_detection_Naive_Bayes
   ```
---

## Dataset
Source: UCI Machine Learning Repository.

Format: Two columns (no headers):

label: ham (non-spam) or spam.

sms_message: Raw text of the SMS.

Size: 5,574 messages (747 spam, 4,827 ham).
  

---
## Implementation Steps
Data Preprocessing:

Convert labels to binary (ham → 0, spam → 1).

Split data into training (75%) and testing (25%) sets.

Feature Extraction:

Use CountVectorizer to create a bag-of-words matrix.

Model Training:

Train a MultinomialNB classifier from scikit-learn.

Evaluation:

Compute accuracy, precision, recall, and F1-score.

---

## Results
Metric	Score
Accuracy	98.85%
Precision	97.21%
Recall	94.05%
F1-Score	95.60%
