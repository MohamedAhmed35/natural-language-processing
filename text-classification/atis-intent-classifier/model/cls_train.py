import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_extraction import feature_gen


# Load data
df = pd.read_csv('../data/atis_intents.csv', header=None, names=['intent', 'query'])


texts = df['query'].values
labels = df['intent'].values

X = []
for text in texts:
    emb = feature_gen(text)
    X.append(emb[0])

X = np.array(X)
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state = 42, max_iter = 1000)
clf.fit(X, y)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open('intent_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved as mode/intent_model.pkl")