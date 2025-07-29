import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_extraction import feature_gen
from torch.utils.data import Dataset, DataLoader


class TextDataset(DataLoader):
    """
    Custom Dataset for texts and labels
    """
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# Load dataset
df = pd.read_csv('../data/atis_intents.csv', header=None, names=['intent', 'query'])
texts = df['query'].values
labels = df['intent'].values


# Train/Test split
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

data_dict = {"query": texts_test,
            "label": labels_test}

pd.DataFrame(data_dict).to_csv("test.csv", index=False)

# Prepare datasets & loaders
train_dataset = TextDataset(texts_train, labels_train)
test_dataset = TextDataset(texts_test, labels_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)


# Generate embeddings in batches
def get_embeddings_and_labels(dataloader):
    X_list = []
    y_list = []

    for batch_texts, batch_labels in dataloader:
        emb = feature_gen(batch_texts)
        X_list.append(emb)
        y_list.extend(batch_labels)

    X = np.vstack(X_list)
    y = np.array(y_list)

    return X, y


print("Generating training embeddings...")
X_train, y_train = get_embeddings_and_labels(train_loader)

print("Generating test embeddings...")
X_test, y_test = get_embeddings_and_labels(test_loader)


clf = LogisticRegression(random_state = 42, max_iter = 1000, class_weight='balanced')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open('intent_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("Model trained and saved as mode/intent_model.pkl")
