import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
fake = pd.read_csv('data/Fake.csv')
real = pd.read_csv('data/True.csv')
fake['label'] = 0
real['label'] = 1

df = pd.concat([fake, real], ignore_index=True)
df['text'] = df['title'] + ' ' + df['text']
df = df[['text', 'label']].dropna()
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# TF-IDF + Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1,2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print(f"✅ Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

import os
os.makedirs('model', exist_ok=True)
pickle.dump(model, open('model/lr_model.pkl', 'wb'))
pickle.dump(tfidf, open('model/tfidf.pkl', 'wb'))
print("✅ Model saved!")