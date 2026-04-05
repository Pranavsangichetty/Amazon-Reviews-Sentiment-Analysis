# =========================================
# Import the Libraries
# =========================================

import pandas as pd
import numpy as np
import re
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# =========================================
# Download NLTK data
# =========================================
nltk.download('stopwords')

# =========================================
# Load Dataset
# =========================================
train_path = r"C:\Users\sangi\Downloads\excel_data\train.csv"
test_path = r"C:\Users\sangi\Downloads\excel_data\test.csv"

train_df = pd.read_csv(train_path, header=None)
test_df = pd.read_csv(test_path, header=None)

# Assign column names
train_df.columns = ['label', 'title', 'text']
test_df.columns = ['label', 'title', 'text']

print("Data Loaded Successfully")

# =========================================
# Sample Data
# =========================================
train_df = train_df.sample(50000, random_state=42)
test_df = test_df.sample(10000, random_state=42)

print("Sampling Done")
print("Train size:", len(train_df))
print("Test size:", len(test_df))

# =========================================
# Handle Missing Values
# =========================================
train_df.dropna(subset=['text'], inplace=True)
test_df.dropna(subset=['text'], inplace=True)

# =========================================
# Convert Labels (1,2) → (0,1)
# =========================================
train_df['label'] = train_df['label'].astype(int).map({1: 0, 2: 1})
test_df['label'] = test_df['label'].astype(int).map({1: 0, 2: 1})

print("\n Label Distribution:")
print(train_df['label'].value_counts())

# =========================================
# Text Cleaning Function
# =========================================
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
train_df['clean_text'] = train_df['text'].apply(clean_text)
test_df['clean_text'] = test_df['text'].apply(clean_text)

print("Text Cleaning Done")

# =========================================
# TF-IDF Vectorization
# =========================================
vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_df['clean_text'])
X_test = vectorizer.transform(test_df['clean_text'])

y_train = train_df['label']
y_test = test_df['label']

print("Vectorization Done")

# =========================================
# Model Training
# =========================================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Model Training Completed")

# =========================================
# Model Evaluation
# =========================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n Accuracy:", accuracy)
print("\n Classification Report:\n")
print(classification_report(y_test, y_pred))

# =========================================
# Save Model & Vectorizer
# =========================================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\n Model Saved as model.pkl")
print("Vectorizer Saved as vectorizer.pkl")

# =========================================
# Sample Prediction (Quick Test)
# =========================================
sample_text = "This product is amazing and works perfectly!"

cleaned = clean_text(sample_text)
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)[0]

print("\n Sample Prediction:")
print("Text:", sample_text)
print("Prediction:", "Positive 😊" if prediction == 1 else "Negative 😞")