# 🛍️ Amazon Review Sentiment Analysis Dashboard

## 📝 Project Overview

This project implements a sentiment analysis system that classifies Amazon product reviews as **positive or negative** using Natural Language Processing (NLP) and machine learning techniques.

The system processes raw text data, transforms it into numerical features, and applies a classification model to predict sentiment. It also includes an interactive web dashboard for real-time predictions and bulk analysis.

The objective is to demonstrate end-to-end ML workflow including data preprocessing, feature engineering, model training, evaluation, and deployment.

---

## 📂 Repository Contents

| File Name          | Type           | Description                                        |
| ------------------ | -------------- | -------------------------------------------------- |
| `train_model.py`   | Python Script  | Data preprocessing, model training, and evaluation |
| `app.py`           | Streamlit App  | Interactive dashboard for sentiment prediction     |
| `model.pkl`        | Model File     | Trained Logistic Regression model                  |
| `vectorizer.pkl`   | Model Artifact | TF-IDF vectorizer for text transformation          |
| `requirements.txt` | Dependencies   | Required Python libraries                          |
| `README.md`        | Documentation  | Project overview and usage                         |

---

## ⚙️ Tech Stack

| Component            | Tool / Library      | Purpose                                  |
| -------------------- | ------------------- | ---------------------------------------- |
| Data Processing      | Pandas, NumPy       | Data cleaning and manipulation           |
| NLP Processing       | NLTK                | Text preprocessing (stopwords, stemming) |
| Feature Engineering  | TF-IDF Vectorizer   | Convert text into numerical features     |
| Model                | Logistic Regression | Sentiment classification                 |
| Visualization        | Matplotlib          | Sentiment distribution charts            |
| Deployment           | Streamlit           | Interactive web application              |
| Programming Language | Python              | Core implementation                      |

---

## 🏗️ Data Preparation & Feature Engineering

### 1️⃣ Data Cleaning

* Removed special characters and noise from text

* Converted text to lowercase

* Removed stopwords using NLTK

* Applied stemming using PorterStemmer

* Handled missing values

---

### 2️⃣ Feature Extraction

* Converted cleaned text into numerical form using **TF-IDF Vectorization**

* Limited features to top 5000 terms for efficiency

* Transformed both training and test data consistently

---

## 🤖 Model Training & Evaluation

* Used **Logistic Regression** for classification

* Trained on labeled Amazon review dataset

* Evaluated using:

  * Accuracy

  * Precision

  * Recall

  * F1-score

👉 Achieved approximately **85% accuracy** on test data

---

## 🚀 Application Features

### 🔹 Single Prediction

* Input custom text

* Get instant sentiment prediction

* Displays confidence score

---

### 🔹 Bulk Prediction

* Upload CSV file with `text` column

* Analyze multiple reviews at once

* View results in table format

* Download predictions

---

### 📊 Visualization

* Sentiment distribution chart (Positive vs Negative)

* Helps understand overall trends

---

## 📊 Output & Insights

* Classifies reviews as **Positive 😊** or **Negative 😞**

* Demonstrates real-world NLP pipeline

* Useful for:

  * Product feedback analysis

  * Customer experience insights

  * Market research

---

## 🔗 Live Demo

👉 https://amazon-reviews-sentiment-analysis-05.streamlit.app/

---

## 💡 Future Enhancements

* Add Neutral sentiment class
* Use advanced models (BERT, LSTM)
* Improve UI with interactive charts
* Deploy on cloud with API support

---

## 🎯 Conclusion

This project showcases a complete **end-to-end NLP pipeline**, from raw text processing to deployment. It highlights how machine learning can be applied to extract 
meaningful insights from textual data in real-world scenarios.

---


