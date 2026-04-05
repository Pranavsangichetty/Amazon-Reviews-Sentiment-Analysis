# =========================================
# Sentiment Analysis Dashboard
# =========================================

import streamlit as st
import joblib
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =========================================
# Fix NLTK
# =========================================
nltk.download('stopwords', quiet=True)

# =========================================
# Load Model
# =========================================
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except:
    st.error("Model files not found!")
    st.stop()

# =========================================
# Text Cleaning
# =========================================
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# =========================================
# Page Config
# =========================================
st.set_page_config(page_title="Sentiment Dashboard", layout="wide")

# =========================================
# HEADER
# =========================================
st.title("Sentiment Analysis Dashboard")
st.markdown("### Analyze text sentiment using Machine Learning")

st.markdown("---")

# =========================================
# SIDEBAR
# =========================================
st.sidebar.title("Options")

mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Single Prediction", "Bulk Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info("Built with NLP + Logistic Regression")

# =========================================
# SINGLE PREDICTION
# =========================================
if mode == "Single Prediction":

    st.subheader("Enter Text")

    user_input = st.text_area("Type your sentence here:")

    if st.button("Analyze"):

        if user_input.strip() == "":
            st.warning("Please enter text")
        else:
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            confidence = model.predict_proba(vectorized)[0]

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Prediction")
                if prediction == 1:
                    st.success("😊 Positive")
                else:
                    st.error("😞 Negative")

            with col2:
                st.subheader("Confidence")
                st.write(f"{max(confidence):.2f}")

# =========================================
# BULK PREDICTION (UPLOAD CSV)
# =========================================
elif mode == "Bulk Prediction":

    st.subheader("Upload CSV File")

    uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if 'text' not in df.columns:
            st.error("CSV must contain 'text' column")
        else:
            df['clean_text'] = df['text'].apply(clean_text)
            vectors = vectorizer.transform(df['clean_text'])

            df['prediction'] = model.predict(vectors)

            st.subheader("Results")
            st.dataframe(df.head())

            # Chart
            st.subheader("Sentiment Distribution")

            counts = df['prediction'].value_counts()

            fig, ax = plt.subplots()
            ax.bar(['Negative', 'Positive'], counts)
            st.pyplot(fig)

            # Download
            st.download_button(
                label="Download Results",
                data=df.to_csv(index=False),
                file_name="sentiment_results.csv",
                mime="text/csv"
            )

# =========================================
# FOOTER
# =========================================
st.markdown("---")
st.caption("Advanced Sentiment Analysis App | Built with Streamlit")