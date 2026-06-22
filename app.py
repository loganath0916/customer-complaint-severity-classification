import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Preprocessing
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# UI
st.title("Customer Complaint Severity Classification")

st.write(
    "Predict whether a complaint is Low, Medium, or High Priority"
)

user_input = st.text_area(
    "Enter Customer Complaint"
)

if st.button("Predict"):

    cleaned_text = preprocess_text(user_input)

    vectorized_text = vectorizer.transform(
        [cleaned_text]
    )

    prediction = model.predict(
        vectorized_text
    )[0]

    labels = {
        0: "Low Priority",
        1: "Medium Priority",
        2: "High Priority"
    }

    st.success(
        f"Predicted Severity: {labels[prediction]}"
    )