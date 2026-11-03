
import streamlit as st
import pickle
import string
import nltk
import os

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered"
)

# -----------------------------------
# Download NLTK Resources
# -----------------------------------
@st.cache_resource
def download_nltk_resources():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)

    return set(stopwords.words("english"))

try:
    stop_words = download_nltk_resources()
except Exception as e:
    st.error(f"Error loading NLTK resources: {e}")
    st.stop()

# -----------------------------------
# Initialize Porter Stemmer
# -----------------------------------
ps = PorterStemmer()

# -----------------------------------
# Load Model and Vectorizer
# -----------------------------------
@st.cache_resource
def load_models():

    if not os.path.exists("vectorizer.pkl"):
        raise FileNotFoundError("vectorizer.pkl not found!")

    if not os.path.exists("model.pkl"):
        raise FileNotFoundError("model.pkl not found!")

    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    return tfidf, model

try:
    tfidf, model = load_models()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -----------------------------------
# Text Preprocessing Function
# -----------------------------------
def transform_text(text):

    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove special characters
    tokens = [
        word for word in tokens
        if word.isalnum()
    ]

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
    ]

    # Stemming
    tokens = [
        ps.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)

# -----------------------------------
# Streamlit Application UI
# -----------------------------------
st.title("📩 Email/SMS Spam Detection")

st.write(
    "Enter your SMS or Email message below "
    "to check whether it is Spam or Not Spam."
)

st.divider()

# User Input
input_sms = st.text_area(
    "Enter your message:",
    placeholder="Example: Congratulations! You have won a free prize. Click here!",
    height=150
)

# Prediction Button
if st.button("🔍 Predict", use_container_width=True):

    if not input_sms.strip():

        st.warning("⚠️ Please enter a message first.")

    else:

        try:
            # Step 1: Text Preprocessing
            transformed_sms = transform_text(input_sms)

            # Step 2: Vectorization
            vector_input = tfidf.transform([transformed_sms])

            # Step 3: Prediction
            result = model.predict(vector_input)[0]

            # Step 4: Display Result
            st.divider()

            if result == 1:

                st.error("🚨 Spam Message Detected!")

                st.write(
                    "This message appears to be spam. "
                    "Please be careful before clicking links "
                    "or sharing personal information."
                )

            else:

                st.success("✅ Not Spam (Ham)")

                st.write(
                    "This message appears to be legitimate."
                )

        except Exception as e:

            st.error(f"Prediction Error: {e}")

# -----------------------------------
# Footer
# -----------------------------------
st.divider()

st.caption(
    "Built with Python, Machine Learning, NLTK, "
    "Scikit-learn and Streamlit."
)
