import streamlit as st
import pickle
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialize
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Load model and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


# Text preprocessing function
def transform_text(text):

    text = text.lower()

    # Tokenization
    text = nltk.word_tokenize(text)

    # Remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords
    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Streamlit application
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered"
)

st.title("📩 Email/SMS Spam Detection")
st.write("Enter your SMS or Email message to check whether it is Spam or Not Spam.")

# User input
input_sms = st.text_area("Enter your message")

# Prediction button
if st.button("Predict"):

    if input_sms.strip() == "":
        st.warning("Please enter a message first.")

    else:
        # 1. Preprocessing
        transformed_sms = transform_text(input_sms)

        # 2. Vectorization
        vector_input = tfidf.transform([transformed_sms])

        # 3. Prediction
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.error("🚨 Spam Message")

        else:
            st.success("✅ Not Spam (Ham)")