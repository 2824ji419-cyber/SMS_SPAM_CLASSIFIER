import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

nltk.download('stopwords')

ps = PorterStemmer()

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

st.title("📩 SMS Spam Classifier")

msg = st.text_area("Enter SMS Message")

if st.button("Predict"):
    processed_msg = preprocess(msg)
    prediction = model.predict([processed_msg])

    if prediction[0] == 1:
        st.error("🚨 Spam Message")
    else:
        st.success("✅ Not Spam")
