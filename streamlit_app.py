import streamlit as st
import joblib
import re
import numpy as np

# =======================
# Load Models & Vectorizer
# =======================

logreg_model = joblib.load("logistic_regression_model.pkl")
nb_model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# =======================
# Text Preprocessing
# =======================

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)   # Remove punctuation
    text = re.sub(r"\d+", "", text)       # Remove numbers
    tokens = text.split()                 # Simple whitespace tokenizer
    return " ".join(tokens)

# =======================
# Streamlit UI
# =======================

st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 IMDb Review Sentiment Analyzer")

st.markdown(
    """
    Enter a movie review, choose your model,  
    and get a real-time sentiment prediction 🚀  
    """
)

# Text input
user_input = st.text_area("✍️ Your review:", height=150, placeholder="E.g. I absolutely loved this movie...")

# Model selector
model_choice = st.radio("🤖 Choose a model:", ["Logistic Regression", "Naive Bayes"])

# Predict button
if st.button("🔍 Analyze"):
    if not user_input.strip():
        st.warning("Please enter a review before analyzing.")
    else:
        cleaned_text = preprocess_text(user_input)
        features = vectorizer.transform([cleaned_text])

        if model_choice == "Logistic Regression":
            model = logreg_model
        else:
            model = nb_model

        prediction = model.predict(features)[0]
        confidence = model.predict_proba(features)[0][1] if prediction == 1 else model.predict_proba(features)[0][0]
        confidence_pct = round(confidence * 100, 2)

        sentiment = "🟢 Positive" if prediction == 1 else "🔴 Negative"

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.markdown(f"**Sentiment:** {sentiment}")
        st.markdown(f"**Confidence:** `{confidence_pct}%`")
        st.markdown("---")
        st.caption("Confidence shows how sure the model is about this prediction.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center;'>Made with ❤️ for Elevvo Internship Task 1</div>",
    unsafe_allow_html=True
)