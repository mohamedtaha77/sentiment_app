# 🎯 Sentiment Analysis Web App - IMDb Reviews

This is a simple yet effective **Sentiment Analysis** app built with **Streamlit**. It analyzes movie or product reviews and classifies them as **Positive** or **Negative** using trained machine learning models.

## 🌟 Features

- Clean web interface using **Streamlit**
- Pre-trained models: **Logistic Regression** & **Naive Bayes**
- Custom text preprocessing (tokenization, lemmatization, stopword removal)
- Visual feedback with sentiment label and confidence score
- Easily deployable on **Streamlit Cloud**

---

## 🚀 Try It Locally

1. **Clone the repository**:

```
git clone https://github.com/mohamedtaha77/sentiment_app.git
cd sentiment_app
```

2. **Install dependencies**:

Make sure Python 3.8+ is installed.

```
pip install -r requirements.txt
```

3. **Run the app**:

```
streamlit run streamlit_app.py
```

---

## 📦 File Structure

```
sentiment_app/
│
├── logistic_regression_model.pkl    # Trained Logistic Regression model
├── naive_bayes_model.pkl            # Trained Naive Bayes model
├── tfidf_vectorizer.pkl             # Fitted TF-IDF Vectorizer
├── streamlit_app.py                 # Main app file
└── README.md                        # You're here!
```

---

## 🧠 Model Training Info

The models were trained using the **IMDb Movie Review Dataset** with TF-IDF features and standard text preprocessing techniques.

- **Accuracy:**
  - Logistic Regression: ~88%
  - Naive Bayes: ~86%

---
