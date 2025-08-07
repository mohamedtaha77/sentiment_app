# 🎯 Sentiment Analysis Web App - IMDb Reviews

This is a simple yet effective **Sentiment Analysis** app built with **Streamlit**.  
It analyzes movie or product reviews and classifies them as **Positive** or **Negative** using two models:
- **Logistic Regression**
- **Naive Bayes**

---

## 📌 Features

✅ Clean & lemmatize input text  
✅ TF-IDF vectorization (top 5000 words)  
✅ Predict using two trained models  
✅ Displays **sentiment** and **model confidence**  
✅ Visual interface built in **Streamlit**  
✅ Includes **notebook** for model training

---
## 🔗 Live Demo

Try the app live here:  
👉 [https://sentimentapp77.streamlit.app/](https://sentimentapp77.streamlit.app)

---

## 📁 Files Included

| File | Description |
|------|-------------|
| `streamlit_app.py` | Main app for prediction |
| `train_and_save.py` | Script to train and save models |
| `Elevvo_NLP_Internship_Task1.ipynb` | Notebook used for full pipeline |
| `logistic_regression_model.pkl` | Trained logistic model |
| `naive_bayes_model.pkl` | Trained NB model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `imdb_train_processed.csv` | Cleaned training data |
| `requirements.txt` | All required packages |

---

## 🚀 How to Run Locally

1. **Clone this repo**:

```bash
git clone https://github.com/mohamedtaha77/sentiment_app.git
cd sentiment_app
```

2. **Create a virtual environment (optional but recommended)**:

```bash
python -m venv venv
venv\Scripts\activate  # on Windows
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Run the app**:

```bash
streamlit run streamlit_app.py
```

---

## 🌍 Deployment (Optional)

You can deploy the app to **Streamlit Cloud**:

- Go to: https://streamlit.io/cloud
- Connect your GitHub repo
- Set the **main file** to: `streamlit_app.py`

---

## 📓 Notebook Workflow

The full training pipeline is in the notebook:

> [`Elevvo_NLP_Internship_Task1.ipynb`](./Elevvo_NLP_Internship_Task1.ipynb)

And for better visibility, you can check it on Colab:

> [Elevvo_NLP_Internship_Task1.ipynb]((https://colab.research.google.com/drive/1-VjoXPIX3mrjQONyJhEksj5HupEN3fvm?usp=sharing))

It covers:
- Dataset loading from HuggingFace
- Text preprocessing
- TF-IDF vectorization
- Model training & evaluation
- WordCloud visualizations
- Model saving with `joblib`
