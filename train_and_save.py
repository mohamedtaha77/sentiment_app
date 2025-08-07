import pandas as pd
import re
import joblib
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Load IMDb dataset
dataset = load_dataset("imdb")
df = pd.DataFrame(dataset['train'])

# Clean text
def clean(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    return text

df["clean_text"] = df["text"].apply(clean)

# Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["clean_text"])
y = df["label"]

# Train models
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X, y)

nb = MultinomialNB()
nb.fit(X, y)

# Save all
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(logreg, "logistic_regression_model.pkl")
joblib.dump(nb, "naive_bayes_model.pkl")

print("✅ All models and vectorizer saved.")