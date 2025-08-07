import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("✅ Loaded vectorizer.")
print("idf_ exists?", hasattr(vectorizer, "idf_"))
