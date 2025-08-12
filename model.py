# model.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("svm.csv")

# Split data
X = df["review"]
y = df["label"]

# Create pipeline: TF-IDF + SVM
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words='english')),
    ("svm", LinearSVC())
])

# Train
model.fit(X, y)

# Save model
joblib.dump(model, "svm_model.pkl")
print("✅ Model trained and saved as svm_model.pkl")
