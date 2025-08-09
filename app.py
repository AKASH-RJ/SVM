from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
df = pd.read_csv("svm.csv")

# Feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = LinearSVC()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review_text = request.form['review']
    review_vector = vectorizer.transform([review_text])
    prediction = model.predict(review_vector)[0]
    return render_template('result.html', review=review_text, sentiment=prediction)

if __name__ == '__main__':
    app.run(debug=True)
