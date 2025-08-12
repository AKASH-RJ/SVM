# app.py
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load("svm_model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    review_text = ""
    if request.method == "POST":
        review_text = request.form["review"]
        prediction = model.predict([review_text])[0]
    return render_template("index.html", prediction=prediction, review=review_text)

if __name__ == "__main__":
    app.run(debug=True)
