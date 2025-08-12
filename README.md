# Product Review Classification using SVM & Flask

## Overview

This project classifies **product reviews** as either **Positive** or **Negative** using **Support Vector Machine (SVM)**. The model is trained on a dataset of customer reviews and deployed as a **Flask web application** with HTML & CSS.

-----

## Features

  - **SVM Classifier** for accurate text classification.
  - **Flask backend** for serving predictions.
  - **Interactive HTML form** to input a product review.
  - **CSS styling** for a clean, responsive interface.
  - **CSV dataset** with 200 product reviews.

-----

## Project Structure

```
product_review_svm/
│
├── model.py             # Trains and saves the SVM model
├── app.py               # Flask application for predictions
├── templates/
│   ├── index.html       # Main input form
│   └── result.html      # Displays classification result
├── static/
│   └── style.css        # CSS for styling
├── dataset.csv          # Dataset (200 product reviews)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

-----

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` file contains:

```
Flask==3.0.0
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.26.2
```

-----

## Dataset

The dataset (`dataset.csv`) contains product reviews labeled as Positive or Negative.
Example:

```
review,label
"This product is amazing and works perfectly!",Positive
"Terrible quality, broke after one week",Negative
"Very comfortable and great design",Positive
```

Features:

  - `review`: Text of the customer review.
  - `label`: Sentiment category (Positive / Negative).

-----

## How It Works

### Model Training (`model.py`)

  - Loads the dataset from `dataset.csv`.
  - Uses TF-IDF Vectorizer to convert text into numerical features.
  - Trains an SVM classifier.
  - Saves the trained model as `model.pkl`.

### Web Application (`app.py`)

  - Loads `model.pkl`.
  - Accepts user review input from an HTML form.
  - Classifies the review as Positive or Negative.
  - Displays the classification result.

-----

## Running the Project

1.  **Train the Model**
    ```bash
    python model.py
    ```
2.  **Run Flask App**
    ```bash
    python app.py
    ```
3.  **Open in Browser**
    Go to: `http://127.0.0.1:5000/`

-----

## Screenshots
---
Home Page

<img width="702" height="304" alt="Screenshot 2025-08-12 121952" src="https://github.com/user-attachments/assets/17d4c745-fa8c-4109-8d2b-e4fdf11c44e7" />

---
Classification Result

<img width="669" height="361" alt="Screenshot 2025-08-12 122003" src="https://github.com/user-attachments/assets/fece4c4e-2817-4cbf-b5a6-f70f7f71d09c" />

-----
