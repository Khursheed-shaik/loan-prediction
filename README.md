# 🏦 Loan Approval Prediction System

A Machine Learning web application that predicts whether a loan application will be approved or rejected based on applicant details. The model is trained on historical loan data and deployed using Streamlit.

## 🚀 Live Demo
Add your Streamlit link here after deployment:
https://loan-prediction-fzqe9fv7oi28vt5b7vnj69.streamlit.app/

---

## 📌 Project Overview

This project builds an end-to-end ML pipeline for loan approval prediction including:

- Data preprocessing
- Feature engineering
- Model training & evaluation
- Prediction pipeline
- Interactive Streamlit web app
- Cloud deployment

The system takes applicant inputs and predicts loan approval status in real time.

---

## 🧠 Machine Learning Details

- Model: Random Forest Classifier
- Problem Type: Binary Classification
- Features: Applicant income, credit history, loan amount, marital status, etc.
- Preprocessing: Missing value handling, encoding, feature transformation
- Evaluation: Accuracy & validation metrics

---

## 🗂 Project Structure
loan-prediction/
│
├── app/
│   └── app.py                ✅ Streamlit entry point
│
├── data/
│   ├── loan_predictions.csv
│   ├── loan_sanction_test.csv
│   └── loan_sanction_train.csv
│
├── models/
│   └── random_forest_model.pkl   ✅ trained model
│
├── notebooks/
│   └── EDA.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── evaluate_model.py
│   ├── predict.py
│   └── __init__.py
│
└── requirements.txt


