# FlowScore

## Explainable Credit Scoring for India's Informal Economy Using UPI Behavioral Analytics

FlowScore is an AI-based credit scoring system designed to evaluate the financial reliability of informal businesses using behavioral transaction patterns.

The system analyzes simulated UPI transaction behavior, performs feature engineering, trains an XGBoost classification model, and generates an interpretable credit score along with insights explaining the prediction.

The project demonstrates how alternative financial data can support credit assessment for businesses that may not have traditional financial histories.

---

# Project Overview

Many small businesses and informal vendors face difficulty accessing credit due to limited documentation and lack of conventional financial records.

FlowScore addresses this challenge by analyzing digital payment behavior and converting transaction patterns into meaningful financial indicators.

The system evaluates:

* Transaction activity
* Merchant diversity
* Income patterns
* Savings behavior
* Payment consistency
* Financial stability indicators

It generates a FlowScore rating and provides supporting analysis for each vendor.

---

# Key Features

## AI-Based Credit Scoring

FlowScore generates a credit score between **300 and 900** and classifies vendors into different risk categories.

| Score Range | Category  |
| ----------- | --------- |
| 750–900     | Elite     |
| 600–749     | Strong    |
| 450–599     | Emerging  |
| 300–449     | High Risk |

---

## Explainable AI

FlowScore focuses on making predictions interpretable instead of treating the model as a black box.

The system uses **SHAP (SHapley Additive exPlanations)** to analyze how different financial features influence vendor predictions.

Explainability components include:

* SHAP-based feature importance analysis
* Feature contribution analysis
* Raw financial data visualization
* Engineered feature analysis
* Vendor-level score interpretation
* AI-generated improvement recommendations

This allows users to understand not only the generated score but also the financial factors responsible for the prediction.

---

## Interactive Dashboard

The Flask-based dashboard provides:

* Vendor database exploration
* Vendor search functionality
* Risk category filtering
* Sorting functionality
* Detailed vendor analysis cards
* Financial behavior visualization
* Score insights
* AI-generated roadmap recommendations

---

# System Workflow

```
Synthetic UPI Transaction Data
            |
            |
Data Generation
            |
            |
Feature Engineering
            |
            |
Data Preprocessing
            |
            |
XGBoost Model Training
            |
            |
Model Evaluation
            |
            |
Saved Model Components
            |
            |
Flask Dashboard
```

---

# Dataset

The project generates synthetic UPI transaction data representing vendor financial behavior.

The generated transaction dataset contains:

* Average weekly transactions
* Unique merchants
* Monthly inflow
* Income volatility
* Savings ratio
* Bill payment timing

The generated data is processed and transformed into additional behavioral features used for credit scoring.

---

# Feature Engineering

The system creates additional features to capture deeper financial behavior patterns.

| Feature                  | Description                                |
| ------------------------ | ------------------------------------------ |
| Transaction Consistency  | Measures stability of transaction activity |
| Merchant Diversity Score | Measures diversity of payment sources      |
| Income Stability Score   | Measures consistency of income flow        |
| Savings Score            | Represents saving behavior                 |
| Temporal Score           | Represents payment timing behavior         |

These engineered features help identify financial reliability beyond simple transaction volume.

---

# Machine Learning Pipeline

FlowScore follows an end-to-end machine learning workflow:

1. Generate synthetic UPI transaction data
2. Perform feature engineering
3. Preprocess the dataset
4. Train the XGBoost classification model
5. Evaluate model performance
6. Save trained model components
7. Load the trained model in Flask
8. Generate vendor predictions and explanations

The trained model and preprocessing objects are stored locally and loaded by the Flask application.

---

# Technology Stack

## Backend

* Python
* Flask

## Machine Learning

* XGBoost
* Scikit-learn
* SHAP
* Pandas
* NumPy

## Frontend

* HTML
* Tailwind CSS
* JavaScript

---

# Project Structure

```
FlowScore/
│
├── data/
│   ├── upi_transactions.csv
│   └── upi_features.csv
│
├── models/
│   ├── data.pkl
│   ├── scaler.pkl
│   └── xgb.pkl
│
├── src/
│   ├── generate_data.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── templates/
│   └── dashboard.html
│
├── flowscore.py
├── requirements.txt
├── setup.py
└── README.md
```

---

# Installation and Setup

## 1. Clone Repository

```bash
git clone https://github.com/pratik-mohan-404/flowscore-engine.git

cd flowscore-engine
```

---

## 2. Install Requirements

Install all project dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Run Setup Pipeline

Execute:

```bash
python setup.py
```

The setup script runs the complete machine learning pipeline:

* Data generation
* Feature engineering
* Data preprocessing
* Model training
* Model evaluation

After successful execution, the following files will be generated.

### Data Folder

```
data/
├── upi_transactions.csv
└── upi_features.csv
```

### Models Folder

```
models/
├── data.pkl
├── scaler.pkl
└── xgb.pkl
```

---

## 4. Start Flask Application

After completing the setup pipeline, run:

```bash
python flowscore.py
```

Open the dashboard:

```
http://127.0.0.1:5000
```

---

# Dashboard Usage

1. Open the vendor database.
2. Search or filter vendors.
3. Select a vendor.
4. Review:

* FlowScore rating
* Risk category
* Raw financial data
* Engineered features
* Score insights
* AI improvement roadmap

---

# Future Improvements

Possible extensions:

* Real UPI transaction API integration
* Real-time credit monitoring
* Fraud detection module
* Loan recommendation system
* Mobile application support
* Continuous model improvement using repayment history

---

# Author

- Pratik Mohan
- Vindhya Patcha

---

# License

This project is developed for academic and research purposes.
