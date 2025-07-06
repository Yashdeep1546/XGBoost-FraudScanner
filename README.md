# 💳 XGBoost FraudScanner

**XGBoost FraudScanner** is a real-time credit card fraud detection app powered by **machine learning** and deployed using **Streamlit**. This project identifies fraudulent transactions from highly imbalanced data using the XGBoost algorithm.

![App Screenshot](https://your-screenshot-link-if-any.png)

---

## 🚀 Live Demo

👉 [Click here to try the app (Streamlit Cloud)](https://your-app-url.streamlit.app)  
*(Add the link after deployment)*

---

## 📂 Project Structure

XGBoost-FraudScanner/
├── app.py # Streamlit frontend
├── xgboost_fraud_model_full.pkl # Trained XGBoost model
├── scaler_full.pkl # Scaler for Time and Amount
├── requirements.txt # Dependencies
├── README.md # This file
└── .gitignore # Optional

yaml
Copy
Edit

---

## 📊 Dataset

This project uses the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains:

- **284,807** transactions  
- **492** fraud cases (~0.17%)  
- PCA-transformed features `V1` to `V28`, plus `Time`, `Amount`, and `Class`

---

## ⚙️ Features

- Upload a `.csv` file of transactions
- Model predicts **"Fraud"** or **"Not Fraud"** per transaction
- Displays:
  - Prediction results
  - Total predicted frauds
  - Total actual frauds (if `Class` column is included)
  - True positives & recall (fraud detection rate)

---

## 🧠 Model Details

- **Model:** XGBoost Classifier
- **Scaler:** StandardScaler (only for `Time` and `Amount`)
- **Training Method:** Used `scale_pos_weight` to handle imbalance
- **Performance on Full Dataset:**
  - Precision: 0.89
  - Recall: 0.79
  - F1 Score: 0.84
  - Fraud Detection Rate (on real data): **91.87%**

---

## 📦 How to Run Locally

```bash
git clone https://github.com/Yashdeep1546/XGBoost-FraudScanner.git
cd XGBoost-FraudScanner
pip install -r requirements.txt
streamlit run app.py
📈 Future Improvements
Add interactive visualizations

Manual transaction entry

Deploy as a REST API

Add threshold slider for sensitivity control

👨‍💻 Author
Made with 💙 by Yashdeep
B.Tech CS Student | ML Learner | Aiming 1 Cr 😎

🛡️ License
This project is open source under the MIT License.
