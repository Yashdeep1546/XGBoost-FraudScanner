# 💳 XGBoost FraudScanner

An end-to-end **Credit Card Fraud Detection Web App** built with **XGBoost**, **Streamlit**, and real-world data. This app empowers users to detect fraudulent transactions through an interactive interface, custom threshold control, detailed analytics, and even single-transaction simulation.

> 🧠 Powered by machine learning. Designed for interactivity. Built from scratch.

---

## 🚀 Live Demo

👉 [Try it live on Streamlit Cloud](https://xgboost-fraudscanner-hm2skpberllwckz3nmt5sg.streamlit.app/)

---

## 📦 Features

- ✅ Upload a CSV of credit card transactions and detect frauds in seconds
- 🎯 Model trained on highly imbalanced dataset (only 0.17% fraud)
- 📈 Real-time **visualizations** (pie & bar charts)
- 🎛️ Customizable **threshold slider** to control sensitivity
- ✍️ **Manual transaction simulator** to test single cases live
- 📊 Evaluation metrics: precision, recall, F1 score
- 🌐 Fully deployed on Streamlit Cloud

---

## 🔍 About the Dataset

- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions (492 frauds)
- Features: `V1–V28` (PCA transformed), `Time`, `Amount`, `Class`
- Highly imbalanced, making it perfect for anomaly detection

---

## 🤖 Model Details

- **Algorithm**: `XGBoostClassifier`
- **Imbalance handled via**:
  - `scale_pos_weight` tuning
  - Precision/recall-based evaluation
- **Scaler**: `StandardScaler` for `Time` and `Amount`
- **Performance on full dataset**:
  - 🔹 Precision: **0.89**
  - 🔹 Recall: **0.79**
  - 🔹 F1 Score: **0.84**
  - 🔹 Fraud Detection Rate: **91.87%**

---

## 📁 Project Structure

XGBoost-FraudScanner/
├── app.py ← Streamlit application
├── xgboost_fraud_model_full.pkl ← Trained XGBoost model
├── scaler_full.pkl ← Scaler used for Time & Amount
├── requirements.txt ← Required Python packages
└── README.md ← You're here

yaml
Copy
Edit

---

## 📸 Screenshots

<details>
<summary>📊 Prediction Breakdown</summary>
  
![Prediction Pie Chart](https://via.placeholder.com/600x300?text=Prediction+Breakdown+Pie+Chart)
</details>

<details>
<summary>🎛️ Manual Transaction Input</summary>

![Manual Form](https://via.placeholder.com/600x300?text=Manual+Transaction+Input+Form)
</details>

<details>
<summary>📈 Fraud Detection Summary</summary>

![Bar Chart](https://via.placeholder.com/600x300?text=Bar+Chart+of+Actual+vs+Predicted)
</details>

---

## ▶️ Run Locally

```bash
git clone https://github.com/Yashdeep1546/XGBoost-FraudScanner.git
cd XGBoost-FraudScanner
pip install -r requirements.txt
streamlit run app.py
🛠 Tech Stack
Python 3.x

Streamlit

XGBoost

Scikit-learn

Pandas, NumPy

Matplotlib, Seaborn

🙋‍♂️ Author
Made with ❤️ by Yashdeep
B.Tech CS | Machine Learning Enthusiast | Aiming for 1 Cr 💸

🧠 Future Enhancements
✅ Download prediction results as CSV

🛡️ SHAP explainability for model transparency

⏱️ Time series view of fraud evolution

📦 Docker containerization & API exposure

🧪 Add more models (Isolation Forest, AutoEncoders)

🪪 License
This project is open-source under the MIT License.

yaml
Copy
Edit

---

## 🔥 Want to go further?

I can help you:
- Add auto-updating screenshots/GIFs
- Create badges (e.g., Python version, Streamlit live)
- Polish the GitHub description & topics for visibility

Let me know when you want that final resume-ready polish!
