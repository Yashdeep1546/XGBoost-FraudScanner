# Credit Card Fraud Detection Web App

This is a Streamlit web application for detecting fraudulent credit card transactions.

## Setup and Running

1. Install the required dependencies:
```
pip install streamlit pandas joblib scikit-learn xgboost
```

2. Run the application:
```
streamlit run app.py
```

3. Upload a CSV file with credit card transactions. The CSV should contain at least the following columns:
   - Time
   - Amount
   - (Other features used by the model)

## Files
- `app.py`: The main Streamlit application
- `xgboost_fraud_model_full.pkl`: The trained XGBoost model
- `scaler_full.pkl`: The scaler for preprocessing Time and Amount features

## Troubleshooting
If you encounter errors:
- Ensure all dependencies are installed
- Make sure the model files are in the same directory as app.py
- Check that your CSV file has the required columns 