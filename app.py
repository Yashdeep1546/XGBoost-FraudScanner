import streamlit as st
import joblib
import pandas as pd
import os

def load_model():
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(current_dir, "xgboost_fraud_model_full.pkl"))
        scaler = joblib.load(os.path.join(current_dir, "scaler_full.pkl"))
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, scaler = load_model()

st.title("Credit Card Fraud Detection")
st.write("Upload a CSV file containing credit card transactions to check for fraud.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file and model is not None and scaler is not None:
    try:
        df = pd.read_csv(uploaded_file)
        actual_labels = None
        if 'Class' in df.columns:
            actual_labels = df['Class']
            df = df.drop('Class', axis=1)
        
        # Check if required columns exist
        if 'Time' in df.columns and 'Amount' in df.columns:
            # Create a copy to avoid SettingWithCopyWarning
            df_scaled = df.copy()
            df_scaled[['Time','Amount']] = scaler.transform(df[['Time','Amount']])
            predictions = model.predict(df_scaled)
            df['Predictions'] = predictions
            df['Predictions'] = df['Predictions'].apply(lambda x: 'Not Fraud' if x == 0 else 'Fraud')
            st.write("### Prediction Results:")
            st.dataframe(df)
            st.write("### Prediction Summary:")
            st.write(df['Predictions'].value_counts())
            
            if actual_labels is not None:
                actual_labels_mapped = actual_labels.map(lambda x: "Not Fraud" if x == 0 else "Fraud")
                st.write(f"Total Actual Fraud: {sum(actual_labels==1)}")
                st.write(f"Total Predicted Fraud: {sum(predictions==1)}")
                true_positives = sum((predictions == 1) & (actual_labels == 1))
                st.write(f"True Positives (Correctly Detected Frauds): {true_positives}")
                fraud_recall = true_positives / sum(actual_labels == 1)
                st.write(f"Fraud Recall (Detection Rate): {fraud_recall:.2%}")
        else:
            st.error("CSV file must contain 'Time' and 'Amount' columns")
    except Exception as e:
        st.error(f"Error processing file: {e}")