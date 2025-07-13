import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
            
            # Pie chart of prediction results
            st.write("### 📊 Prediction Breakdown:")
            
            fraud_counts = df['Predictions'].value_counts()
            # Ensure labels are strings for matplotlib pie chart
            labels = [str(label) for label in fraud_counts.index.tolist()]
            sizes = fraud_counts.values.tolist()
            
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'salmon'], startangle=90)
            ax1.axis('equal')  # Equal aspect ratio for pie chart
            st.pyplot(fig1)
            
            if actual_labels is not None:
                actual_labels_mapped = actual_labels.map(lambda x: "Not Fraud" if x == 0 else "Fraud")
                st.write(f"Total Actual Fraud: {int(actual_labels.sum())}")
                st.write(f"Total Predicted Fraud: {int(predictions.sum())}")
                true_positives = int(((predictions == 1) & (actual_labels == 1)).sum())
                st.write(f"True Positives (Correctly Detected Frauds): {true_positives}")
                fraud_recall = true_positives / int(actual_labels.sum()) if int(actual_labels.sum()) > 0 else 0
                st.write(f"Fraud Recall (Detection Rate): {fraud_recall:.2%}")
                
                # Bar chart: Actual vs Predicted vs Correct
                st.write("### 📊 Fraud Detection Summary:")
                
                chart_data = {
                    'Category': ['Actual Frauds', 'Predicted Frauds', 'True Positives'],
                    'Count': [int(actual_labels.sum()), int(predictions.sum()), true_positives]
                }
                
                chart_df = pd.DataFrame(chart_data)
                
                fig2, ax2 = plt.subplots()
                sns.barplot(x='Category', y='Count', data=chart_df, palette='pastel', ax=ax2)
                st.pyplot(fig2)
        else:
            st.error("CSV file must contain 'Time' and 'Amount' columns")
    except Exception as e:
        st.error(f"Error processing file: {e}")
        
st.markdown("---")
st.write("### 🔍 Test a Single Transaction")


with st.expander("Enter Transaction Details Manually "):
    # Default to Not Fraud example
    example_type = st.radio("Choose Example Type:", ["Legit (Not Fraud)", "Fraud (Malicious)"])

    if example_type == "Legit (Not Fraud)":
        default_time = 10000.0
        default_amount = 149.62
        default_vs = [-1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388,
                      0.239599, 0.098698, 0.363787, 0.090794, -0.5516, -0.617801,
                      -0.99139, -0.311169, 1.468177, -0.4704, 0.207971, 0.025791,
                      0.403993, 0.251412, -0.018307, 0.277838, -0.110474, 0.066928,
                      0.128539, -0.189115, 0.133558, -0.021053]
    else:
    # Confirmed fraud example
        default_time = 406.0
        default_amount = 0.0  # Amount not included, set to 0 or guess based on average fraud
        default_vs = [-2.312226542, 1.951992011, -1.609850732, 3.997905588, -0.522187865,
                  -1.426545319, -2.537387306, 1.391657248, -2.770089277, -2.772272145,
                  3.202033207, -2.899907388, -0.595221881, -4.289253782, 0.38972412,
                  -1.14074718, -2.830055675, -0.016822468, 0.416955705, 0.126910559,
                  0.517232371, -0.035049369, -0.465211076, 0.320198199, 0.044519167,
                  0.177839798, 0.261145003, -0.143275875]


    time = st.number_input("Time", value=default_time, min_value=0.0)
    amount = st.number_input("Amount", value=default_amount, min_value=0.0)

    v_features = []
    for i in range(1, 29):
        v = st.number_input(f"V{i}", value=default_vs[i-1], format="%.6f")
        v_features.append(v)

    # Move this above the button
    threshold = st.slider("Fraud Threshold", 0.0, 1.0, 0.5, 0.01)
    st.write(f"Selected Threshold: {threshold}")

if st.button("Predict Transaction"):
    input_df = pd.DataFrame([[time] + v_features + [amount]],
                            columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])  # type: ignore

    # Check model and scaler are loaded
    if model is None or scaler is None:
        st.error("Model or scaler not loaded. Please check your setup.")
    else:
        # Scale Time and Amount
        input_df[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])

        # Predict
        fraud_proba = model.predict_proba(input_df)[0][1]
        pred = 1 if fraud_proba >= threshold else 0

        # Display result with correct confidence
        if pred == 1:
            st.error(f"⚠️ This transaction is predicted as **FRAUD**.\n\n(Confidence: {fraud_proba:.2%})")
        else:
            not_fraud_proba = 1 - fraud_proba
            st.success(f"✅ This transaction is predicted as **Not Fraud**.\n\n(Confidence: {not_fraud_proba:.2%})")
        
        # Show the raw probabilities for clarity
        st.info(f"Raw Probabilities: Not Fraud: {1-fraud_proba:.2%}, Fraud: {fraud_proba:.2%}")
