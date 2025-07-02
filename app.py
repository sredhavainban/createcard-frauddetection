import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import json

# Load model and column names
def load_model():
    return joblib.load('fraud_model.pkl')

def load_scaler():
    return joblib.load('scaler.pkl')

def load_columns():
    try:
        with open('columns.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("columns.json not found. Please run train_model.py first to generate the required files.")
        st.stop()

# Load all required components
try:
    model = load_model()
    scaler = load_scaler()
    MODEL_FEATURES = load_columns()
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.error("Please ensure you have run train_model.py first and all files are present.")
    st.stop()

st.set_page_config(page_title="Fraud Detector Dashboard", layout="wide")
st.title("💳 Real-Time Fraud Detection Dashboard")

st.sidebar.header("Prediction Settings")
threshold = st.sidebar.slider("Fraud Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

st.write("Upload a CSV file or enter transaction details below:")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

sample_input = None
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.write("Sample of uploaded data:", df.head())
        sample_input = df
    except Exception as e:
        st.error(f"Error reading uploaded file: {str(e)}")
        sample_input = None
else:
    st.write("Or enter transaction details manually:")
    # Use the actual column names from training
    input_data = {}
    for col in MODEL_FEATURES:
        input_data[col] = st.number_input(f"{col}", value=0.0)
    sample_input = pd.DataFrame([input_data])

if 'summary' not in st.session_state:
    st.session_state['summary'] = {'total': 0, 'frauds': 0}

if st.button("Predict Fraud"):
    if sample_input is None:
        st.error("Please upload a file or enter transaction details first.")
    else:
        try:
            X = sample_input.copy()
            
            # Remove target column if present
            if 'Class' in X.columns:
                X = X.drop('Class', axis=1)
            
            # Check for missing columns
            missing_cols = [col for col in MODEL_FEATURES if col not in X.columns]
            if missing_cols:
                st.error(f"Missing columns for prediction: {missing_cols}")
                st.error(f"Required columns: {MODEL_FEATURES}")
            else:
                # Ensure columns are in the same order as during training
                X = X[MODEL_FEATURES]
                
                # Debug information
                st.write(f"Input data shape: {X.shape}")
                st.write(f"Expected columns: {MODEL_FEATURES}")
                st.write(f"Actual columns: {list(X.columns)}")
                
                # Transform the data
                X_scaled = scaler.transform(X)
                
                # Make predictions
                proba = model.predict_proba(X_scaled)[:,1]
                prediction = (proba > threshold).astype(int)
                
                frauds = int(prediction.sum())
                total = len(prediction)
                st.session_state['summary']['total'] += total
                st.session_state['summary']['frauds'] += frauds
                
                # Display results
                for i, (pred, conf) in enumerate(zip(prediction, proba)):
                    status = '🚨 YES' if pred else '✅ No'
                    color = 'red' if pred else 'green'
                    st.markdown(f"**Transaction {i+1}:** Fraud: <span style='color:{color}'>{status}</span> (Confidence: {conf:.2f})", unsafe_allow_html=True)
                    
                    if pred:
                        # Play sound alert using base64-encoded audio
                        try:
                            sound_file = open("alert.mp3", "rb").read()
                            b64 = base64.b64encode(sound_file).decode()
                            md = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
                            st.markdown(md, unsafe_allow_html=True)
                        except Exception:
                            pass
                
                st.success(f"Prediction completed! Found {frauds} fraudulent transactions out of {total} total.")
                
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.error("Please check your input data format and try again.")

# Summary Report Section
st.markdown("---")
st.header("📊 Summary Report")
total = st.session_state['summary']['total']
frauds = st.session_state['summary']['frauds']
fraud_rate = (frauds / total) * 100 if total > 0 else 0
st.write(f"**Total transactions processed:** {total}")
st.write(f"**Fraudulent transactions detected:** {frauds}")
st.write(f"**Fraud rate:** {fraud_rate:.2f}%")

# Model Information Section
st.markdown("---")
st.header("🔧 Model Information")
st.write(f"**Model type:** Random Forest Classifier")
st.write(f"**Number of features:** {len(MODEL_FEATURES)}")
st.write(f"**Features used:** {', '.join(MODEL_FEATURES[:5])}... (and {len(MODEL_FEATURES)-5} more)")

# Sample data creation (optional)
st.markdown("---")
st.header("📁 Data Management")
if st.button("Create Sample Data File"):
    try:
        df = pd.read_csv("creditcard.csv")
        sample_df = df.sample(500)
        sample_df.to_csv("creditcard_sample.csv", index=False)
        st.success("Sample data file created successfully!")
        st.write(f"Created creditcard_sample.csv with {len(sample_df)} records")
    except FileNotFoundError:
        st.warning("creditcard.csv not found. Please ensure the original data file is present.")
    except Exception as e:
        st.error(f"Error creating sample file: {str(e)}")

# Instructions
st.markdown("---")
st.header("📖 Instructions")
st.write("""
1. **Upload a CSV file** with transaction data, or
2. **Enter transaction details manually** using the input fields above
3. **Click 'Predict Fraud'** to analyze the transactions
4. **Adjust the threshold** in the sidebar to control sensitivity
5. **View results** and summary statistics below
""")

st.write("**Note:** The uploaded CSV should contain the same columns as the training data (excluding 'Class' if present).")