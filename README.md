# Real-Time Fraud Detector Dashboard

A Streamlit app for real-time credit card fraud detection using a trained machine learning model.

## Features
- Upload CSV or enter transaction details manually
- Live fraud prediction with confidence threshold
- Sound alert for detected fraud
- Easy deployment to HuggingFace Spaces

## Local Setup
1. Clone the repo or download the files.
2. Place your trained `fraud_model.pkl` in the project folder.
3. (Optional) Add a sample `creditcard.csv` for demo/testing.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Run the app:
   ```bash
   streamlit run app.py
   ```

## Deploy to HuggingFace Spaces
1. Go to [HuggingFace Spaces](https://huggingface.co/spaces)
2. Create a new Space (choose Streamlit SDK)
3. Upload all project files
4. (Optional) Add `.huggingface.yml`:
   ```yaml
   sdk: streamlit
   app_file: app.py
   python_version: 3.11
   ```
5. Click Deploy!

## Credits
- Built with [Streamlit](https://streamlit.io/)
- Model: scikit-learn RandomForest (or your own) 