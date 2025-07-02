import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
import json
import os

def train_fraud_detection_model():
    """
    Train a fraud detection model and save all necessary components
    """
    print("Loading data...")
    
    # Load your data
    try:
        df = pd.read_csv('creditcard.csv')
        print(f"Data loaded successfully! Shape: {df.shape}")
    except FileNotFoundError:
        print("creditcard.csv not found. Please add it to the folder.")
        return False
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return False

    # Prepare features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    # Save column names for later use
    feature_columns = list(X.columns)
    print(f"Feature columns: {feature_columns}")
    
    # Create a sample file for testing/demo
    try:
        sample_df = df.sample(500, random_state=42)
        sample_df.to_csv('creditcard_sample.csv', index=False)
        print("Sample data file created: creditcard_sample.csv")
    except Exception as e:
        print(f"Warning: Could not create sample file: {str(e)}")

    # Scale the data
    print("Scaling data...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Data scaling completed!")

    # Train/test split
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # Train model on scaled data
    print("Training Random Forest model...")
    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        n_jobs=-1,  # Use all available cores
        class_weight='balanced'  # Handle imbalanced classes
    )
    clf.fit(X_train, y_train)
    print("Model training completed!")

    # Evaluate model
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Calculate additional metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Save model and scaler
    print("\nSaving model components...")
    try:
        dump(clf, 'fraud_model.pkl')
        dump(scaler, 'scaler.pkl')
        
        # Save column names
        with open('columns.json', 'w') as f:
            json.dump(feature_columns, f)
        
        print("✅ All files saved successfully!")
        print("   - fraud_model.pkl (trained model)")
        print("   - scaler.pkl (data scaler)")
        print("   - columns.json (feature column names)")
        print("   - creditcard_sample.csv (sample data)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving files: {str(e)}")
        return False

def main():
    """
    Main function to run the training process
    """
    print("="*60)
    print("FRAUD DETECTION MODEL TRAINING")
    print("="*60)
    
    success = train_fraud_detection_model()
    
    if success:
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nYou can now run the dashboard with:")
        print("streamlit run app.py")
        print("\nOr use the model in your own scripts!")
    else:
        print("\n" + "="*60)
        print("❌ TRAINING FAILED!")
        print("="*60)
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()