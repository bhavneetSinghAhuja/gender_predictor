import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

FEATURES = ['height_cm', 'weight_kg', 'bmi']
MODEL_PATH = 'models/gender_predictor.pkl'
LABEL_ENCODER_PATH = 'models/scaler.pkl'

def evaluate_model():
    print("Loading model and test data...")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    test_df = pd.read_csv('data/test.csv')
    
    X_test = test_df[FEATURES]
    y_test = test_df['gender']
    y_test_encoded = label_encoder.transform(y_test)
    
    y_pred_encoded = model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    
    y_prob = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"              Predicted")
    print(f"              Male  Female")
    print(f"Actual Male   {cm[0][0]:>5}  {cm[0][1]:>5}")
    print(f"       Female {cm[1][0]:>5}  {cm[1][1]:>5}")
    
    male_precision = precision_score(y_test, y_pred, pos_label='Male')
    female_precision = precision_score(y_test, y_pred, pos_label='Female')
    print(f"\nMale Precision:   {male_precision:.4f}")
    print(f"Female Precision: {female_precision:.4f}")
    
    return accuracy, precision, recall, f1

if __name__ == '__main__':
    evaluate_model()
