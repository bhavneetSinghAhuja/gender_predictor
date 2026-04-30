import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import os

FEATURES = ['height_cm', 'weight_kg', 'bmi']
MODEL_PATH = 'models/gender_predictor.pkl'
SCALER_PATH = 'models/scaler.pkl'

def load_data():
    train_df = pd.read_csv('data/train.csv')
    return train_df

def train_model():
    print("Loading training data...")
    df = load_data()
    
    X = df[FEATURES]
    y = df['gender']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("Training GradientBoosting model...")
    model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=42
    )
    
    cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    model.fit(X, y_encoded)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, SCALER_PATH)
    
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Feature importances:")
    for feature, importance in zip(FEATURES, model.feature_importances_):
        print(f"  {feature}: {importance:.4f}")
    
    return model, label_encoder

if __name__ == '__main__':
    train_model()
