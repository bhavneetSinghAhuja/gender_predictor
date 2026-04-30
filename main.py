import os
import sys
import argparse
import joblib
import pandas as pd
from data_generator import generate_synthetic_data, save_datasets
from model import train_model, FEATURES, MODEL_PATH
from evaluate import evaluate_model

def predict_gender(height_cm, weight_kg, model, label_encoder):
    bmi = weight_kg / ((height_cm / 100) ** 2)
    features = pd.DataFrame([[height_cm, weight_kg, bmi]], columns=FEATURES)
    
    prediction_encoded = model.predict(features)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    return prediction, confidence

def cli_predict(args):
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Building pipeline first...")
        os.makedirs('data', exist_ok=True)
        df = generate_synthetic_data(n_samples=10000)
        save_datasets(df)
        train_model()
    
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(MODEL_PATH.replace('gender_predictor.pkl', 'scaler.pkl'))
    
    prediction, confidence = predict_gender(args.height, args.weight, model, label_encoder)
    
    if args.json:
        import json
        result = {
            "gender": prediction,
            "confidence": round(float(confidence), 4),
            "height_cm": args.height,
            "weight_kg": args.weight
        }
        print(json.dumps(result))
    else:
        print(f"\nPredicted Gender: {prediction}")
        print(f"Confidence: {confidence:.1%}")
        if confidence > 0.9:
            print("Confidence Level: HIGH")
        elif confidence > 0.7:
            print("Confidence Level: MODERATE")
        else:
            print("Confidence Level: LOW (ambiguous case)")

def interactive_mode():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Building pipeline...")
        print("\nStep 1: Generating synthetic data...")
        df = generate_synthetic_data(n_samples=10000)
        save_datasets(df)
        
        print("\nStep 2: Training model...")
        train_model()
        
        print("\nStep 3: Evaluating model...")
        evaluate_model()
    
    print("\nLoading model...")
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(MODEL_PATH.replace('gender_predictor.pkl', 'scaler.pkl'))
    
    print("\n" + "="*50)
    print("GENDER PREDICTOR")
    print("="*50)
    print("Enter height (cm) and weight (kg) to predict gender.")
    print("Type 'quit' or 'q' to exit.\n")
    
    while True:
        try:
            height_input = input("Enter height in cm (or 'q' to quit): ").strip()
            if height_input.lower() in ['q', 'quit', 'exit']:
                print("Goodbye!")
                break
            
            height_cm = float(height_input)
            
            weight_input = input("Enter weight in kg: ").strip()
            weight_kg = float(weight_input)
            
            if not (140 <= height_cm <= 210):
                print("Height should be between 140-210 cm. Please try again.\n")
                continue
            
            if not (35 <= weight_kg <= 160):
                print("Weight should be between 35-160 kg. Please try again.\n")
                continue
            
            prediction, confidence = predict_gender(height_cm, weight_kg, model, label_encoder)
            
            print(f"\nPredicted Gender: {prediction}")
            print(f"Confidence: {confidence:.1%}")
            if confidence > 0.9:
                print("Confidence Level: HIGH")
            elif confidence > 0.7:
                print("Confidence Level: MODERATE")
            else:
                print("Confidence Level: LOW (ambiguous case)")
            print("-" * 30)
            
        except ValueError:
            print("Invalid input. Please enter numeric values.\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gender Predictor - Predict gender from height and weight')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    predict_parser = subparsers.add_parser('predict', help='Predict gender from height and weight')
    predict_parser.add_argument('--height', type=float, required=True, help='Height in centimeters (140-210)')
    predict_parser.add_argument('--weight', type=float, required=True, help='Weight in kilograms (35-160)')
    predict_parser.add_argument('--json', action='store_true', help='Output as JSON')
    predict_parser.set_defaults(func=cli_predict)
    
    train_parser = subparsers.add_parser('train', help='Generate data and train model')
    train_parser.add_argument('--samples', type=int, default=10000, help='Number of synthetic samples')
    train_parser.set_defaults(func=None)
    
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test set')
    eval_parser.set_defaults(func=None)
    
    interactive_parser = subparsers.add_parser('interactive', help='Run interactive mode')
    interactive_parser.set_defaults(func=None)
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        args.func(args)
    elif args.command == 'train':
        os.makedirs('data', exist_ok=True)
        print("Generating synthetic data...")
        df = generate_synthetic_data(n_samples=args.samples)
        save_datasets(df)
        print("\nTraining model...")
        train_model()
    elif args.command == 'evaluate':
        evaluate_model()
    elif args.command == 'interactive':
        interactive_mode()
    else:
        interactive_mode()
