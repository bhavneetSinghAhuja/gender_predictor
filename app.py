import gradio as gr
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "gender-predictor"

model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename="model.pkl"))
label_encoder = joblib.load(hf_hub_download(repo_id=REPO_ID, filename="label_encoder.pkl"))

FEATURES = ['height_cm', 'weight_kg', 'bmi']

def predict_gender(height_cm, weight_kg):
    if not (140 <= height_cm <= 210):
        return "Height must be between 140-210 cm", "N/A"
    if not (35 <= weight_kg <= 160):
        return "Weight must be between 35-160 kg", "N/A"
    
    bmi = weight_kg / ((height_cm / 100) ** 2)
    features = pd.DataFrame([[height_cm, weight_kg, bmi]], columns=FEATURES)
    
    prediction_encoded = model.predict(features)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities)
    
    return prediction, f"{confidence:.1%}"

def get_confidence_level(confidence_str):
    try:
        conf = float(confidence_str.replace("%", "")) / 100
        if conf > 0.9:
            return "HIGH"
        elif conf > 0.7:
            return "MODERATE"
        else:
            return "LOW"
    except:
        return "N/A"

with gr.Blocks(title="Gender Predictor", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # Gender Predictor
    Predict gender from height and weight using a GradientBoosting model (87.4% accuracy).
    """)
    
    with gr.Row():
        with gr.Column():
            height = gr.Number(label="Height (cm)", value=170, minimum=140, maximum=210, step=0.1)
            weight = gr.Number(label="Weight (kg)", value=70, minimum=35, maximum=160, step=0.1)
            predict_btn = gr.Button("Predict", variant="primary")
        
        with gr.Column():
            gender_output = gr.Textbox(label="Predicted Gender", interactive=False)
            confidence_output = gr.Textbox(label="Confidence", interactive=False)
    
    examples = gr.Examples(
        examples=[
            [180, 85],
            [162, 55],
            [175, 75],
            [165, 60],
            [190, 95],
            [155, 50],
        ],
        inputs=[height, weight],
        label="Example inputs"
    )
    
    gr.Markdown("""
    **Model Details:**
    - Algorithm: GradientBoosting Classifier
    - Features: height, weight, BMI
    - Accuracy: 87.4%
    - Training data: 10,000 synthetic samples
    """)
    
    predict_btn.click(
        fn=predict_gender,
        inputs=[height, weight],
        outputs=[gender_output, confidence_output]
    )

if __name__ == "__main__":
    app.launch()
