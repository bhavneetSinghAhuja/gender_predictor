from huggingface_hub import HfApi, ModelCard
from huggingface_hub import login
import os
import joblib

REPO_NAME = "gender-predictor"

def publish_to_hub(repo_id=None):
    print("=== Gender Predictor - Hugging Face Publisher ===\n")
    
    token = os.getenv("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face API token: ").strip()
    login(token)
    
    api = HfApi()
    
    if not repo_id:
        username = api.whoami(token)["name"]
        repo_id = f"{username}/{REPO_NAME}"
    
    print(f"Creating/uploading to repository: {repo_id}\n")
    
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        print(f"Repository ready: {repo_id}")
    except Exception as e:
        print(f"Repo error: {e}")
    
    model_card = """---
license: mit
tags:
- gender-prediction
- scikit-learn
- gradient-boosting
- tabular
pipeline_tag: tabular-classification
---

# Gender Predictor

A GradientBoosting model that predicts gender from height and weight inputs.

## Model Details
- **Algorithm**: GradientBoosting Classifier
- **Features**: height_cm, weight_kg, bmi
- **Training Data**: 10,000 synthetic samples
- **Accuracy**: 87.4%

## Usage

```python
from huggingface_hub import hf_hub_download
import joblib
import pandas as pd
import numpy as np

model = joblib.load(hf_hub_download(repo_id="{repo_id}", filename="model.pkl"))
le = joblib.load(hf_hub_download(repo_id="{repo_id}", filename="label_encoder.pkl"))

def predict(height_cm, weight_kg):
    bmi = weight_kg / ((height_cm / 100) ** 2)
    features = pd.DataFrame([[height_cm, weight_kg, bmi]], columns=['height_cm', 'weight_kg', 'bmi'])
    pred = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    return le.inverse_transform([pred])[0], confidence

gender, confidence = predict(175, 75)
print(f"Gender: {{gender}}, Confidence: {{confidence:.1%}}")
```

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87.4% |
| Precision | 87.4% |
| F1-Score | 87.4% |

## Interactive Demo

Check out the [Gradio Space](https://huggingface.co/spaces/{username}/gender-predictor) for an interactive demo!
"""
    
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model"
    )
    print("Uploaded README.md (model card)")
    
    api.upload_file(
        path_or_fileobj="models/gender_predictor.pkl",
        path_in_repo="model.pkl",
        repo_id=repo_id,
        repo_type="model"
    )
    print("Uploaded model.pkl")
    
    api.upload_file(
        path_or_fileobj="models/scaler.pkl",
        path_in_repo="label_encoder.pkl",
        repo_id=repo_id,
        repo_type="model"
    )
    print("Uploaded label_encoder.pkl")
    
    print(f"\nModel published: https://huggingface.co/{repo_id}")
    return repo_id

if __name__ == "__main__":
    publish_to_hub()
