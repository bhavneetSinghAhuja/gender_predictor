# Gender Predictor

A machine learning model that predicts gender from height and weight inputs using GradientBoosting classifier.

## Quick Start

### Install Dependencies
```bash
.venv/bin/pip install -r requirements.txt
```

### Command Line Usage

**Predict gender from height and weight:**
```bash
.venv/bin/python main.py predict --height 180 --weight 85
```

Output:
```
Predicted Gender: Male
Confidence: 99.6%
Confidence Level: HIGH
```

**JSON output (for API/integration):**
```bash
.venv/bin/python main.py predict --height 162 --weight 58 --json
```

Output:
```json
{"gender": "Female", "confidence": 0.9644, "height_cm": 162.0, "weight_kg": 58.0}
```

### All Commands

| Command | Description |
|---------|-------------|
| `main.py predict --height H --weight W` | Predict gender |
| `main.py predict --height H --weight W --json` | JSON output |
| `main.py train` | Generate data & train model |
| `main.py evaluate` | Evaluate on test set |
| `main.py interactive` | Interactive prompt mode |

## Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87.4% |
| Precision | 87.4% |
| Male Precision | 87.9% |
| Female Precision | 86.9% |

## Project Structure

```
gender_predictor/
├── main.py              # CLI & interactive interface
├── data_generator.py    # Synthetic data generation
├── model.py             # Model training
├── evaluate.py          # Validation metrics
├── app.py               # Gradio Space app
├── publish.py           # Publish to HF (Model + Space)
├── publish_to_hub.py    # Upload model to HF Hub
├── create_space.py      # Deploy Gradio Space
├── requirements.txt     # Dependencies
├── space_requirements.txt  # Space dependencies
├── README.md            # Documentation
├── data/                # Training/test datasets
└── models/              # Saved model files

## Publish to Hugging Face

### Get API Token
1. Go to https://huggingface.co/settings/tokens
2. Create a new token (write access)
3. Set environment variable: `export HF_TOKEN="your-token-here"`

### Publish Model + Space (One Command)
```bash
.venv/bin/pip install huggingface_hub gradio
.venv/bin/python publish.py
```

### Or Publish Separately
```bash
# Upload model to Hub only
.venv/bin/python publish_to_hub.py

# Deploy Space only
.venv/bin/python create_space.py
```

This will:
- Upload model and label encoder to Hugging Face Model Hub
- Deploy interactive Gradio Space demo
- Generate model card with usage examples
```
