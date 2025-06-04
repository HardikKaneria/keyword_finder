import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import pickle
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import numpy as np

# === Load tokenizer ===
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))  # 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
# === Load trained models ===
media_model = DistilBertForSequenceClassification.from_pretrained("MediaFormat_classifier/checkpoint-2211")
platform_model = DistilBertForSequenceClassification.from_pretrained("Platform_classifier/checkpoint-2211")
tone_model = DistilBertForSequenceClassification.from_pretrained("ToneCluster_classifier/checkpoint-2211")
marketing_model = DistilBertForSequenceClassification.from_pretrained("MarketingIntent_classifier/checkpoint-2211")


media_model.eval()
platform_model.eval()
tone_model.eval()
marketing_model.eval()

# === Load encoders ===
with open("label_encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

# === Prediction helper ===
def predict_label(model, encoder, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    pred_id = torch.argmax(logits, dim=1).item()
    return encoder.inverse_transform([pred_id])[0]

# === Generate prediction ===
def generate_predictions(df):
    results = []

    for _, row in df.iterrows():
        keyword = row["Keyword"]
        stage = row["Stage"]
        ConfidenceScores = row["ConfidenceScores"]

        input_text = f"Keyword: {keyword} | AIDA Stage: {stage}"
        media = predict_label(media_model, encoders["MediaFormat"], input_text)
        platform = predict_label(platform_model, encoders["Platform"], input_text)
        tone = predict_label(tone_model, encoders["ToneCluster"], input_text)
        marketing_intent = predict_label(marketing_model, encoders["MarketingIntent"], input_text)
        
        results.append({
                "Keyword": keyword,
                "Stage": stage,
                "Media": media,
                "Platform": platform,
                "Tone": tone,
                "Marketing_Intent": marketing_intent,
                "ConfidenceScores": ConfidenceScores
            })
    
    return pd.DataFrame(results)