import pickle
import os
import sys
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))  # 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
# === Load trained models ===
media_model = DistilBertForSequenceClassification.from_pretrained("MediaFormat_classifier/checkpoint-2211")
platform_model = DistilBertForSequenceClassification.from_pretrained("Platform_classifier/checkpoint-2211")
tone_model = DistilBertForSequenceClassification.from_pretrained("ToneCluster_classifier/checkpoint-2211")
marketing_model = DistilBertForSequenceClassification.from_pretrained("MarketingIntent_classifier/checkpoint-2211")

encoders = {
    "MediaFormat": media_label_encoder,
    "Platform": platform_label_encoder,
    "ToneCluster": tone_label_encoder,
    "MarketingIntent": marketing_label_encoder,
}

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
