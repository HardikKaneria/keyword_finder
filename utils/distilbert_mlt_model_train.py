import os
import pandas as pd
import numpy as np
import pickle
import glob
import gc
from datasets import Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score
import torch

# === Load Datasets ===
train_df = pd.read_csv("utils/train.csv")
val_df = pd.read_csv("utils/validation.csv")

# === Combine into model input ===
def create_input_text(row):
    return f"Keyword: {row['Keyword']} | AIDA Stage: {row['AIDAStage']}"

train_df["text"] = train_df.apply(create_input_text, axis=1)
val_df["text"] = val_df.apply(create_input_text, axis=1)

# === Label Encoding ===
label_columns = ["MediaFormat", "Platform", "ToneCluster", "MarketingIntent"]
encoders = {}

for label in label_columns:
    le = LabelEncoder()
    train_df[label] = le.fit_transform(train_df[label])
    val_df[label] = le.transform(val_df[label])
    encoders[label] = le

# Save encoders for future inference
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

# === Tokenizer ===
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# === Metric ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# === Resume from latest checkpoint if exists ===
def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

# === Training Function ===
def train_model(label_name, num_labels):
    print(f"\n🔁 Training model for {label_name} ({num_labels} classes)")
    
    classifier_dir = f"{label_name}_classifier"
    os.makedirs(classifier_dir, exist_ok=True)
    latest_ckpt = get_latest_checkpoint(classifier_dir)

    if latest_ckpt:
        print(f"📦 Resuming from checkpoint: {latest_ckpt}")
        model = DistilBertForSequenceClassification.from_pretrained(latest_ckpt, num_labels=num_labels)
    else:
        print("🆕 Starting training from base model.")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

    train_dataset = Dataset.from_pandas(train_df[["text", label_name]].rename(columns={label_name: "label"}))
    val_dataset = Dataset.from_pandas(val_df[["text", label_name]].rename(columns={label_name: "label"}))

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=classifier_dir,
        logging_dir=os.path.join(classifier_dir, "logs"),
        save_total_limit=2,  # Keep last 2 checkpoints
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        learning_rate=3e-5,
        no_cuda=True  # keep True because c4.4xlarge has no GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=latest_ckpt if latest_ckpt else None)

    model.cpu()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()
    gc.collect()

    return model

# === Train Models for Each Label ===
models = {}
for label in label_columns:
    model = train_model(label, len(encoders[label].classes_))
    models[label] = model

print("\n✅ All models trained (or resumed) and saved under respective directories.")
