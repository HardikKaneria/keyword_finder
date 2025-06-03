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

train_df = pd.read_csv("utils/aida_train_clean.csv")
val_df = pd.read_csv("utils/aida_val_clean.csv")

def create_input_text(row):
    return f"Keyword: {row['Keyword']} | AIDA Stage: {row['AIDAStage']}"

train_df["text"] = train_df.apply(create_input_text, axis=1)
val_df["text"] = val_df.apply(create_input_text, axis=1)

label_columns = ["MediaFormat", "Platform", "ToneCluster", "MarketingIntent"]

encoders = {}
for label in label_columns:
    le = LabelEncoder()
    train_df[label] = le.fit_transform(train_df[label])
    val_df[label] = le.transform(val_df[label])
    encoders[label] = le

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

def get_latest_checkpoint(path):
    checkpoints = glob.glob(os.path.join(path, "checkpoint-*"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda x: int(x.split("-")[-1]))

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
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=100,
        learning_rate=2e-5,
        no_cuda=True  # Set to False if using GPU
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    model.cpu()
    torch.cuda.empty_cache()
    torch.mps.empty_cache()
    gc.collect()

    return model

models = {}
for label in label_columns:
    model = train_model(label, len(encoders[label].classes_))
    models[label] = model

print("\n✅ All models trained or resumed and saved under ...")
