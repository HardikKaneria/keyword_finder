import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))  # 
BOOK_PATH = os.path.join(PROJECT_ROOT, "streamlit_ui/book.csv")

def perform_sentiment_analysis(keywords, progress_bar=None, status_callback=None):
    sia = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    
    results = []
    total = len(keywords)
    
    if status_callback:
        status_callback.text(f"🧠 Starting sentiment analysis for {total} keywords...")

    for i, kw in enumerate(tqdm(keywords, disable=progress_bar is None)):
        enriched_kw = f"I am reflecting on the topic of {kw} and how it makes people feel."
        result = sia(enriched_kw)[0]
        stars = int(result["label"].split()[0])  
        confidence = round(result["score"], 4)

        result_data = {
            "Keyword": kw,
            "Sentiment Score": stars,
            "Sentiment Confidence": confidence
        }
        results.append(result_data)

        # Show live update
        if status_callback:
            status_callback.text(f"[{i+1}/{total}] ✅ {kw}: {stars} stars ({confidence})")

        if progress_bar:
            progress_bar.progress((i + 1) / total)

    if status_callback:
        status_callback.text("✅ Sentiment analysis complete.")

    return pd.DataFrame(results)

def calculate_keyword_scores(df, progress_bar=None, status_callback=None):
    def normalize(series, scale=1):
        min_val, max_val = series.min(), series.max()
        return ((series - min_val) / (max_val - min_val) * scale).fillna(0) if max_val > min_val else 0

    if status_callback:
        status_callback.text("📊 Normalizing volume and CPC...")

    df["volume_score"] = normalize(df["Avg Monthly Searches"], scale=40)
    df["cpc_score"] = normalize(df["High CPC (USD)"], scale=30)

    if progress_bar:
        progress_bar.progress(0.4)

    if status_callback:
        status_callback.text("📊 Scoring competition and sentiment...")

    comp_map = {"LOW": 1, "MEDIUM": 0.5, "HIGH": 0}
    df["competition_score"] = df["Competition"].str.upper().map(comp_map).fillna(0) * 20
    df["sentiment_score_scaled"] = ((df["Sentiment Score"].clip(-1, 1) + 1) / 2 * 10).fillna(0)

    if progress_bar:
        progress_bar.progress(0.8)

    df["keyword_score"] = (
        df["volume_score"] * 0.30 +
        df["cpc_score"] * 0.30 +
        df["competition_score"] * 0.2 +
        df["sentiment_score_scaled"] * 0.1 +
        df["content_match_score"].fillna(0) * 0.1
    ).round(2)

    if status_callback:
        status_callback.text("✅ Final keyword scoring done.")
    if progress_bar:
        progress_bar.progress(1.0)

    return df