import hashlib, os, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
import sqlite3
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.data_store import get_combined_website_texts

# === Setup ===
model = SentenceTransformer("intfloat/e5-small-v2")

# === Utility: Split content into chunks ===
def chunk_text(text, max_tokens=100):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

# === Utility: Cache embeddings ===
EMBEDDING_CACHE_DIR = "embedding_cache"
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

def get_embedding_cache_path(text):
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    return os.path.join(EMBEDDING_CACHE_DIR, f"{key}.pkl")

def get_or_compute_embedding(text, model):
    path = get_embedding_cache_path(text)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    emb = model.encode(text, convert_to_tensor=True)
    with open(path, "wb") as f:
        pickle.dump(emb, f)
    return emb

# === Main scorer ===
def calculate_chunked_keyword_scores(df_scored, df_pages, scale_0_to_10=True, max_tokens=100):
    results = []
    for idx, row in tqdm(df_scored.iterrows(), total=len(df_scored)):
        keyword = f"query: {row['Keyword']}"
        keyword_embed = model.encode(keyword, convert_to_tensor=True)

        best_score = -1
        for page_text in df_pages["visible_text"].fillna(""):
            for chunk in chunk_text(page_text, max_tokens):
                passage = f"passage: {chunk}"
                passage_embed = get_or_compute_embedding(passage, model)
                score = util.cos_sim(keyword_embed, passage_embed).item()
                if score > best_score:
                    best_score = score

        final_score = round(best_score * 10, 2) if scale_0_to_10 else best_score
        results.append((row["Keyword"], final_score))

    return pd.DataFrame(results, columns=["Keyword", "content_match_score"])

def main():
    DB_FILE = "data/keyword_data.db"
    conn = sqlite3.connect(DB_FILE)
    
    # Load keywords table (assuming it has 'Keyword' column)
    df_keywords = pd.read_sql("SELECT Keyword FROM keyword_ranked", conn)

    # Load page text
    df_website = get_combined_website_texts()
    df_pages = pd.DataFrame({
        "visible_text": df_website["visible_text"].tolist()
    })

    # Compute content match scores
    df_scores = calculate_chunked_keyword_scores(df_keywords, df_pages)

    # Store back scores in 'keywords' table
    cursor = conn.cursor()
    for _, row in df_scores.iterrows():
        cursor.execute(
            "UPDATE keyword_ranked SET content_match_score = ? WHERE Keyword = ?",
            (row["content_match_score"], row["Keyword"])
        )

    conn.commit()
    conn.close()
    print("✅ Content match scores updated successfully.")

if __name__ == "__main__":
    main()
