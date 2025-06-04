import os
import sqlite3
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
WEBSITE_FILE = os.path.join(DATA_DIR, "website_data.db")

def match_urls_to_aida(
    df,
    depth_penalty=0.05,
    progress_bar=None,
    status_callback=None
):
    try:
        if status_callback:
            status_callback.text("🔄 Step 1 – Loading website data...")

        # Load website data
        conn_web = sqlite3.connect(WEBSITE_FILE)
        website_df = pd.read_sql("SELECT * FROM pages", conn_web)
        conn_web.close()

        if progress_bar:
            progress_bar.progress(0.2)
        if status_callback:
            status_callback.text("🧠 Step 2 – Preparing website content for similarity scoring...")

        # Combine website text
        website_df["combined_text"] = (
            website_df["title"].fillna("") + " " +
            website_df["meta_description"].fillna("") + " " +
            website_df["h1"].fillna("") + " " +
            website_df["h2"].fillna("") + " " +
            website_df["h3"].fillna("") + " " +
            website_df["visible_text"].fillna("")
        )

        # Calculate URL depth
        website_df["url_depth"] = website_df["url"].apply(lambda x: len(x.strip("/").split("/")))

        if progress_bar:
            progress_bar.progress(0.4)
        if status_callback:
            status_callback.text("📊 Step 3 – Vectorizing keywords and website content...")

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        keyword_vecs = vectorizer.fit_transform(df["Keyword"].fillna(""))
        page_vecs = vectorizer.transform(website_df["combined_text"])

        if progress_bar:
            progress_bar.progress(0.6)
        if status_callback:
            status_callback.text("⚖️ Step 4 – Computing similarities and applying depth penalty...")

        # Cosine similarity
        similarity_matrix = cosine_similarity(keyword_vecs, page_vecs)

        max_depth = website_df["url_depth"].max()
        website_df["normalized_depth"] = website_df["url_depth"] / max_depth


        penalty_matrix = np.tile(
            depth_penalty * website_df["normalized_depth"].values, 
            (len(df), 1)  
        )

        adjusted_scores = similarity_matrix - penalty_matrix

        if progress_bar:
            progress_bar.progress(0.8)
        if status_callback:
            status_callback.text("🔗 Step 5 – Selecting best URLs...")

        # Select best URL per keyword
        best_indices = adjusted_scores.argmax(axis=1)
        best_urls = [website_df.iloc[i]["url"] for i in best_indices]

        df["URL Target"] = best_urls

        if progress_bar:
            progress_bar.progress(1.0)
        if status_callback:
            status_callback.text("✅ URL matching complete!")

        return df

    except Exception as e:
        if status_callback:
            status_callback.text("❌ Error during URL matching.")
        st.error(f"❌ Error during URL matching: {e}")
        return df  # return original df to avoid pipeline break