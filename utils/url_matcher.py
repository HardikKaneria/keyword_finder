import os
import re
import sqlite3
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Paths setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
WEBSITE_FILE = os.path.join(DATA_DIR, "website_data.db")


def match_urls_to_aida(
    df: pd.DataFrame,
    depth_bonus: float = 0.1,
    ngram_range=(1, 2),
    max_features=5000,
    min_token_len: int = 3,
    progress_bar=None,
    status_callback=None
) -> pd.DataFrame:
    """
    For each keyword, pick only pages whose visible_text contains
    at least one of the keyword’s longer tokens—then TF-IDF + depth_bonus
    on that small candidate set.  Falls back to ALL pages if no candidate.
    """
    df = df.copy()

    # 1) load and prep pages
    if status_callback: status_callback.text("🔄 Loading pages…")
    with sqlite3.connect(WEBSITE_FILE) as conn:
        pages = pd.read_sql("SELECT url, visible_text FROM pages", conn)

    if pages.empty:
        st.error("No pages found in the DB.")
        return df

    # compute depth_norm once
    pages["url_depth"] = pages["url"].str.strip("/").str.count("/") + 1
    dmin, dmax = pages["url_depth"].min(), pages["url_depth"].max()
    pages["depth_norm"] = (pages["url_depth"] - dmin) / (dmax - dmin + 1e-9)

    # 2) vectorize all visible_text
    if status_callback: status_callback.text("📊 Training TF-IDF…")
    texts = pages["visible_text"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    vectorizer.fit(texts)
    page_matrix = vectorizer.transform(texts)

    # 3) for each keyword, do candidate filtering + similarity
    urls = []
    total = len(df)
    for i, kw in enumerate(df["Keyword"].fillna("").astype(str)):
        if progress_bar:
            progress_bar.progress(i / total)
        if status_callback:
            status_callback.text(f"🔎 Matching “{kw}” ({i+1}/{total})…")

        # a) extract “long” tokens from keyword
        tokens = [tok for tok in re.split(r"\W+", kw.lower()) if len(tok) >= min_token_len]
        if tokens:
            pat = "|".join(re.escape(tok) for tok in tokens)
            mask = pages["visible_text"].str.contains(pat, case=False, regex=True)
            cand_idxs = np.where(mask)[0]
        else:
            cand_idxs = np.arange(len(pages))

        # b) compute sim on candidates (or fallback to all)
        kw_vec = vectorizer.transform([kw])
        if 0 < len(cand_idxs) < len(pages):
            submatrix = page_matrix[cand_idxs]
            sim = cosine_similarity(kw_vec, submatrix)[0]
            bonus = depth_bonus * pages.loc[cand_idxs, "depth_norm"].values
            sim += bonus
            best_local = np.argmax(sim)
            best_idx = cand_idxs[best_local]
        else:
            sim = cosine_similarity(kw_vec, page_matrix)[0]
            bonus = depth_bonus * pages["depth_norm"].values
            sim += bonus
            best_idx = int(np.argmax(sim))

        urls.append(pages.iloc[best_idx]["url"])

    df["URL Target"] = urls
    if progress_bar:
        progress_bar.progress(1.0)
    if status_callback:
        status_callback.text("✅ Matching complete!")
    return df
