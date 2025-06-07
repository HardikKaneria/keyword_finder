import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import pipeline
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from joblib import Parallel, delayed

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

def compute_semantic_scores_batch(keywords_batch, content_embeddings, model):
    keyword_embeddings = model.encode(keywords_batch, convert_to_tensor=True)
    scores = cosine_similarity(keyword_embeddings, content_embeddings)
    return scores.max(axis=1)


def calculate_website_content_scores_parallel(
    df_keywords,
    df_pages,
    use_context=False,
    scale_0_to_10=True,
    verbose=True,
    progress_bar=None,
    status_callback=None,
    use_tfidf=True,
    blend_weight_semantic=0.7,
    n_jobs=2,
    batch_size=250,
    device="cpu"
):
    if df_keywords.empty or df_pages.empty:
        if verbose:
            print("⚠️ No keyword or website content data available.")
        df_keywords["content_match_score"] = 0.0
        return df_keywords

    if status_callback:
        status_callback.text("🔍 Step 2 – Calculating content match scores...")

    try:
        model = SentenceTransformer("all-mpnet-base-v2", device=device)

        # Use full visible_text from website content
        website_texts = df_pages["visible_text"].fillna("").tolist()
        website_urls = df_pages["url"].tolist()

        if verbose:
            print(f"⚙️ Encoding {len(website_texts)} website content blocks...")
        content_embeddings = model.encode(website_texts, convert_to_tensor=True, show_progress_bar=True)

        keyword_texts = [
            f"{row['Source Keyword']} {row['Keyword']}" if use_context and "Source Keyword" in row else row["Keyword"]
            for _, row in df_keywords.iterrows()
        ]

        # === SEMANTIC SIMILARITY ===
        if verbose:
            print(f"⚙️ Calculating semantic similarity for {len(keyword_texts)} keywords...")
        batches = [keyword_texts[i:i + batch_size] for i in range(0, len(keyword_texts), batch_size)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_semantic_scores_batch)(batch, content_embeddings, model) for batch in tqdm(batches)
        )
        sem_scores = np.concatenate(results)
        df_keywords["semantic_score"] = np.round(sem_scores * 10, 2) if scale_0_to_10 else sem_scores

        # === TF-IDF SIMILARITY ===
        if use_tfidf:
            if verbose:
                print("⚙️ Calculating TF-IDF similarity...")
            tfidf = TfidfVectorizer(stop_words="english", max_features=10000)
            tfidf.fit(website_texts)
            website_matrix = tfidf.transform(website_texts)
            keyword_matrix = tfidf.transform(keyword_texts)

            tfidf_sim = cosine_similarity(keyword_matrix, website_matrix)
            max_scores = tfidf_sim.max(axis=1)
            avg_scores = tfidf_sim.mean(axis=1)
            tfidf_scores = np.clip(0.7 * max_scores + 0.3 * avg_scores, 0, 1)
            df_keywords["tfidf_score"] = np.round(tfidf_scores * 10, 2) if scale_0_to_10 else tfidf_scores
        else:
            df_keywords["tfidf_score"] = 0

        # === FINAL SCORE ===
        df_keywords["content_match_score"] = np.round(
            blend_weight_semantic * df_keywords["semantic_score"] + 
            (1 - blend_weight_semantic) * df_keywords["tfidf_score"], 2
        )

        if progress_bar:
            progress_bar.progress(1.0)
        if status_callback:
            status_callback.text("✅ Step 2 complete: Content match scoring done.")

    except Exception as e:
        print(f"❌ Scoring failed: {e}")
        df_keywords["content_match_score"] = 0.0

    return df_keywords

def calculate_book_content_scores_parallel(
    df_keywords,
    use_context=False,
    scale_0_to_10=True,
    verbose=True,
    progress_bar=None,
    status_callback=None,
    use_tfidf=True,
    blend_weight_semantic=0.7,
    n_jobs=2,
    batch_size=500,
    device="cpu"
):
    book_csv_path = BOOK_PATH
    if df_keywords.empty or not book_csv_path or not os.path.exists(book_csv_path):
        if verbose:
            print("⚠️ No keyword data or valid book CSV path provided.")
        df_keywords["content_match_score"] = 0.0
        return df_keywords

    if status_callback:
        status_callback.text("🔍 Step 2 – Calculating book content match scores...")

    try:
        model = SentenceTransformer("all-mpnet-base-v2", device=device)

        # Load book content
        book_df = pd.read_csv(book_csv_path)
        book_texts = book_df.iloc[:, 0].dropna().astype(str).tolist()

        if verbose:
            print(f"📘 Loaded {len(book_texts)} book sections.")

        # Prepare keywords
        keyword_texts = [
            f"{row['Source Keyword']} {row['Keyword']}" if use_context and "Source Keyword" in row else row["Keyword"]
            for _, row in df_keywords.iterrows()
        ]

        # Encode content once
        content_embeddings = model.encode(book_texts, convert_to_tensor=True, show_progress_bar=True)

        # Split into batches for parallel computation
        batches = [keyword_texts[i:i + batch_size] for i in range(0, len(keyword_texts), batch_size)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_semantic_scores_batch)(batch, content_embeddings, model) for batch in batches
        )

        # Merge scores
        sem_scores = np.concatenate(results)
        df_keywords["semantic_score"] = np.round(sem_scores * 10, 2) if scale_0_to_10 else sem_scores

        # === TF-IDF Matching ===
        if use_tfidf:
            if verbose:
                print("🧮 Calculating TF-IDF scores...")
            vectorizer = TfidfVectorizer(stop_words="english", max_features=10000)
            vectorizer.fit(book_texts)
            content_matrix = vectorizer.transform(book_texts)
            keyword_matrix = vectorizer.transform(keyword_texts)
            sim_matrix = cosine_similarity(keyword_matrix, content_matrix)
            max_scores = sim_matrix.max(axis=1)
            avg_scores = sim_matrix.mean(axis=1)
            tfidf_scores = np.clip(0.7 * max_scores + 0.3 * avg_scores, 0, 1)
            df_keywords["tfidf_score"] = np.round(tfidf_scores * 10, 2) if scale_0_to_10 else tfidf_scores
        else:
            df_keywords["tfidf_score"] = 0

        # === Final Score ===
        df_keywords["content_match_score"] = np.round(
            blend_weight_semantic * df_keywords["semantic_score"] +
            (1 - blend_weight_semantic) * df_keywords["tfidf_score"], 2
        )

        if progress_bar:
            progress_bar.progress(1.0)
        if status_callback:
            status_callback.text("✅ Book content match scoring complete.")

    except Exception as e:
        print(f"❌ Error during book scoring: {e}")
        df_keywords["content_match_score"] = 0.0

    return df_keywords

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