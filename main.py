# keyword_analyzer/main.py
from utils.text_analysis import perform_sentiment_analysis, calculate_keyword_content_scores, calculate_keyword_scores
from utils.intent_classifier import classify_keyword_intent
from utils.clustering import cluster_keywords
from services.keyword_fetcher import fetch_keyword_metrics, read_keywords_from_csv
from services.data_store import store_data_in_sqlite, export_top_keywords_by_intent, store_top_keywords_per_source, visualize_data, get_combined_website_texts

import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "scrape_cache.pkl")
INPUT_CSV = os.path.join(BASE_DIR, "../data/keywords.csv")

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        SCRAPE_CACHE = pickle.load(f)
        print(f"🧠 Loaded {len(SCRAPE_CACHE)} cached URLs.")
else:
    SCRAPE_CACHE = {}


def run_pipeline():
    all_keywords = read_keywords_from_csv(INPUT_CSV)

    if not all_keywords:
        print("⚠️ No keywords found in CSV.")
        return

    print(f"📥 Total keywords from CSV: {len(all_keywords)}")

    keyword_metrics = fetch_keyword_metrics(all_keywords)
    sentiments = perform_sentiment_analysis([km["Keyword"] for km in keyword_metrics])
    clusters = cluster_keywords([km["Keyword"] for km in keyword_metrics])

    df_all = pd.DataFrame(keyword_metrics)
    df_sent = pd.DataFrame(sentiments)
    df_scored = df_all.merge(df_sent, on="Keyword")
    df_scored["Intent"] = df_scored["Keyword"].apply(classify_keyword_intent)

    website_texts = get_combined_website_texts()
    df_scored = calculate_keyword_content_scores(df_scored, website_texts, scale_0_to_10=True)

    df_scored.loc[df_scored["Intent"] == "unknown", "Intent"] = df_scored[df_scored["Intent"] == "unknown"]["Keyword"].apply(classify_keyword_intent)

    df_scored["relevance_bucket"] = pd.cut(
        df_scored["content_match_score"],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Very High"]
    )

    df_scored = calculate_keyword_scores(df_scored)

    df_final = df_scored.merge(clusters, on="Keyword")

    visualize_data(keyword_metrics)
    store_data_in_sqlite(df_final)
    export_top_keywords_by_intent(df_final)
    store_top_keywords_per_source(df_final)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(SCRAPE_CACHE, f)
        print(f"💾 Saved {len(SCRAPE_CACHE)} scraped pages to cache.")

    print("✅ All data processed, scored, and saved successfully.")


if __name__ == "__main__":
    run_pipeline()