# streamlit_ui/ui.py
import streamlit as st
import pandas as pd
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.keyword_fetcher import fetch_keyword_metrics, fetch_keyword_metrics_from_url, fetch_exact_keyword_metrics
from utils.text_analysis import perform_sentiment_analysis, calculate_keyword_scores, score_keywords_for_content
from utils.keyword_content_matcher import calculate_chunked_keyword_scores
from utils.intent_classifier import bulk_classify_keyword_intents
from utils.clustering import cluster_keywords
from services.data_store import store_data_in_sqlite, store_top_keywords_per_source, get_combined_website_texts

# === Load environment and set constants ===

CACHE_FILE = "scrape_cache.pkl"

if CACHE_FILE and isinstance(CACHE_FILE, str):
    try:
        with open(CACHE_FILE, "rb") as f:
            SCRAPE_CACHE = pickle.load(f)
    except:
        SCRAPE_CACHE = {}
else:
    SCRAPE_CACHE = {}

def run_keyword_pipeline_ui():
    st.title("🧠 Keyword Analyzer")

    mode = st.radio("Choose input method:", ["Manual Keywords", "Exact Keywords", "Website URL"])

    if mode == "Manual Keywords":
        keyword_input = st.text_area(
            "Enter keywords (one per line)",
            placeholder="e.g.\nemotional intelligence\nhow to lead remotely\nbuy self-reflection journal"
        )
        if st.button("🚀 Run Keyword Analysis"):
            keywords = list({kw.strip() for kw in keyword_input.strip().split("\n") if kw.strip()})
            if not keywords:
                st.warning("⚠️ Please enter at least one keyword.")
                return

            unique_keywords = sorted(set(keywords))
            st.info(f"🔍 Processing {len(unique_keywords)} unique keywords...")
            fetch_progress = st.progress(0)
            fetch_status = st.empty()

            with st.spinner("📡 Fetching metrics from Google Ads..."):
                keyword_metrics = fetch_keyword_metrics(unique_keywords, progress_bar=fetch_progress, status_text=fetch_status)

            process_and_display_results(keyword_metrics)
    
    elif mode == "Exact Keywords":
        e_keyword_input = st.text_area(
            "Enter keywords (one per line)",
            placeholder="e.g.\nemotional intelligence\nhow to lead remotely\nbuy self-reflection journal"
        )
        if st.button("🚀 Run Keyword Analysis"):
            keywords = list({kw.strip() for kw in e_keyword_input.strip().split("\n") if kw.strip()})
            if not keywords:
                st.warning("⚠️ Please enter at least one keyword.")
                return

            unique_keywords = sorted(set(keywords))
            st.info(f"🔍 Processing {len(unique_keywords)} unique keywords...")
            fetch_progress = st.progress(0)
            fetch_status = st.empty()

            with st.spinner("📡 Fetching metrics from Google Ads..."):
                keyword_metrics = fetch_exact_keyword_metrics(unique_keywords, progress_bar=fetch_progress, status_text=fetch_status)

            process_and_display_results(keyword_metrics)

    elif mode == "Website URL":
        url_input = st.text_input("🔗 Enter a webpage URL:")
        use_full_site = st.checkbox("Use full website domain", value=True)
        if st.button("🌐 Analyze Website Keywords"):
            if url_input.strip():
                fetch_progress = st.progress(0)
                fetch_status = st.empty()
                with st.spinner("📡 Fetching keywords from URL..."):
                    keyword_metrics = fetch_keyword_metrics_from_url(url_input.strip(), full_site=use_full_site, progress_bar=fetch_progress, status_text=fetch_status)
                process_and_display_results(keyword_metrics)
            else:
                st.warning("⚠️ Please enter a valid URL.")

def process_and_display_results(keyword_metrics):
    try:
        if not keyword_metrics:
            st.error("❌ No keyword metrics fetched.")
            return
        
        final_results = [r for r in keyword_metrics if r["Avg Monthly Searches"] >= 1000]

        keyword_texts = [km.get("Keyword") for km in final_results if km.get("Keyword")]

        if not keyword_texts:
            st.error("❌ No valid keywords found.")
            return

        # === Sentiment ===
        try:
            sentiment_progress = st.progress(0)
            sentiment_status = st.empty()
            df_sent = perform_sentiment_analysis(keyword_texts, progress_bar=sentiment_progress, status_callback=sentiment_status)
        except Exception as e:
            st.error(f"❌ Sentiment analysis failed: {e}")
            return

        if df_sent.empty or "Keyword" not in df_sent.columns:
            st.error("❌ Sentiment analysis result invalid.")
            return

        # === Clustering ===
        try:
            cluster_progress = st.progress(0)
            cluster_status = st.empty()
            clusters = cluster_keywords(keyword_texts, progress_callback=cluster_progress, status_callback=cluster_status)
        except Exception as e:
            st.warning(f"⚠️ Clustering failed: {e}")
            clusters = pd.DataFrame({"Keyword": keyword_texts, "Cluster": -1})

        # === Merge Core Data ===
        df_all = pd.DataFrame(final_results).drop_duplicates(subset=["Keyword"])
        df_scored = df_all.merge(df_sent, on="Keyword")

        # === Intent Classification ===
        try:
            intent_progress = st.progress(0)
            intent_status = st.empty()
            df_scored["Intent"] = bulk_classify_keyword_intents(
                df_scored["Keyword"].tolist(),
                progress_bar=intent_progress,
                status_callback=intent_status
            )
            print(df_scored.head())
        except Exception as e:
            st.warning(f"⚠️ Intent classification failed: {e}")
            df_scored["Intent"] = "Unknown"

        # === Content Score ===
        try:

            df_pages = get_combined_website_texts()

            df_content_scored = calculate_chunked_keyword_scores(
                df_scored,
                df_pages,
                scale_0_to_10=True,
            )

            print(df_scored.head())

        except Exception as e:
            st.warning(f"⚠️ Content score calculation failed: {e}")
            df_content_scored["content_match_score"] = 0.0

        # === Relevance Bucket ===
        df_content_scored["relevance_bucket"] = pd.cut(
            df_content_scored["content_match_score"],
            bins=[0, 2, 5, 8, 10],
            labels=["Low", "Medium", "High", "Very High"]
        )

        df_final_1 = df_scored.merge(df_content_scored, on="Keyword", how="left")

        print(df_content_scored.head())

        # === Final Score ===
        try:
            score_progress = st.progress(0)
            score_status = st.empty()
            df_k_scored = score_keywords_for_content(df_final_1, progress_bar=score_progress, status_callback=score_status)
            print(df_scored.head())
            st.dataframe(df_k_scored[["Keyword", "content_score", "Intent"]].head(20))
        except Exception as e:
            st.warning(f"⚠️ Final keyword scoring failed: {e}")
            df_k_scored["content_score"] = 0.0

        # === Merge Clusters ===
        df_final = df_k_scored.merge(clusters, on="Keyword", how="left")
        st.dataframe(df_final.head(20))

        # === Save to DB ===
        try:
            store_status = st.empty()
            store_status.text("💾 Saving results to database...")
            df_all = df_final.drop_duplicates(subset=["Keyword"])
            store_data_in_sqlite(df_all)
            store_top_keywords_per_source(df_all)
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(SCRAPE_CACHE, f)
            store_status.text("✅ Completed!")
            st.success("✅ Analysis complete.")
        except Exception as e:
            st.warning(f"⚠️ Saving to database failed: {e}")

        csv = df_final.to_csv(index=False)
        st.download_button("📥 Download Results CSV", data=csv, file_name="keyword_results.csv", mime="text/csv")

    except Exception as e:
        st.exception(f"💥 Unhandled error occurred:\n{e}")

