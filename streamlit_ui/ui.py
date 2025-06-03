# streamlit_ui/ui.py
import streamlit as st
import pandas as pd
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.keyword_fetcher import fetch_keyword_metrics, fetch_keyword_metrics_from_url, fetch_exact_keyword_metrics
from utils.text_analysis import perform_sentiment_analysis, calculate_website_content_scores_parallel, calculate_book_content_scores_parallel, calculate_keyword_scores
from utils.intent_classifier import bulk_classify_keyword_intents
from utils.clustering import cluster_keywords
from services.data_store import store_data_in_sqlite, export_top_keywords_by_intent, store_top_keywords_per_source, get_combined_website_texts

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
    if not keyword_metrics:
        st.error("❌ No keyword metrics fetched.")
        return

    keyword_texts = [km.get("Keyword") for km in keyword_metrics if km.get("Keyword")]

    sentiment_progress = st.progress(0)
    sentiment_status = st.empty()
    df_sent = perform_sentiment_analysis(keyword_texts, progress_bar=sentiment_progress, status_callback=sentiment_status)

    if df_sent.empty or "Keyword" not in df_sent.columns:
        st.error("❌ Sentiment analysis failed: 'Keyword' column missing or result empty.")
        return

    cluster_progress = st.progress(0)
    cluster_status = st.empty()
    clusters = cluster_keywords(keyword_texts, progress_callback=cluster_progress, status_callback=cluster_status)

    df_all = pd.DataFrame(keyword_metrics).drop_duplicates(subset=["Keyword"])
    df_scored = df_all.merge(df_sent, on="Keyword")

    intent_progress = st.progress(0)
    intent_status = st.empty()
    df_scored["Intent"] = bulk_classify_keyword_intents(
        df_scored["Keyword"].tolist(),
        progress_bar=intent_progress,
        status_callback=intent_status
    )

    content_progress = st.progress(0)
    content_status = st.empty()
    website_texts = get_combined_website_texts()
    df_scored = calculate_website_content_scores_parallel(
        df_scored,
        website_texts=website_texts,
        use_context=True,
        scale_0_to_10=True,
        use_tfidf=True,
        blend_weight_semantic=0.8,
        progress_bar=content_progress,
        status_callback=content_status
    )
    # df_scored = calculate_book_content_scores_parallel(
    #     df_scored,
    #     use_context=True,
    #     scale_0_to_10=True,
    #     use_tfidf=True,
    #     blend_weight_semantic=0.8,
    #     progress_bar=content_progress,
    #     status_callback=content_status
    # )

    df_scored["relevance_bucket"] = pd.cut(
        df_scored["content_match_score"],
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Very High"]
    )

    score_progress = st.progress(0)
    score_status = st.empty()
    df_scored = calculate_keyword_scores(df_scored, progress_bar=score_progress, status_callback=score_status)

    df_final = df_scored.merge(clusters, on="Keyword")

    store_status = st.empty()
    store_status.text("💾 Saving results to database...")
    df_all = df_final.drop_duplicates(subset=["Keyword"])
    store_data_in_sqlite(df_all)
    export_top_keywords_by_intent(df_all)
    store_top_keywords_per_source(df_all)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(SCRAPE_CACHE, f)

    store_status.text("✅ Completed!")
    st.success("✅ Analysis complete.")
    # === Pagination Setup ===
    rows_per_page = 1000  # Adjust as needed
    total_rows = len(df_final)
    total_pages = (total_rows - 1) // rows_per_page + 1

    page_number = st.number_input(
        "Keyword Table Page",
        min_value=1,
        max_value=total_pages,
        step=1,
        key="df_final_page"
    )

    start_idx = (page_number - 1) * rows_per_page
    end_idx = min(start_idx + rows_per_page, total_rows)

    # === Paginated & Sorted Table View ===
    st.markdown(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
    df_page = df_final.sort_values("keyword_score", ascending=False).iloc[start_idx:end_idx]
    st.dataframe(df_page, use_container_width=True)


    csv = df_final.to_csv(index=False)
    st.download_button("📥 Download Results CSV", data=csv, file_name="keyword_results.csv", mime="text/csv")
