import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import json
import pandas as pd
import streamlit as st
from utils.url_matcher import match_urls_to_aida

# === Load environment and set constants ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))  # 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
WEBSITE_FILE = os.path.join(DATA_DIR, "website_data.db")
DB_FILE = os.path.join(DATA_DIR, "keyword_data.db")

def load_data():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT Keyword FROM top_keywords_per_source", conn)
    return df

# === Streamlit App ===
def run_aida_page():
    st.title("AIDA Content Generator")
    st.markdown("---")
    st.markdown("### Top Keywords Per Source")

    try:
        conn = sqlite3.connect(DB_FILE)
        df_top = pd.read_sql("SELECT * FROM aida_output", conn)

        if df_top.empty:
            st.info("No keywords found.")
        else:
            # Separate DataFrame for display only (no ConfidenceScores)
            df_display = df_top.drop(columns=["ConfidenceScores"], errors="ignore")

            # === Pagination Setup ===
            rows_per_page = 100
            total_rows = len(df_display)
            total_pages = (total_rows - 1) // rows_per_page + 1

            page_number = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
            start_idx = (page_number - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)

            # === Sorting Controls ===
            sort_col = st.selectbox("Sort by column", df_display.columns)
            sort_dir = st.radio("Sort direction", ["Ascending", "Descending"], horizontal=True)

            df_sorted = df_display.sort_values(sort_col, ascending=(sort_dir == "Ascending"))

            # === Paginated + Sorted View ===
            df_page = df_sorted.iloc[start_idx:end_idx]

            # Editor view for paginated data
            edited_df = st.data_editor(
                df_page,
                column_config={
                    "rowid": st.column_config.NumberColumn("Row ID", disabled=True),
                },
                num_rows="dynamic",
                use_container_width=True,
                key="top_keywords_editor",
            )

            # Export button using full original data (with ConfidenceScores)
            st.download_button(
                label="📥 Export Full AIDA Table (CSV)",
                data=df_top.to_csv(index=False).encode('utf-8'),
                file_name="aida_output_full.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Failed to load or modify aida_output table: {e}")

    st.markdown("### ⚙️ Choose Optional Steps")

    run_media = st.checkbox("🗂️ Generate Media Plan")
    run_urls = st.checkbox("🔗 Match URLs for CTA")
    # use_chatgpt = st.checkbox("🤖 Use ChatGPT for Theme Generation")

    if st.button("🚀 Run AIDA Pipeline"):
        df = load_data()

        with st.spinner("🔍 Classifying AIDA stages..."):
            # stage_df = classify_keyword_stage(df)
            stage_df = df  # Pass-through if stage classification is skipped

        if run_media:
            with st.spinner("📄 Generating Media Plans..."):
                # media_plans_df = generate_predictions(stage_df)
                media_plans_df = stage_df
        else:
            media_plans_df = stage_df  # Pass-through if media step is skipped

        if run_urls:
            with st.spinner("🎯 Generating 2 Themes for Media Plan"):
                # themes_df = generate_aida_content_plan(media_plans_df, use_chatgpt)
                themes_df = media_plans_df  # Pass-through if media step is skipped

            url_depth = 10
            with st.spinner("🔗 Matching URLs for CTA Target..."):
                final_df = match_urls_to_aida(themes_df, url_depth)
        else:
            final_df = media_plans_df  # Pass-through if URL step is skipped

        with st.spinner("💾 Saving Final AIDA Table..."):
            conn = sqlite3.connect(DB_FILE)
            if "ConfidenceScores" in final_df.columns:
                final_df["ConfidenceScores"] = final_df["ConfidenceScores"].apply(json.dumps)
            final_df.to_sql("aida_output", conn, if_exists="replace", index=False)
            conn.close()

        st.success("✅ Selected AIDA processes completed and saved successfully!")
