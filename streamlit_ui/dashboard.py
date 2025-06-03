import streamlit as st
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import altair as alt
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sqlalchemy import create_engine
from services.crawler_service import run_crawler_ui
from streamlit_ui.ui import run_keyword_pipeline_ui
from streamlit_ui.aida import run_aida_page

st.set_page_config(page_title="Keyword Explorer Dashboard", layout="wide")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "dashboard"

def set_active_tab(tab_name):
    st.session_state.active_tab = tab_name

with st.sidebar:
    st.markdown("""
        <style>
        .sidebar-title {
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 20px;
        }
        .stButton>button {
            width: 100%;
            padding: 0.6em 1em;
            margin-bottom: 0.5em;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-title">Keyword Explorer</div>', unsafe_allow_html=True)

    if st.button("Keyword Dashboard"):
        set_active_tab("dashboard")
    if st.button("Website Crawler"):
        set_active_tab("crawler")
    if st.button("Manual Keyword Input"):
        set_active_tab("keyword_input")
    if st.button("AIDA Content Planner"):
        set_active_tab("aida")

if st.session_state.active_tab == "dashboard":


    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    DB_FILE = os.path.join(DATA_DIR, "keyword_data.db")
    engine = create_engine(f"sqlite:///{DB_FILE}")

    st.title("📊 Keyword Insights Dashboard")

    try:
        df = pd.read_sql("SELECT * FROM keyword_ranked", engine)

        col1, col2, col3 = st.columns(3)
        col1.metric("🔑 Total Keywords", len(df))
        col2.metric("💰 Avg CPC (USD)", f"${df['High CPC (USD)'].mean():.2f}")
        col3.metric("📊 Avg Monthly Searches", round(df["Avg Monthly Searches"].mean(), 1))

        intent_list = ["All"] + sorted(df["Intent"].dropna().unique())
        selected_intent = st.selectbox("🎯 Filter by Intent", intent_list)
        if selected_intent != "All":
            df = df[df["Intent"] == selected_intent]

        st.markdown("---")
        st.markdown("### Top Keywords Per Source")

        try:
            df_top = pd.read_sql("SELECT rowid, * FROM top_keywords_per_source", engine)

            if df_top.empty:
                st.info("No keywords found.")
            else:
                sort_col = st.selectbox("Sort by column", df_top.columns)
                sort_dir = st.radio("Sort direction", ["Ascending", "Descending"], horizontal=True)

                df_sorted = df_top.sort_values(sort_col, ascending=(sort_dir == "Ascending"))

                # === Pagination for Top Keywords ===
                rows_per_page = 100
                total_rows = len(df_sorted)
                total_pages = (total_rows - 1) // rows_per_page + 1

                page_number = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
                start_idx = (page_number - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, total_rows)

                df_page = df_sorted.iloc[start_idx:end_idx]

                edited_df = st.data_editor(
                    df_page,
                    column_config={
                        "rowid": st.column_config.NumberColumn("Row ID", disabled=True),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="top_keywords_editor",
                )

        except Exception as e:
            st.error(f"❌ Failed to load or modify top_keywords_per_source table: {e}")

        st.markdown("---")
        st.markdown("### Row Keyword Table")

        # === Pagination for Main Keyword Table ===
        rows_per_page = 30
        total_rows = len(df)
        total_pages = (total_rows - 1) // rows_per_page + 1

        page_number = st.number_input("Keyword Table Page", min_value=1, max_value=total_pages, step=1, key="keyword_page")
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)

        st.markdown(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")
        st.dataframe(df.sort_values("keyword_score", ascending=False).iloc[start_idx:end_idx], use_container_width=True)

        st.markdown("---")
        st.markdown("### 💰 High CPC + Low Competition")
        low_comp = df["Competition"].str.upper().isin(["LOW", "VERY_LOW"])
        high_cpc = df["High CPC (USD)"] > 1.0
        gold_keywords = df[low_comp & high_cpc]
        st.dataframe(gold_keywords)

        st.markdown("---")
        st.markdown("### 🧠 Cluster Distribution")
        cluster_counts = df["Cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        st.markdown("### 🔝 Top Keywords by Monthly Searches")
        top_volume = df.sort_values("Avg Monthly Searches", ascending=False).head(10)
        st.bar_chart(top_volume.set_index("Keyword")["Avg Monthly Searches"])

        st.markdown("---")
        st.markdown("### ❤️ Sentiment Score Distribution")
        if "Sentiment Score" in df.columns:
            df_sent = df[["Keyword", "Sentiment Score"]].dropna().sort_values("Sentiment Score", ascending=False)
            top_n = st.slider("Number of keywords to display", 10, len(df_sent), 50)
            df_sent = df_sent.head(top_n)

            chart = alt.Chart(df_sent).mark_bar(color="#4e79a7").encode(
                x=alt.X("Sentiment Score:Q", title="Sentiment Score"),
                y=alt.Y("Keyword:N", sort="-x", title="Keyword"),
                tooltip=["Keyword", "Sentiment Score"]
            ).properties(
                width=700,
                height=25 * len(df_sent)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("⚠️ Sentiment Score column missing.")

        st.markdown("---")
        st.markdown("### 📥 Download Filtered Keywords")
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="keywords_filtered.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Failed to load keyword_ranked table: {e}")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))  # 
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    DB_FILE = os.path.join(DATA_DIR, "keyword_data.db")
    engine = create_engine(f"sqlite:///{DB_FILE}")

    st.title("📊 Keyword Insights Dashboard")

    try:
        df = pd.read_sql("SELECT * FROM keyword_ranked", engine)

        col1, col2, col3 = st.columns(3)
        col1.metric("🔑 Total Keywords", len(df))
        col2.metric("💰 Avg CPC (USD)", f"${df['High CPC (USD)'].mean():.2f}")
        col3.metric("📊 Avg Monthly Searches", round(df["Avg Monthly Searches"].mean(), 1))

        intent_list = ["All"] + sorted(df["Intent"].dropna().unique())
        selected_intent = st.selectbox("🎯 Filter by Intent", intent_list)
        if selected_intent != "All":
            df = df[df["Intent"] == selected_intent]

        st.markdown("---")
        st.markdown("### Top Keywords Per Source")
        try:
            df_top = pd.read_sql("SELECT rowid, * FROM top_keywords_per_source", engine)

            if df_top.empty:
                st.info("No keywords found.")
            else:
                sort_col = st.selectbox("Sort by column", df_top.columns)
                sort_dir = st.radio("Sort direction", ["Ascending", "Descending"], horizontal=True)
                df_sorted = df_top.sort_values(sort_col, ascending=(sort_dir == "Ascending"))

                edited_df = st.data_editor(
                    df_sorted,
                    column_config={
                        "rowid": st.column_config.NumberColumn("Row ID", disabled=True),
                    },
                    num_rows="dynamic",
                    use_container_width=True,
                    key="top_keywords_editor",
                )

        except Exception as e:
            st.error(f"❌ Failed to load or modify top_keywords_per_source table: {e}")

        st.markdown("---")
        st.markdown("### Row Keyword Table")
        st.dataframe(df.sort_values("keyword_score", ascending=False), use_container_width=True)

        st.markdown("---")
        st.markdown("### 💰 High CPC + Low Competition")
        low_comp = df["Competition"].str.upper().isin(["LOW", "VERY_LOW"])
        high_cpc = df["High CPC (USD)"] > 1.0
        gold_keywords = df[low_comp & high_cpc]
        st.dataframe(gold_keywords)

        st.markdown("---")
        st.markdown("### 🧠 Cluster Distribution")
        cluster_counts = df["Cluster"].value_counts().sort_index()
        st.bar_chart(cluster_counts)

        st.markdown("### 🔝 Top Keywords by Monthly Searches")
        top_volume = df.sort_values("Avg Monthly Searches", ascending=False).head(10)
        st.bar_chart(top_volume.set_index("Keyword")["Avg Monthly Searches"])

        st.markdown("---")
        st.markdown("### ❤️ Sentiment Score Distribution")
        if "Sentiment Score" in df.columns:
            df_sent = df[["Keyword", "Sentiment Score"]].dropna().sort_values("Sentiment Score", ascending=False)
            top_n = st.slider("Number of keywords to display", 10, len(df_sent), 50)
            df_sent = df_sent.head(top_n)

            chart = alt.Chart(df_sent).mark_bar(color="#4e79a7").encode(
                x=alt.X("Sentiment Score:Q", title="Sentiment Score"),
                y=alt.Y("Keyword:N", sort="-x", title="Keyword"),
                tooltip=["Keyword", "Sentiment Score"]
            ).properties(
                width=700,
                height=25 * len(df_sent)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("⚠️ Sentiment Score column missing.")

        st.markdown("---")
        st.markdown("### 📥 Download Filtered Keywords")
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="keywords_filtered.csv", mime="text/csv")

    except Exception as e:
        st.error(f"❌ Failed to load keyword_ranked table: {e}")

elif st.session_state.active_tab == "crawler":
    run_crawler_ui()

elif st.session_state.active_tab == "keyword_input":
    run_keyword_pipeline_ui()

elif st.session_state.active_tab == "aida":
    run_aida_page()