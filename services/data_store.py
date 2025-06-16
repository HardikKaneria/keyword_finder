import os
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# === Unified Path Setup ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)

DB_FILE = os.path.join(DATA_DIR, "keyword_data.db")
WEBSITE_FILE = os.path.join(DATA_DIR, "website_data.db")


def get_combined_website_texts(db_file=WEBSITE_FILE):
    conn = sqlite3.connect(db_file)
    try:
        df = pd.read_sql("SELECT url, title, meta_description, h1, h2, h3, visible_text FROM pages", conn)
        df.fillna("", inplace=True)
        df["visible_text"] = (
            df["title"] + " " + df["meta_description"] + " " +
            df["h1"] + " " + df["h2"] + " " + df["h3"] + " " + df["visible_text"]
        )
        return df[["url", "visible_text"]]  # ✅ important: return as DataFrame
    except Exception as e:
        print(f"❌ Failed to load website data: {e}")
        return pd.DataFrame(columns=["url", "visible_text"])
    finally:
        conn.close()

def get_combined_website_keywords(db_file=WEBSITE_FILE):
    # Load keyword and page data
    conn = sqlite3.connect("data/website_data.db")
    df_keywords = pd.read_sql_query("SELECT * FROM keywords_extraction", conn)
    conn.close()
    df_keywords = df_keywords.rename(columns={"keyword": "Keyword", "score": "content_score"})

    return df_keywords


def store_data_in_sqlite(df):
    def safe_serialize(x):
        if isinstance(x, (list, dict, tuple)):
            return json.dumps(x)
        return x

    df = df.copy()
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict, tuple))).any():
            df[col] = df[col].apply(safe_serialize)

    conn = sqlite3.connect(DB_FILE)
    df.to_sql("keyword_ranked", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def store_top_keywords_per_source(df, table_name="top_keywords_per_source", top_n=10, min_score_ratio=0.3):
    if "Source Keyword" not in df.columns:
        print("⚠️ 'Source Keyword' missing – defaulting to 'Keyword' as source.")
        df["Source Keyword"] = df["Keyword"]

    conn = sqlite3.connect(DB_FILE)
    top_keywords = []

    for source_kw, group in df.groupby("Source Keyword"):
        sorted_group = group.sort_values("content_score", ascending=False).reset_index(drop=True)
        if sorted_group.empty or "content_score" not in sorted_group.columns:
            continue

        top_score = sorted_group.loc[0, "content_score"]
        min_allowed = top_score * min_score_ratio

        filtered = sorted_group[
            (sorted_group["content_score"] >= min_allowed) &
            (sorted_group["Avg Monthly Searches"] > 1000) &
            (sorted_group["content_match_score"] >= 8) &
            (sorted_group["content_match_score"] < 10)
        ]

        limited = filtered.head(top_n)
        top_keywords.append(limited)

    if top_keywords:
        final_df = pd.concat(top_keywords, ignore_index=True)
        final_df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"📦 Stored top content-optimized keywords into '{table_name}' table.")
    else:
        print(f"⚠️ No qualifying keywords found using content_score. Nothing stored in '{table_name}'.")

    conn.commit()
    conn.close()


def visualize_data(keyword_metrics):
    df = pd.DataFrame(keyword_metrics)

    required_cols = {"Keyword", "Avg Monthly Searches"}
    if not required_cols.issubset(df.columns):
        print("⚠️ Cannot visualize — missing required columns.")
        return

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Avg Monthly Searches",
        y="Keyword",
        data=df.sort_values("Avg Monthly Searches", ascending=False).head(10)
    )
    plt.title("Top 10 Keywords by Avg Monthly Searches")
    plt.xlabel("Avg Monthly Searches")
    plt.ylabel("Keyword")
    plt.tight_layout()

    output_path = os.path.join(EXPORT_DIR, "top_keywords.png")
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Saved visualization to {output_path}")
