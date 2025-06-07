from keybert import KeyBERT
import sqlite3
import pandas as pd
import os

# --- Paths ---
def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../"))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    DB_FILE = os.path.join(data_dir, "website_data.db")
    return project_root, data_dir, DB_FILE

BASE_DIR, DATA_DIR, DB_FILE = get_paths()

# --- Keyword Extraction ---
def extract_keywords_from_visible_text(min_words=2, max_words=5, max_keywords=20, max_text_words=1000):
    # Connect to DB
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT url, visible_text FROM pages", conn)
    conn.close()

    # Load KeyBERT model
    kw_model = KeyBERT()
    results = []

    print("🔄 Starting keyword extraction...\n")

    # Process each row
    for i, row in df.iterrows():
        url = row['url']
        text = row['visible_text']

        if not isinstance(text, str) or not text.strip():
            continue

        trimmed_text = " ".join(text.split()[:max_text_words])

        try:
            keywords = kw_model.extract_keywords(
                trimmed_text,
                keyphrase_ngram_range=(min_words, max_words),
                stop_words='english',
                use_mmr=True,
                diversity=0.7,
                top_n=max_keywords
            )

            for phrase, score in keywords:
                results.append({
                    'url': url,
                    'keyword': phrase,
                    'score': score,
                    'n_gram': len(phrase.split())
                })

            print(f"✅ {i+1}/{len(df)} | {len(keywords)} keywords from {url}")

        except Exception as e:
            print(f"⚠️ {i+1}/{len(df)} | Error with {url}: {e}")

    # Store to database
    if results:
        result_df = pd.DataFrame(results)
        conn = sqlite3.connect(DB_FILE)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS keywords_extraction (
                url TEXT,
                keyword TEXT,
                score REAL,
                n_gram INTEGER
            )
        """)
        conn.execute("DELETE FROM keywords_extraction")
        result_df.to_sql("keywords_extraction", conn, if_exists="append", index=False)
        conn.close()

        print(f"\n✅ Inserted {len(result_df)} keyword records into database.")
    else:
        print("\n⚠️ No keywords extracted.")

# --- Run ---
if __name__ == '__main__':
    extract_keywords_from_visible_text()
