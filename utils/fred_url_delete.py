import sqlite3
from urllib.parse import urlparse
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))  # 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_FILE = os.path.join(DATA_DIR, "website_data.db")  # Update path if needed

def delete_fragment_and_query_urls_from_db(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT url FROM pages")
    rows = cursor.fetchall()
    to_delete = []

    for (url,) in rows:
        parsed = urlparse(url)
        if parsed.fragment or parsed.query:
            to_delete.append(url)

    for url in to_delete:
        cursor.execute("DELETE FROM pages WHERE url = ?", (url,))

    conn.commit()
    conn.close()
    print(f"🧹 Deleted {len(to_delete)} URLs with fragments or query parameters from the database.")

if __name__ == "__main__":
    delete_fragment_and_query_urls_from_db(DB_FILE)