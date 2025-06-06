import sqlite3
import os

def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../"))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    DB_FILE = os.path.join(data_dir, "website_data.db")
    CSV_FILE = os.path.join(data_dir, "crawled_data.csv")
    return project_root, data_dir, DB_FILE, CSV_FILE

BASE_DIR, DATA_DIR, DB_FILE, CSV_FILE = get_paths()
db_path = DB_FILE

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create the table if it doesn't exist
create_table_query = """
        CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY,
            title TEXT,
            meta_description TEXT,
            og_title TEXT,
            og_description TEXT,
            h1 TEXT,
            h2 TEXT,
            h3 TEXT,
            visible_text TEXT
        )
"""

try:
    cursor.execute(create_table_query)
    conn.commit()
    print("✅ Table 'website_keywords' created successfully (or already exists).")
except sqlite3.Error as e:
    print(f"❌ SQLite error: {e}")
finally:
    conn.close()
