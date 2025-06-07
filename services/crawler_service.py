import streamlit as st
import os
import sqlite3
import random
import requests
import pandas as pd
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import urllib.robotparser
import re
import time
from collections import deque
from utils.keyword_extractor import extract_keywords_from_visible_text
# This script is a web crawler that extracts content from websites, stores it in a SQLite database, and provides a Streamlit UI for interaction.

# --- Config ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F)",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_2 like Mac OS X)"
]
PROXIES = [None]  # You can add proxy dicts here
CRAWL_DELAY = 0.05
MAX_RETRY_WAIT = 15

# --- Paths ---
def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../"))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    DB_FILE = os.path.join(data_dir, "website_data.db")
    return project_root, data_dir, DB_FILE

BASE_DIR, DATA_DIR, DB_FILE = get_paths()

# --- Browser ---
def get_browser():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    return webdriver.Chrome(options=options)

# --- Robots.txt ---
def is_allowed_by_robots(url):
    try:
        rp = urllib.robotparser.RobotFileParser()
        parsed = urlparse(url)
        rp.set_url(f"{parsed.scheme}://{parsed.netloc}/robots.txt")
        rp.read()
        return rp.can_fetch(random.choice(USER_AGENTS), url)
    except:
        return True

# --- URL Validation ---
def is_valid_url(url, domain):
    parsed = urlparse(url)
    return (
        parsed.scheme in ("http", "https") and
        domain in parsed.netloc and
        not re.search(r"\.(jpg|jpeg|png|gif|pdf|svg|js|css|webp|mp4|zip|woff|ico)$", parsed.path, re.IGNORECASE)
    )

def get_status_code(url):
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        response = requests.head(url, headers=headers, allow_redirects=True, timeout=5)
        return response.status_code
    except:
        return None

# --- Content Extraction ---
def extract_content(url, homepage_url, use_selenium=False):
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        proxy = {"http": PROXIES[0], "https": PROXIES[0]} if PROXIES[0] else None

        if use_selenium:
            browser = get_browser()
            browser.set_page_load_timeout(MAX_RETRY_WAIT)
            browser.get(url)
            final_url = browser.current_url
            html = browser.page_source
            browser.quit()
        else:
            resp = requests.get(url, headers=headers, proxies=proxy, timeout=MAX_RETRY_WAIT, allow_redirects=True)
            status = resp.status_code
            if status in [301, 302, 404]:
                return None
            html = resp.text
            final_url = resp.url

        if homepage_url.rstrip('/') == final_url.rstrip('/') and url.rstrip('/') != final_url.rstrip('/'):
            return None

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string.strip() if soup.title else ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        og_title = soup.find("meta", attrs={"property": "og:title"})
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        visible_text = re.sub(r"\\s+", " ", soup.get_text(separator=" ", strip=True))
        headings = {
            "h1": "; ".join([h.get_text(strip=True) for h in soup.find_all("h1")]),
            "h2": "; ".join([h.get_text(strip=True) for h in soup.find_all("h2")]),
            "h3": "; ".join([h.get_text(strip=True) for h in soup.find_all("h3")]),
        }
        return {
            "url": url,
            "title": title,
            "meta_description": meta_desc["content"].strip() if meta_desc else "",
            "og_title": og_title["content"].strip() if og_title else "",
            "og_description": og_desc["content"].strip() if og_desc else "",
            "h1": headings["h1"],
            "h2": headings["h2"],
            "h3": headings["h3"],
            "visible_text": visible_text
        }
    except:
        return None

# --- DB ---
def save_to_db(data):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
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
    """)
    c.execute("DELETE FROM pages WHERE url = ?", (data["url"],))
    c.execute("INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", tuple(data.values()))
    conn.commit()
    conn.close()

def clear_db():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("DELETE FROM pages")
    conn.commit()
    conn.close()

def load_data():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT * FROM pages ORDER BY rowid DESC", conn)
    except:
        df = None
    conn.close()
    return df

# --- Crawler Logic ---
def crawl(start_url, use_selenium=False, status_area=None, max_depth=2, progress_bar=None, stats_box=None):
    visited = set()
    retry_queue = []
    domain = urlparse(start_url).netloc
    queue = deque([(start_url, 0)])
    page_counter = 0
    start_time = time.time()
    total_estimate = 100

    while queue:
        current_url, depth = queue.popleft()
        if current_url in visited or depth > max_depth:
            continue

        status = get_status_code(current_url)
        if not status or status in [301, 302, 404]:
            continue

        visited.add(current_url)
        if not is_allowed_by_robots(current_url):
            continue

        content = extract_content(current_url, homepage_url=start_url, use_selenium=use_selenium)
        if content:
            save_to_db(content)
            page_counter += 1
        else:
            retry_queue.append((current_url, depth))

        if status_area:
            status_area.markdown(f"✅ Crawled: `{current_url}`")

        elapsed = time.time() - start_time
        remaining = (elapsed / page_counter * (len(queue) + page_counter)) - elapsed if page_counter else 0

        if stats_box:
            stats_box.markdown(f"Pages: `{page_counter}` | Queue: `{len(queue)}` | Time: `{int(elapsed)}s` | ETA: `{int(remaining)}s`")
        if progress_bar:
            progress_bar.progress(min(page_counter / total_estimate, 1.0))

        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            html = requests.get(current_url, headers=headers, timeout=10).text
            soup = BeautifulSoup(html, "lxml")
            for tag in soup.find_all("a", href=True):
                next_url = urljoin(current_url, tag['href'])
                parsed = urlparse(next_url)
                clean = parsed._replace(fragment="", query="").geturl().rstrip('/')
                if is_valid_url(clean, domain) and clean not in visited:
                    queue.append((clean, depth + 1))
        except:
            continue

        time.sleep(CRAWL_DELAY)

    for retry_url, depth in retry_queue:
        content = extract_content(retry_url, homepage_url=start_url, use_selenium=use_selenium)
        print(f"Retrying {retry_url} at depth {depth}...")
        if content:
            save_to_db(content)
            page_counter += 1

    return page_counter

# --- UI ---
def run_crawler_ui():
    st.title("🕸️ Smart Website Crawler")
    url_input = st.text_input("Enter Homepage URL", "https://humanness.ing/")
    use_selenium = st.checkbox("Enable JS Rendering (via Selenium)", value=True)
    max_depth = st.slider("Max Crawl Depth", 1, 10, value=2)
    start_crawl = st.button("🚀 Start Crawling")

    status_area = st.empty()
    stats_box = st.empty()
    progress_bar = st.progress(0)

    if start_crawl and url_input:
        clear_db()
        status_area.markdown("🔄 Starting crawl...")
        count = crawl(
            start_url=url_input,
            use_selenium=use_selenium,
            status_area=status_area,
            max_depth=max_depth,
            progress_bar=progress_bar,
            stats_box=stats_box
        )
        st.success(f"🎉 Completed! Pages Crawled: {count}")

    st.markdown("---")
    st.subheader("📄 Crawled Pages")
    df = load_data()
    if df is not None and not df.empty:
        rows_per_page = 100
        total_rows = len(df)
        page_number = st.number_input("Page", 1, max(1, (total_rows - 1) // rows_per_page + 1))
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        st.dataframe(df.loc[start_idx:end_idx-1, ['url', 'title', 'meta_description', 'h1', 'h2', 'h3', 'visible_text']], use_container_width=True)
    else:
        st.info("No data yet. Start a crawl to begin.")

    st.markdown("---")
    st.subheader("🧠 Keyword Extraction")

    if st.button("🔑 Extract Keywords from Visible Text"):
        extract_keywords_from_visible_text()
        st.success("✅ Keywords extracted and stored in `keywords_extraction` table.")


if __name__ == '__main__':
    run_crawler_ui()
