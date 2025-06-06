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

# --- Config ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)"
]
PROXIES = [None]
CRAWL_DELAY = 0.1

def get_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, "../"))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    DB_FILE = os.path.join(data_dir, "website_data.db")
    return project_root, data_dir, DB_FILE

BASE_DIR, DATA_DIR, DB_FILE = get_paths()

# --- Browser ---
def get_browser(headless=True):
    options = Options()
    if headless:
        options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    return webdriver.Chrome(options=options)

# --- Robots.txt ---
def is_allowed_by_robots(url):
    rp = urllib.robotparser.RobotFileParser()
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch("*", url)
    except:
        return True

# --- URL validation ---
def is_valid_url(url, domain):
    parsed = urlparse(url)
    return (
        parsed.scheme in ("http", "https") and
        domain in parsed.netloc and
        not re.search(r"\.(jpg|jpeg|png|gif|pdf|svg|js|css|webp|mp4|zip|woff|ico)$", parsed.path, re.IGNORECASE)
    )

# --- Page scraping ---
def extract_content(url, use_selenium=False):
    try:
        headers = {"User-Agent": random.choice(USER_AGENTS)}
        proxy = {"http": random.choice(PROXIES), "https": random.choice(PROXIES)} if PROXIES[0] else None

        if use_selenium:
            browser = get_browser()
            browser.get(url)
            html = browser.page_source
            browser.quit()
        else:
            resp = requests.get(url, headers=headers, proxies=proxy, timeout=10)
            resp.raise_for_status()
            html = resp.text

        soup = BeautifulSoup(html, "lxml")
        title = soup.title.string.strip() if soup.title else ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        og_title = soup.find("meta", attrs={"property": "og:title"})
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        headings = {
            "h1": [h.get_text(strip=True) for h in soup.find_all("h1")],
            "h2": [h.get_text(strip=True) for h in soup.find_all("h2")],
            "h3": [h.get_text(strip=True) for h in soup.find_all("h3")],
        }
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        visible_text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))

        return {
            "url": url,
            "title": title,
            "meta_description": meta_desc["content"].strip() if meta_desc else "",
            "og_title": og_title["content"].strip() if og_title else "",
            "og_description": og_desc["content"].strip() if og_desc else "",
            "h1": "; ".join(headings["h1"]),
            "h2": "; ".join(headings["h2"]),
            "h3": "; ".join(headings["h3"]),
            "visible_text": visible_text
        }
    except:
        return None

# --- Save to DB ---
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

# --- Load ---
def load_data():
    conn = sqlite3.connect(DB_FILE)
    try:
        df = pd.read_sql_query("SELECT * FROM pages ORDER BY rowid DESC", conn)
    except:
        df = None
    conn.close()
    return df

# --- Clear DB ---
def clear_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM pages")
    conn.commit()
    conn.close()

# --- Crawl Logic ---
def crawl(start_url, use_selenium=False, status_area=None, max_depth=2, progress_bar=None, stats_box=None):
    visited = set()
    domain = urlparse(start_url).netloc
    queue = deque([(start_url, 0)])
    page_counter = 0
    start_time = time.time()
    total_estimate = 100

    while queue:
        current_url, depth = queue.popleft()
        current_url = current_url.rstrip('/')
        if current_url in visited or depth > max_depth:
            continue
        visited.add(current_url)

        if not is_allowed_by_robots(current_url):
            continue

        page_data = extract_content(current_url, use_selenium=use_selenium)
        if page_data:
            save_to_db(page_data)
            page_counter += 1

        elapsed = time.time() - start_time
        estimated_total = (elapsed / page_counter) * (len(queue) + page_counter) if page_counter else 1
        remaining = estimated_total - elapsed

        if status_area:
            status_area.markdown(f"✅ Crawled: `{current_url}`")
        if stats_box:
            stats_box.markdown(
                f"""
                - Pages Crawled: `{page_counter}`  
                - Queue Size: `{len(queue)}`  
                - Elapsed Time: `{int(elapsed)}s`  
                - Est. Remaining: `{int(remaining)}s`
                """
            )
        if progress_bar:
            progress_bar.progress(min(page_counter / total_estimate, 1.0))

        try:
            headers = {"User-Agent": random.choice(USER_AGENTS)}
            if use_selenium:
                browser = get_browser()
                browser.get(current_url)
                html = browser.page_source
                browser.quit()
            else:
                html = requests.get(current_url, headers=headers, timeout=10).text

            soup = BeautifulSoup(html, "lxml")
            for a_tag in soup.find_all("a", href=True):
                raw_link = urljoin(current_url, a_tag["href"])
                parsed_link = urlparse(raw_link)

                # Skip fragment or query links
                if parsed_link.fragment or parsed_link.query:
                    continue

                clean_link = parsed_link._replace(fragment="", query="").geturl().rstrip('/')

                if is_valid_url(clean_link, domain) and clean_link not in visited:
                    queue.append((clean_link, depth + 1))
        except:
            continue

        time.sleep(CRAWL_DELAY)

    return page_counter

# --- UI ---
def run_crawler_ui():
    st.title("🕸️ Website Crawler")
    url_input = st.text_input("Enter Homepage URL", "https://humanness.ing/")
    use_selenium = st.checkbox("Enable JavaScript Rendering (via Selenium)", value=True)
    max_depth = st.slider("Max Crawl Depth", 1, 10, value=2)
    start_crawl = st.button("🚀 Start Crawling")

    status_area = st.empty()
    stats_box = st.empty()
    progress_bar = st.progress(0)

    if start_crawl and url_input:
        clear_db()
        status_area.markdown("🔄 Starting crawl...")
        page_count = crawl(
            start_url=url_input,
            use_selenium=use_selenium,
            status_area=status_area,
            max_depth=max_depth,
            progress_bar=progress_bar,
            stats_box=stats_box
        )
        st.success(f"🎉 Crawl Complete! Total Pages: {page_count}")

    st.markdown("---")
    st.subheader("📄 Crawled Pages")
    df = load_data()

    if df is not None and not df.empty:
        st.markdown(f"**Total pages stored: {len(df)}**")

        # === Pagination Setup ===
        rows_per_page = 100
        total_rows = len(df)
        total_pages = (total_rows - 1) // rows_per_page + 1

        page_number = st.number_input("Page", min_value=1, max_value=total_pages, step=1)
        start_idx = (page_number - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)

        # === Paginated View ===
        st.markdown(f"**Showing rows {start_idx + 1} to {end_idx} of {total_rows}**")
        st.dataframe(
            df.loc[start_idx:end_idx - 1, ['url', 'title', 'meta_description', 'h1', 'h2', 'h3', 'visible_text']],
            use_container_width=True
        )
    else:
        st.info("No data available yet. Run a crawl first.")


# --- Entry ---
if __name__ == '__main__':
    run_crawler_ui()