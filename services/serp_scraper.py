import requests
from bs4 import BeautifulSoup
import re
import os
import pickle
from googlesearch import search

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "scrape_cache.pkl")

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "rb") as f:
        SCRAPE_CACHE = pickle.load(f)
        print(f"🧠 Loaded {len(SCRAPE_CACHE)} cached URLs.")
else:
    SCRAPE_CACHE = {}

def is_html_url(url):
    """Skip URLs that point to files (pdf, image, doc, etc)."""
    return not re.search(r"\.(pdf|docx?|xlsx?|pptx?|jpg|jpeg|png|gif|svg|webp|mp4|zip|rar|ico|txt|json|csv)$", url, re.IGNORECASE)

def extract_text_from_url(url, retries=2):
    if not is_html_url(url):
        print(f"❌ Skipping non-HTML file URL: {url}")
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            content_type = response.headers.get("Content-Type", "")
            if "html" not in content_type.lower():
                print(f"❌ Skipping non-HTML content type: {content_type}")
                return ""

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else ""
            headings = " ".join(h.get_text() for h in soup.find_all(['h1', 'h2', 'h3']))
            paragraphs = " ".join(p.get_text() for p in soup.find_all('p'))
            return f"{title} {headings} {paragraphs}".lower()
        except Exception as e:
            print(f"⚠️ Retry {attempt + 1} failed for {url}: {e}")
    return ""

def extract_text_from_url_cached(url):
    if url in SCRAPE_CACHE:
        return SCRAPE_CACHE[url]
    content = extract_text_from_url(url)
    SCRAPE_CACHE[url] = content
    return content
    
def classify_keyword_intent_with_scrape(keyword: str) -> str:
    from collections import defaultdict
    keyword = keyword.lower()

    info_mods = [
        "how to", "what is", "why is", "why do", "when to", "when is", "where to", "where is", "who is",
        "guide", "tutorial", "tips", "tricks", "strategy", "ideas", "methods", "ways to", "benefits of",
        "explained", "definition", "meaning", "examples", "introduction to", "learn", "resources", "manual",
        "overview", "principles", "framework", "types of", "history of", "step by step", "step-by-step",
        "explanation of", "for beginners", "beginner’s guide", "understanding", "getting started with",
        "how it works", "how does", "how do", "tips and tricks", "best practices"
    ]
    trans_mods = [
        "buy", "purchase", "order", "book", "shop", "download", "install", "get", "subscribe", "sign up",
        "free trial", "apply now", "join now", "checkout", "pricing", "cost", "cheap", "affordable", "sale",
        "deal", "discount", "offer", "coupon", "promo code", "best price", "lowest price", "get started",
        "start now", "rent", "register", "claim", "add to cart", "compare prices", "limited time", "buy now",
        "in stock", "book now", "available now", "schedule", "near me", "free shipping", "buy online"
    ]
    comm_mods = [
        "best", "top", "compare", "comparison", "vs", "versus", "alternative", "alternatives", "review",
        "reviews", "pros and cons", "features", "benefits", "ranked", "ratings", "analysis", "editor’s pick",
        "editor's choice", "recommended", "trusted", "top rated", "side-by-side", "buyer’s guide",
        "worth it", "is it good", "should you buy", "feature comparison", "2023", "2024", "2025", "latest",
        "new", "updated", "top 10", "comparison chart", "best value", "brand comparison", "user feedback",
        "which is better", "top picks", "buyer reviews", "user reviews", "detailed comparison"
    ]
    nav_mods = [
        "facebook", "youtube", "instagram", "linkedin", "twitter", "tiktok", "login", "sign in",
        "dashboard", "homepage", "official", "official site", "app", "website", "site", "platform",
        "brand name", "amazon", "flipkart", "spotify", "netflix", "apple", "openai", "notion",
        "login page", "access", "portal", "support page", "account page", "my account", "admin panel",
        "sign in to", "app store", "google play", "download page", "official website"
    ]
    conv_mods = [
        "should i", "can i", "do i need", "how do you", "what do you think", "is it okay to", "how can i",
        "can we", "why do we", "does it matter", "is it wrong to", "how does it feel to", "am i supposed to",
        "how do i know if", "is it bad to", "do i have to", "what should", "do you know", "tell me about",
        "how are people", "what happens if", "how would you", "how can you", "what’s the best way to",
        "why is it important", "how long should", "what happens when", "what would happen if", "can someone",
        "what is the point of", "how do people", "what do experts say"
    ]

    modifiers = {
        "informational": info_mods,
        "transactional": trans_mods,
        "commercial": comm_mods,
        "navigational": nav_mods,
        "conversational": conv_mods
    }

    def weighted_token_match(text, mod_list):
        # Use weighted matching based on length and word boundary
        score = 0
        for mod in mod_list:
            count = len(re.findall(rf"\b{re.escape(mod)}\b", text))
            score += count * len(mod.split())  # longer phrases = more weight
        return score

    # Step 1: Token-level weighted match
    token_scores = {intent: weighted_token_match(keyword, mod_list) for intent, mod_list in modifiers.items()}
    if any(token_scores.values()):
        return max(token_scores, key=token_scores.get)

    # Step 2: SERP-based scoring with content analysis
    try:
        if keyword in SCRAPE_CACHE and isinstance(SCRAPE_CACHE[keyword], list):
            urls = SCRAPE_CACHE[keyword]
        else:
            urls = list(search(keyword, num_results=5))
            SCRAPE_CACHE[keyword] = urls

        combined_scores = defaultdict(int)

        for url in urls:
            text = extract_text_from_url_cached(url)
            for intent, mod_list in modifiers.items():
                combined_scores[intent] += weighted_token_match(text, mod_list)

        if any(combined_scores.values()):
            return max(combined_scores, key=combined_scores.get)

    except Exception as e:
        print(f"⚠️ SERP scrape-based fallback failed for '{keyword}': {e}")

    return "unknown"