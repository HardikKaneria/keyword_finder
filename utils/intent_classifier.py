import re
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from googlesearch import search

# === Modifier Phrase Sets (Lowercase) ===
INFO_MODS = set([
    "how to", "what is", "why is", "why do", "when to", "when is", "where to", "where is", "who is",
    "guide", "tutorial", "tips", "tricks", "strategy", "ideas", "methods", "ways to", "benefits of",
    "explained", "definition", "meaning", "examples", "introduction to", "learn", "resources", "manual",
    "overview", "principles", "framework", "types of", "history of", "step by step", "step-by-step",
    "explanation of", "for beginners", "beginner’s guide", "understanding", "getting started with",
    "how it works", "how does", "how do", "tips and tricks", "best practices"
])
TRANS_MODS = set([
    "buy", "purchase", "order", "book", "shop", "download", "install", "get", "subscribe", "sign up",
    "free trial", "apply now", "join now", "checkout", "pricing", "cost", "cheap", "affordable", "sale",
    "deal", "discount", "offer", "coupon", "promo code", "best price", "lowest price", "get started",
    "start now", "rent", "register", "claim", "add to cart", "compare prices", "limited time", "buy now",
    "in stock", "book now", "available now", "schedule", "near me", "free shipping", "buy online"
])
COMM_MODS = set([
    "best", "top", "compare", "comparison", "vs", "versus", "alternative", "alternatives", "review",
    "reviews", "pros and cons", "features", "benefits", "ranked", "ratings", "analysis", "editor’s pick",
    "editor's choice", "recommended", "trusted", "top rated", "side-by-side", "buyer’s guide",
    "worth it", "is it good", "should you buy", "feature comparison", "2023", "2024", "2025", "latest",
    "new", "updated", "top 10", "comparison chart", "best value", "brand comparison", "user feedback",
    "which is better", "top picks", "buyer reviews", "user reviews", "detailed comparison"
])
NAV_MODS = set([
    "facebook", "youtube", "instagram", "linkedin", "twitter", "tiktok", "login", "sign in",
    "dashboard", "homepage", "official", "official site", "app", "website", "site", "platform",
    "amazon", "flipkart", "spotify", "netflix", "apple", "openai", "notion",
    "login page", "access", "portal", "support page", "account page", "my account", "admin panel",
    "sign in to", "app store", "google play", "download page", "official website"
])
CONV_MODS = set([
    "should i", "can i", "do i need", "how do you", "what do you think", "is it okay to", "how can i",
    "can we", "why do we", "does it matter", "is it wrong to", "how does it feel to", "am i supposed to",
    "how do i know if", "is it bad to", "do i have to", "what should", "do you know", "tell me about",
    "how are people", "what happens if", "how would you", "how can you", "what’s the best way to",
    "why is it important", "how long should", "what happens when", "what would happen if", "can someone",
    "what is the point of", "how do people", "what do experts say"
])

INTENT_SETS = {
    "informational": INFO_MODS,
    "transactional": TRANS_MODS,
    "commercial": COMM_MODS,
    "navigational": NAV_MODS,
    "conversational": CONV_MODS,
}

# === Main Classification Function ===
def classify_keyword_intent(keyword: str, use_serp_fallback=True) -> str:
    kw_lc = keyword.lower()

    # Scoring based on modifiers
    score = {
        intent: sum(phrase in kw_lc for phrase in phrases)
        for intent, phrases in INTENT_SETS.items()
    }

    if all(v == 0 for v in score.values()) and use_serp_fallback:
        try:
            serp_urls = list(search(keyword, num_results=10))
            serp_text = " ".join(serp_urls).lower()
            if "youtube" in serp_text or "facebook" in serp_text:
                return "navigational"
            elif "how" in serp_text or "guide" in serp_text or "blog" in serp_text:
                return "informational"
            elif "compare" in serp_text or "vs" in serp_text or "review" in serp_text:
                return "commercial"
            elif "buy" in serp_text or "price" in serp_text or "deal" in serp_text:
                return "transactional"
        except Exception:
            pass
        return "unknown"

    return max(score, key=score.get)

# === LRU Cache Wrapper ===
@lru_cache(maxsize=10000)
def classify_keyword_intent_cached(keyword: str) -> str:
    return classify_keyword_intent(keyword, use_serp_fallback=False)

# === Bulk Processor ===
def bulk_classify_keyword_intents(keywords, use_threads=True, max_workers=8, progress_bar=None, status_callback=None):
    total = len(keywords)
    results = []

    if use_threads:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(classify_keyword_intent_cached, kw): kw for kw in keywords}
            for i, future in enumerate(futures):
                kw = futures[future]
                if status_callback:
                    status_callback.text(f"🧠 {i + 1}/{total} – Intent Classifying: {kw}")
                try:
                    results.append(future.result())
                except Exception:
                    results.append("unknown")
                if progress_bar:
                    progress_bar.progress((i + 1) / total)
    else:
        for i, kw in enumerate(keywords):
            if status_callback:
                status_callback.text(f"🧠 {i + 1}/{total} – Intent Classifying: {kw}")
            try:
                results.append(classify_keyword_intent_cached(kw))
            except Exception:
                results.append("unknown")
            if progress_bar:
                progress_bar.progress((i + 1) / total)
            
    if progress_bar:
        progress_bar.progress(1.0)

    if status_callback:
        status_callback.text("✅ Intent Classification Complete!")

    return results