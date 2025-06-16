import os
import csv
import time
from tqdm import tqdm
from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException

# Configuration (can be externalized)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CREDENTIALS_FILE = os.path.join(BASE_DIR, "google-ads.yaml")
CUSTOMER_ID = "7719095294"
LOGIN_CUSTOMER_ID = "3478433292"
GEO_TARGET = "geoTargetConstants/2840"
LANGUAGE_ID = "1000"
SLEEP_BETWEEN_CALLS = 2


def read_keywords_from_csv(file_path):
    keywords = []
    try:
        with open(file_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                keyword = row.get("keyword")
                if keyword and keyword.strip():
                    keywords.append(keyword.strip())
    except Exception as e:
        print(f"❌ Failed to read CSV: {e}")
    return keywords


def fetch_keyword_metrics(keywords, progress_bar=None, status_text=None):
    try:
        client = GoogleAdsClient.load_from_storage(CREDENTIALS_FILE)
        client.login_customer_id = LOGIN_CUSTOMER_ID
        service = client.get_service("KeywordPlanIdeaService")
        network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
    except Exception as e:
        print(f"❌ Failed to initialize Google Ads client: {e}")
        return []

    results = []
    total = len(keywords)
    for i, kw in enumerate(keywords):
        if status_text:
            status_text.text(f"📦 {i + 1}/{total} – Fetching: {kw}")
        if progress_bar:
            progress_bar.progress((i + 1) / total)

        try:
            request = client.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = CUSTOMER_ID
            request.language = f"languageConstants/{LANGUAGE_ID}"
            request.geo_target_constants.append(GEO_TARGET)
            request.keyword_plan_network = network
            request.keyword_seed.keywords.append(kw)

            response = service.generate_keyword_ideas(request=request)
            for idea in response:
                metrics = idea.keyword_idea_metrics
                results.append({
                    "Source Keyword": kw,
                    "Keyword": idea.text,
                    "Avg Monthly Searches": metrics.avg_monthly_searches,
                    "Competition": metrics.competition.name,
                    "Low CPC (USD)": round(metrics.low_top_of_page_bid_micros / 1e6, 2) if metrics.low_top_of_page_bid_micros else 0.0,
                    "High CPC (USD)": round(metrics.high_top_of_page_bid_micros / 1e6, 2) if metrics.high_top_of_page_bid_micros else 0.0,
                })
        except GoogleAdsException as ex:
            print(f"❌ API error for '{kw}': {ex.failure.errors[0].message}")
        except Exception as ex:
            print(f"❌ Unexpected error for '{kw}': {ex}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        status_text.text("All keywords fetched!")

    return results

def fetch_keyword_metrics_from_url(url, full_site=False, progress_bar=None, status_text=None, min_traffic=5000):
    try:
        client = GoogleAdsClient.load_from_storage(CREDENTIALS_FILE)
        client.login_customer_id = LOGIN_CUSTOMER_ID
        service = client.get_service("KeywordPlanIdeaService")
        network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
    except Exception as e:
        print(f"❌ Failed to initialize Google Ads client: {e}")
        return []

    results = []

    try:
        if status_text:
            status_text.text(f"🌐 Fetching from URL: {url}")
        if progress_bar:
            progress_bar.progress(0.1)

        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = CUSTOMER_ID
        request.language = f"languageConstants/{LANGUAGE_ID}"
        request.geo_target_constants.append(GEO_TARGET)
        request.keyword_plan_network = network

        # ✅ Correct way to assign seed
        if full_site:
            request.url_seed.url = url
        else:
            request.keyword_and_url_seed.url = url

        response = service.generate_keyword_ideas(request=request)

        for idea in response:
            metrics = idea.keyword_idea_metrics
            monthly_searches = metrics.avg_monthly_searches

            if monthly_searches and monthly_searches >= min_traffic:
                results.append({
                    "Source URL": url,
                    "Keyword": idea.text,
                    "Avg Monthly Searches": monthly_searches,
                    "Competition": metrics.competition.name,
                    "Low CPC (USD)": round(metrics.low_top_of_page_bid_micros / 1e6, 2) if metrics.low_top_of_page_bid_micros else 0.0,
                    "High CPC (USD)": round(metrics.high_top_of_page_bid_micros / 1e6, 2) if metrics.high_top_of_page_bid_micros else 0.0,
                })

        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("✅ All keywords fetched!")

    except GoogleAdsException as ex:
        print(f"❌ API error for '{url}': {ex.failure.errors[0].message}")
    except Exception as ex:
        print(f"❌ Unexpected error for '{url}': {ex}")

    return results

def fetch_exact_keyword_metrics(keywords, progress_bar=None, status_text=None):
    try:
        client = GoogleAdsClient.load_from_storage(CREDENTIALS_FILE)
        client.login_customer_id = LOGIN_CUSTOMER_ID
        service = client.get_service("KeywordPlanIdeaService")
        network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH
    except Exception as e:
        print(f"❌ Failed to initialize Google Ads client: {e}")
        return []

    results = []
    total = len(keywords)

    for i, kw in enumerate(keywords):
        if status_text:
            status_text.text(f"📦 {i + 1}/{total} – Fetching: {kw}")
        if progress_bar:
            progress_bar.progress((i + 1) / total)

        try:
            request = client.get_type("GenerateKeywordIdeasRequest")
            request.customer_id = CUSTOMER_ID
            request.language = f"languageConstants/{LANGUAGE_ID}"
            request.geo_target_constants.append(GEO_TARGET)
            request.keyword_plan_network = network
            request.keyword_seed.keywords.append(kw)

            response = service.generate_keyword_ideas(request=request)
            found = False
            for idea in response:
                if idea.text.lower() == kw.lower():  # match exact keyword
                    metrics = idea.keyword_idea_metrics
                    results.append({
                        "Source Keyword": kw,
                        "Keyword": idea.text,
                        "Avg Monthly Searches": metrics.avg_monthly_searches,
                        "Competition": metrics.competition.name,
                        "Low CPC (USD)": round(metrics.low_top_of_page_bid_micros / 1e6, 2) if metrics.low_top_of_page_bid_micros else 0.0,
                        "High CPC (USD)": round(metrics.high_top_of_page_bid_micros / 1e6, 2) if metrics.high_top_of_page_bid_micros else 0.0,
                    })
                    found = True
                    break
            if not found:
                results.append({
                    "Source Keyword": kw,
                    "Keyword": kw,
                    "Avg Monthly Searches": 0,
                    "Competition": "UNKNOWN",
                    "Low CPC (USD)": 0.0,
                    "High CPC (USD)": 0.0,
                })

        except GoogleAdsException as ex:
            print(f"❌ API error for '{kw}': {ex.failure.errors[0].message}")
        except Exception as ex:
            print(f"❌ Unexpected error for '{kw}': {ex}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    if progress_bar:
        progress_bar.progress(1.0)
    if status_text:
        status_text.text("All keywords fetched!")

    return results
