from pytrends.request import TrendReq
import pandas as pd
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

PROXY_URL = "http://scrapeops:568dabc3-ea1b-457b-ac3d-24241d12440b@residential-proxy.scrapeops.io:8181"

user_agents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
    'Mozilla/5.0 (X11; Linux x86_64)',
    'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
    'Mozilla/5.0 (iPad; CPU OS 13_6_1 like Mac OS X)',
    'Mozilla/5.0 (Linux; Android 9; SM-G960F)',
    'Mozilla/5.0 (Windows NT 10.0; WOW64)',
]

logging.basicConfig(level=logging.INFO)

def fetch_trend_for_batch(batch, timeframe, geo):
    try:
        pytrends = TrendReq(
            hl='en-US',
            tz=330,
            requests_args={
                "headers": {
                    "User-Agent": random.choice(user_agents),
                    "Accept-Language": "en-US,en;q=0.9",
                },
                "proxies": {
                    "http": PROXY_URL,
                    "https": PROXY_URL,
                },
                "timeout": 15
            }
        )

        pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
        df = pytrends.interest_over_time()

        scores = []
        for kw in batch:
            if df.empty or kw not in df.columns:
                scores.append({"Keyword": kw, "trend_score": 0, "trend_series": []})
            else:
                score = round(df[kw].mean(), 2)
                series = list(zip(df.index.astype(str), df[kw].tolist()))
                scores.append({"Keyword": kw, "trend_score": score, "trend_series": series})

        return scores

    except Exception as e:
        logging.warning(f"Failed batch {batch}: {e}")
        return [{"Keyword": kw, "trend_score": 0, "trend_series": []} for kw in batch]

def fetch_trends_data_batch(keywords, timeframe="today 3-m", geo="", max_workers=4):
    batch_size = 4
    all_results = []
    batches = [keywords[i:i + batch_size] for i in range(0, len(keywords), batch_size)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch_trend_for_batch, batch, timeframe, geo): batch for batch in batches}

        for future in as_completed(futures):
            result = future.result()
            all_results.extend(result)

    df_scores = pd.DataFrame(all_results)
    return df_scores