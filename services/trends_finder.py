from pytrends.request import TrendReq
import pandas as pd
import time
import random
import logging

def fetch_trends_data_batch(keywords, timeframe="today 3-m", geo="", status_callback=None, progress_bar=None):
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

    trend_scores = {}
    trend_series_data = {}

    batch_size = 4
    max_retries = 1
    total = len(keywords)
    processed = 0
    early_exit_due_to_429 = False

    for i in range(0, total, batch_size):
        batch = keywords[i:i + batch_size]
        success = False
        retries = 0

        while not success and retries < max_retries:
            try:
                pytrends = TrendReq(
                    hl='en-US',
                    tz=330,
                    requests_args={
                        "headers": {
                            "User-Agent": random.choice(user_agents),
                            "Accept-Language": "en-US,en;q=0.9",
                        }
                    }
                )
                pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
                df = pytrends.interest_over_time()

                for j, kw in enumerate(batch):
                    global_index = i + j
                    if df.empty or kw not in df.columns:
                        trend_scores[kw] = 0
                        trend_series_data[kw] = []
                        if status_callback:
                            status_callback.text(f"[{global_index + 1}/{total}] ⚠️ {kw}: No data")
                        logging.warning(f"No data for keyword: {kw}")
                    else:
                        score = round(df[kw].mean(), 2)
                        series = list(zip(df.index.astype(str), df[kw].tolist()))
                        trend_scores[kw] = score
                        trend_series_data[kw] = series
                        if status_callback:
                            status_callback.text(f"[{global_index + 1}/{total}] ✅ {kw}: Trend Score {score}")

                    processed += 1
                    if progress_bar:
                        progress_bar.progress(processed / total)

                success = True  # exit retry loop

            except Exception as e:
                retries += 1
                if "429" in str(e):
                    if i == 0:  # First batch
                        early_exit_due_to_429 = True
                        if status_callback:
                            status_callback.text("🚫 429 on first batch. Stopping all trend fetching immediately.")
                        logging.error("❌ 429 on first batch — aborting entire trend job.")
                        return pd.DataFrame(columns=["Keyword", "trend_score", "trend_series"])
                    else:
                        if status_callback:
                            status_callback.text("⚠️ 429 error. Skipping current batch.")
                        logging.warning("429 after first batch — skipping only this batch.")
                        break  # Skip this batch, continue to next

                wait = 10 * (2 ** retries) + random.uniform(5, 10)
                if status_callback:
                    status_callback.text(f"🔁 Retry {retries}/{max_retries} for batch {batch} — Waiting {int(wait)}s")
                logging.warning(f"Retry {retries}/{max_retries} for batch {batch} — Error: {e}")
                time.sleep(wait)

        if not success and not early_exit_due_to_429:
            for j, kw in enumerate(batch):
                global_index = i + j
                trend_scores[kw] = 0
                trend_series_data[kw] = []
                if status_callback:
                    status_callback.text(f"[{global_index + 1}/{total}] ❌ {kw}: Failed after retries")
                processed += 1
                if progress_bar:
                    progress_bar.progress(processed / total)

        # Delay between batches only if continuing
        if not early_exit_due_to_429:
            batch_delay = 25 + random.uniform(5, 10)
            logging.info(f"Sleeping {int(batch_delay)}s before next batch...")
            time.sleep(batch_delay)

    df_scores = pd.DataFrame({
        "Keyword": list(trend_scores.keys()),
        "trend_score": list(trend_scores.values()),
        "trend_series": [trend_series_data[kw] for kw in trend_scores.keys()]
    })

    return df_scores