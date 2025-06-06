from pytrends.request import TrendReq
import pandas as pd
import time

def fetch_trends_data_batch(keywords, timeframe="today 3-m", geo=""):
    pytrends = TrendReq(hl='en-US', tz=330)
    trend_scores = {}
    trend_series_data = {}

    batch_size = 5
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i:i + batch_size]
        try:
            pytrends.build_payload(batch, timeframe=timeframe, geo=geo)
            df = pytrends.interest_over_time()

            if df.empty:
                print(f"⚠️ No data returned for batch: {batch}")
                for kw in batch:
                    trend_scores[kw] = 0
                    trend_series_data[kw] = []
                continue

            for kw in batch:
                if kw in df.columns:
                    trend_scores[kw] = round(df[kw].mean(), 2)
                    # Save full time series as list of tuples (date, score)
                    trend_series_data[kw] = list(zip(df.index.astype(str), df[kw].tolist()))
                else:
                    trend_scores[kw] = 0
                    trend_series_data[kw] = []

        except Exception as e:
            print(f"❌ Exception for batch {batch}: {e}")
            for kw in batch:
                trend_scores[kw] = 0
                trend_series_data[kw] = []

        time.sleep(1)

    # Merge into a single DataFrame
    df_scores = pd.DataFrame({
        "Keyword": list(trend_scores.keys()),
        "trend_score": list(trend_scores.values()),
        "trend_series": [trend_series_data[kw] for kw in trend_scores.keys()]
    })

    return df_scores
