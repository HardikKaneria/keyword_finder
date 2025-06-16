from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import pandas as pd
from services.keyword_fetcher import fetch_keyword_metrics_from_url
from utils.text_analysis import perform_sentiment_analysis, calculate_keyword_scores
from utils.keyword_content_matcher import calculate_chunked_keyword_scores
from utils.intent_classifier import bulk_classify_keyword_intents
from services.data_store import get_combined_website_texts

router = APIRouter()

class URLRequest(BaseModel):
    url: str

@router.post("/analyze-url")
def analyze_url(data: URLRequest):
    try:
        results = fetch_keyword_metrics_from_url(
            url=data.url,
            full_site=True,
            min_traffic=1000
        )
        if not results:
            return {"error": "No keywords above min_traffic"}

        df_metrics = pd.DataFrame(results)

        keywords = df_metrics["Keyword"].tolist()

        df_sentiment = perform_sentiment_analysis(keywords)
        df_intent = bulk_classify_keyword_intents(keywords)
        df_sentiment["Intent"] = df_intent

        df_combined = df_metrics.merge(df_sentiment, on="Keyword", how="left")

        df_pages = get_combined_website_texts()
        df_relevance = calculate_chunked_keyword_scores(df_combined.copy(), df_pages)

        df_final = df_combined.merge(df_relevance, on="Keyword", how="left")

        df_final = calculate_keyword_scores(df_final)

        print(f"✅ Successfully analyzed URL: {data.url}")


        return df_final.to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
