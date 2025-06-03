import os
import sqlite3
import pandas as pd
import streamlit as st
import json
import openai
from dotenv import load_dotenv


# === Load environment and set constants ===
load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../"))  # 
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
os.makedirs(EXPORT_DIR, exist_ok=True)
WEBSITE_FILE = os.path.join(DATA_DIR, "website_data.db")
DB_FILE = os.path.join(DATA_DIR, "keyword_data.db")
openai.api_key = os.getenv("OPENAI_API_KEY")

# === Helper Functions ===
def load_website_texts():
    try:
        conn = sqlite3.connect(WEBSITE_FILE)
        df = pd.read_sql("SELECT url, visible_text FROM pages", conn)
        conn.close()
        return "\n".join(f"{row['url']}: {row['visible_text']}" for _, row in df.iterrows())
    except Exception as e:
        st.error(f"Failed to load website text: {e}")
        return ""

def get_keywords():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql("SELECT DISTINCT Keyword FROM aida_output", conn)
    conn.close()
    return df["Keyword"].tolist()

def generate_aida_content_plan(df, use_ChatGPT=False):
    results = []

    for _, row in df.iterrows():
        keyword = row["Keyword"]
        stage = row["Stage"]
        media = row["Media"]
        platform = row["Platform"]
        tone = row["Tone"]
        marketing_intent = row["Marketing_Intent"]
        confidence_scores = row["ConfidenceScores"]

        if use_ChatGPT:
            prompt = f"""
You are a senior content strategist working with the AIDA marketing model.

The AIDA Stage is: {stage}
The keyword is: \"{keyword}\"
The Media Format is: {media}
The Platform is: {platform}
The Tone Cluster is: {tone}
The Marketing Intent is: {marketing_intent}

Based on this stage, use the following rules and context to generate a strategic content plan.

Step-by-step:

1. Suggest 2 content themes that best reflect the emotional or informational angle of the keyword within the AIDA stage. Make them unique and human-centered.

Return only this JSON:
{{
"Theme Option 1": "...",
"Theme Option 2": "..."
}}
"""
            try:
                response = openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500,
                )


                # Parse response
                content = response.choices[0].message.content.strip()
                json_result = json.loads(content)

                theme1 = json_result.get("Theme Option 1")
                theme2 = json_result.get("Theme Option 2")

            except Exception as e:
                st.warning(f"❌ Error for '{keyword}': {e}")
                theme1 = None
                theme2 = None
        else:
            theme1 = None
            theme2 = None

        results.append({
            "Keyword": keyword,
            "AIDA Stage": stage,
            "Media Format": media,
            "Platform": platform,
            "Tone Cluster": tone,
            "Marketing_Intent": marketing_intent,
            "ConfidenceScores": confidence_scores,
            "Theme Option 1": theme1,
            "Theme Option 2": theme2
        })

    return pd.DataFrame(results)