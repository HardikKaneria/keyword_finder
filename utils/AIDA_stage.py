import pandas as pd
import os
import numpy as np
import sqlite3
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')  

aida_chunks = {
    "Awareness": [
        "The Awareness stage is about recognition of a need or emotional shift. The person isn't looking for solutions yet.",
        "Emotionally: They may feel curious, mildly confused, or uncertain. Often there's a subtle discomfort or openness.",
        "Psychologically: Self-reflection, naming emotions, and looking for validation without urgency.",
        "Examples: 'Why do I feel emotionally drained?', 'What is emotional resilience?', 'Is this normal?'"
    ],
    "Interest": [
        "The Interest stage is about active exploration. The person wants to learn more about something they care about.",
        "Emotionally: They feel hopeful, engaged, or validated. There's motivation to explore and clarity in direction.",
        "Psychologically: They compare options, seek insights, and move toward possible solutions.",
        "Examples: 'How can I build emotional resilience?', 'Tips for better work-life balance?', 'How to manage stress?'"
    ]
}

def get_mean_embedding(text_chunks, model):
    vectors = [model.encode(chunk.strip()) for chunk in text_chunks if chunk.strip()]
    return np.mean(vectors, axis=0)

aida_embeddings = {
    stage: get_mean_embedding(chunks, model)
    for stage, chunks in aida_chunks.items()
}

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classify_keyword_stage(df):
    results = []

    for keyword in df['Keyword']:  # <-- fixed capitalization
        keyword_context = (
            f"The following phrase reflects a person's inner state or search: '{keyword}'. "
            "Consider their likely emotions, intent, and clarity of thought. "
            "Is this phrase more about becoming aware of a feeling or question (Awareness), "
            "or is it driven by a deeper curiosity to explore and understand potential solutions (Interest)? "
            "Classify this phrase accordingly based on the AIDA model."
        )

        keyword_vec = model.encode(keyword_context)
        raw_scores = [util.cos_sim(keyword_vec, emb).item() for emb in aida_embeddings.values()]
        probabilities = softmax(np.array(raw_scores))
        stage_list = list(aida_embeddings.keys())
        best_stage = stage_list[np.argmax(probabilities)]
        confidence = max(probabilities)

        if confidence < 0.45:
            stage = "Uncertain"
        else:
            stage = best_stage

        results.append({
            "Keyword": keyword,
            "Stage": stage,
            "ConfidenceScores": dict(zip(stage_list, probabilities))
        })

    return pd.DataFrame(results)

print("✅ AIDA classification complete. Data saved to 'aida_output' table in keyword_data.db")
