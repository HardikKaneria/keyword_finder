import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')  

aida_chunks = {
  "Attention": [
    "The Attention stage is about recognition of a need or emotional shift. The person isn't looking for solutions yet.",
    "Emotionally: They may feel curious, mildly confused, or uncertain. Often there's a subtle discomfort or openness.",
    "Psychologically: Self-reflection, naming emotions, and looking for validation without urgency.",
    "Examples: 'Why do I feel off lately?', 'What is burnout?', 'Is this normal behavior for a child?'"
  ],
  "Interest": [
    "The Interest stage is about active exploration. The person wants to learn more about something they care about.",
    "Emotionally: They feel hopeful, engaged, or validated. There's motivation to explore and clarity in direction.",
    "Psychologically: They compare, research, and move toward understanding. They’re open to insights and perspectives.",
    "Examples: 'How can I improve sleep quality?', 'Best practices for remote teams', 'What are the signs of emotional intelligence?'"
  ],
  "Desire": [
    "The Desire stage is about personal resonance and alignment. The person connects emotionally or logically with a possibility, product, or idea.",
    "Emotionally: They feel inspired, understood, or excited. There’s a sense that this could be meaningful for them.",
    "Psychologically: They evaluate benefits, imagine outcomes, and consider how well something fits their needs.",
    "Examples: 'Top-rated meditation apps for beginners', 'Should I switch to a standing desk?', 'Is this parenting approach right for our family?'"
  ],
  "Action": [
    "The Action stage is about taking a step. The person is ready to say yes, sign up, try, or commit to something that feels right.",
    "Emotionally: They feel clear, grounded, and ready. Trust and confidence are high.",
    "Psychologically: They seek ease, assurance, and a clear next step. The choice should feel aligned and accessible.",
    "Examples: 'Subscribe now', 'Book your free consultation', 'Start your 7-day trial'"
  ]
}

def get_mean_embedding(text_chunks, model):
    vectors = model.encode([chunk.strip() for chunk in text_chunks if chunk.strip()])
    return np.mean(vectors, axis=0)

aida_embeddings = {
    stage: get_mean_embedding(chunks, model)
    for stage, chunks in aida_chunks.items()
}
# Then convert to tensor format
for stage in aida_embeddings:
    aida_embeddings[stage] = model.encode(aida_chunks[stage], convert_to_tensor=True).mean(dim=0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def classify_keyword_stage(df):
    results = []

    for keyword in df['Keyword']:  # <-- fixed capitalization
        keyword_context = (
            f"The following phrase reflects a moment in someone's inner journey: '{keyword}'. "
            "Consider their likely emotional state, clarity of thought, and intent. "
            "Classify it into one of the four AIDA stages based on the following: \n\n"
            "• 'Attention' — The person is just becoming aware of a feeling, problem, or inner shift. "
            "They're not looking for solutions yet — just naming or noticing something. Look for curiosity, confusion, or quiet discomfort.\n"
            "• 'Interest' — The person is actively exploring or learning more. They are seeking understanding, insights, or relatable ideas. "
            "There’s forward motion, but still no commitment. Look for motivated curiosity or reflective research.\n"
            "• 'Desire' — The person feels emotionally aligned with a solution or idea. They are evaluating options, imagining outcomes, or considering how something might help them personally. "
            "Look for resonance, emotional pull, or readiness to go deeper.\n"
            "• 'Action' — The person is ready to take a concrete step. They are choosing to try, buy, join, sign up, or commit. "
            "Look for clarity, urgency, or direct intent to act.\n\n"
            "Use the emotional and psychological clues within the phrase to determine which stage it most likely belongs to."
        )

        keyword_vec = model.encode(keyword_context, convert_to_tensor=True)
        if keyword_vec is None:
            continue
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
