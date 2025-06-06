from sklearn.feature_extraction.text import TfidfVectorizer

def extract_keywords_from_text(text, max_keywords=200):
    try:
        text = text[:3000]  # Limit size for performance
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000,
            ngram_range=(1, 3),  # Extract unigrams, bigrams, and trigrams
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Avoid single characters/numbers
        )
        tfidf_matrix = vectorizer.fit_transform([text])
        scores = tfidf_matrix.toarray().flatten()
        keywords = vectorizer.get_feature_names_out()

        # Sort by score and return top N keywords
        keyword_scores = sorted(zip(keywords, scores), key=lambda x: x[1], reverse=True)
        return [kw for kw, _ in keyword_scores[:max_keywords]]
    except Exception as e:
        print(f"❌ Error extracting keywords: {e}")
        return []
