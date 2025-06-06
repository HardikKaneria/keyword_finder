import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np


def cluster_keywords(keywords, num_clusters=None, progress_callback=None, status_callback=None):
    keywords = [kw for kw in keywords if kw.strip()]
    if not keywords:
        print("⚠️ No valid keywords to cluster.")
        return pd.DataFrame(columns=["Keyword", "Cluster"])

    if len(keywords) < 2:
        print("⚠️ Too few keywords to perform clustering. Returning single cluster.")
        return pd.DataFrame({
            "Keyword": keywords,
            "Cluster": ["Cluster 1"] * len(keywords)
        })

    if status_callback:
        status_callback.text("🔠 Embedding keywords with SentenceTransformer...")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(keywords)

    if progress_callback:
        progress_callback.progress(0.3)

    # === Auto-select optimal number of clusters ===
    if not num_clusters:
        if status_callback:
            status_callback.text("🔍 Finding optimal number of clusters...")

        max_k = min(10, len(keywords) - 1)
        best_k = 2
        best_score = -1
        scores = []

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            n_labels = len(set(labels))
            n_samples = len(embeddings)

            try:
                n_labels = len(set(labels))
                n_samples = len(embeddings)

                if 2 <= n_labels <= n_samples - 1:
                    score = silhouette_score(embeddings, labels)
                else:
                    raise ValueError(f"Invalid silhouette params: n_labels={n_labels}, n_samples={n_samples}")
            except Exception as e:
                print(f"⚠️ Skipping silhouette score due to error: {e}")
                score = -1


            scores.append(score)
            if score > best_score:
                best_score = score
                best_k = k

        num_clusters = best_k

        if progress_callback:
            progress_callback.progress(0.5)

        if status_callback:
            status_callback.text(f"✅ Optimal number of clusters selected: {num_clusters}")

        # Optional plot
        plt.figure(figsize=(8, 5))
        plt.plot(range(2, max_k + 1), scores, marker='o')
        plt.title("Optimal Number of Clusters (Silhouette Score)")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === Final KMeans clustering ===
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)

    if progress_callback:
        progress_callback.progress(1.0)
    if status_callback:
        status_callback.text("✅ Clustering complete.")

    return pd.DataFrame({
        "Keyword": keywords,
        "Cluster": [f"Cluster {label + 1}" for label in clusters]
    })
