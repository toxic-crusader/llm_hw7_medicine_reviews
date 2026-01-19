# File: scripts/cluster_topics.py

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_cluster_keywords(
    texts: List[str],
    labels: np.ndarray,
    top_k: int = 15,
    min_df: int = 5,
) -> Dict[int, List[str]]:
    """
    Extract top keywords for each cluster using TF IDF.

    The function fits a single TF IDF vectorizer on the full corpus and then,
    for each cluster, computes mean TF IDF weights across documents in that cluster.
    Top terms by mean TF IDF are returned as cluster keywords.

    Parameters
    ----------
    texts : List[str]
        Input documents.
    labels : np.ndarray
        Cluster labels aligned with texts.
    top_k : int
        Number of keywords to return per cluster.
    min_df : int
        Minimum document frequency for TF IDF.

    Returns
    -------
    Dict[int, List[str]]
        Mapping from cluster id to list of keywords.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length")

    df = pd.DataFrame(
        {
            "text": texts,
            "cluster": labels,
        }
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=min_df,
        max_df=0.9,
    )

    tfidf = vectorizer.fit_transform(df["text"].astype(str))
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    cluster_values = df["cluster"].to_numpy()
    cluster_keywords: Dict[int, List[str]] = {}

    for cluster_id in np.unique(cluster_values):
        mask = cluster_values == cluster_id
        cluster_tfidf = tfidf[mask]

        mean_tfidf = np.asarray(cluster_tfidf.mean(axis=0)).ravel()
        top_indices = np.argsort(mean_tfidf)[::-1][:top_k]

        cluster_keywords[int(cluster_id)] = feature_names[top_indices].tolist()

    return cluster_keywords
