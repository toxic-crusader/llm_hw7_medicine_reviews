# File: scripts/clustering.py

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.cluster import KMeans


class EmbeddingClusterer:
    """
    Utility class for clustering text embeddings.
    """

    def __init__(
        self,
        n_clusters: int,
        random_state: int = 42,
    ) -> None:
        """
        Initialize clusterer.

        Parameters
        ----------
        n_clusters : int
            Number of clusters.
        random_state : int
            Random seed for reproducibility.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.model: Optional[KMeans] = None

    def fit_predict(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Cluster embeddings and return cluster labels.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix.

        Returns
        -------
        np.ndarray
            Cluster labels for each embedding.
        """
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        labels = self.model.fit_predict(embeddings)
        return labels
