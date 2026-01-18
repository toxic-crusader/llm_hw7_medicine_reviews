# File: scripts/embeddings.py

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class TextEmbedder:
    """
    Utility class for computing, saving and loading text embeddings.

    This class wraps a sentence-transformers model and provides
    a reproducible way to convert raw text reviews into dense
    vector representations.

    Embeddings can be cached on disk to avoid repeated expensive
    computation when the input texts do not change.
    """

    def __init__(
        self,
        model_name: str,
        device: torch.device,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize text embedder.

        Parameters
        ----------
        model_name : str
            Name of the sentence-transformers model.
        device : torch.device
            Device used for embedding computation.
        batch_size : int
            Batch size used during encoding.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name,
            device=str(device),
        )

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.

        Parameters
        ----------
        texts : List[str]
            Input texts to be embedded.

        Returns
        -------
        np.ndarray
            Embedding matrix of shape (n_texts, embedding_dim).
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return embeddings

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: Path,
    ) -> None:
        """
        Save embeddings to disk in NumPy format.

        Parameters
        ----------
        embeddings : np.ndarray
            Embedding matrix to be saved.
        path : Path
            Target file path.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    def load_embeddings(
        self,
        path: Path,
    ) -> np.ndarray:
        """
        Load embeddings from disk.

        Parameters
        ----------
        path : Path
            Path to saved embeddings file.

        Returns
        -------
        np.ndarray
            Loaded embedding matrix.
        """
        return np.load(path)
