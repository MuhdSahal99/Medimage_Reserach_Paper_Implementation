"""
K-Nearest Neighbor (KNN) Image-Image Search Classifier.

Implements the "Image-Image Search" mode from MedImageInsight (Section 3.2.1),
which provides inherently transparent and explainable classification by
returning the most similar reference images as evidence for decisions.

Key parameters from the paper:
    - Classification: k=20, weighted voting (dot product)
    - Regression (e.g., bone age): k=100, weighted voting
"""

import logging
import pickle
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

from medimageinsight.model import MedImageInsightModel

logger = logging.getLogger(__name__)


class KNNClassifier:
    """
    KNN-based image classifier using MedImageInsight embeddings.

    Builds a reference database of image embeddings and classifies new
    images by finding the k-nearest neighbors and performing weighted voting.
    This is the "inherently explainable" classification mode described in the paper.

    Args:
        model: A loaded MedImageInsightModel instance.
        k: Number of nearest neighbors for classification (default: 20).
        metric: Distance metric for KNN ('cosine' or 'euclidean').

    Example:
        >>> model = MedImageInsightModel().load()
        >>> knn = KNNClassifier(model=model, k=20)
        >>> knn.build_database(images=train_images, labels=train_labels)
        >>> prediction, evidence = knn.predict_with_evidence(test_image)
    """

    def __init__(
        self,
        model: MedImageInsightModel,
        k: int = 20,
        metric: str = "cosine",
    ):
        self.model = model
        self.k = k
        self.metric = metric

        self._database_embeddings: Optional[np.ndarray] = None
        self._database_labels: Optional[List[str]] = None
        self._database_paths: Optional[List[str]] = None
        self._database_metadata: Optional[List[dict]] = None
        self._nn_index: Optional[NearestNeighbors] = None

    def build_database(
        self,
        images: Optional[List[Image.Image]] = None,
        labels: Optional[List[str]] = None,
        embeddings: Optional[np.ndarray] = None,
        image_paths: Optional[List[str]] = None,
        metadata: Optional[List[dict]] = None,
        batch_size: int = 32,
    ) -> "KNNClassifier":
        """
        Build the reference database for KNN search.

        Provide either (images + labels) for automatic embedding extraction,
        or (embeddings + labels) if embeddings are pre-computed.

        Args:
            images: List of PIL Images for the reference database.
            labels: List of class labels for each image.
            embeddings: Pre-computed embeddings of shape (N, D).
            image_paths: Optional file paths for evidence display.
            metadata: Optional metadata dicts for each image.
            batch_size: Batch size for embedding extraction.

        Returns:
            self for method chaining.
        """
        if embeddings is not None:
            self._database_embeddings = embeddings
        elif images is not None:
            logger.info(f"Extracting embeddings for {len(images)} images...")
            all_embeddings = []
            for i in range(0, len(images), batch_size):
                batch = images[i : i + batch_size]
                batch_emb = self.model.encode_image(batch, normalize=True)
                all_embeddings.append(batch_emb)
            self._database_embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            raise ValueError("Provide either 'images' or 'embeddings'.")

        self._database_labels = labels
        self._database_paths = image_paths
        self._database_metadata = metadata

        # Build the nearest neighbor index
        self._nn_index = NearestNeighbors(
            n_neighbors=min(self.k, len(self._database_embeddings)),
            metric=self.metric,
            algorithm="auto",
        )
        self._nn_index.fit(self._database_embeddings)

        logger.info(
            f"Database built: {self._database_embeddings.shape[0]} images, "
            f"{len(set(labels)) if labels else '?'} classes"
        )
        return self

    def predict(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Classify image(s) using KNN weighted voting.

        Args:
            images: Single image or list of images.

        Returns:
            Dictionary mapping class names to confidence scores.
        """
        self._ensure_database()
        single = isinstance(images, Image.Image)
        if single:
            images = [images]

        query_embeddings = self.model.encode_image(images, normalize=True)
        distances, indices = self._nn_index.kneighbors(query_embeddings)

        results = []
        for i in range(len(images)):
            # Weighted voting using similarity (1 - cosine_distance for cosine metric)
            if self.metric == "cosine":
                similarities = 1.0 - distances[i]
            else:
                similarities = 1.0 / (1.0 + distances[i])

            # Aggregate votes per class
            class_scores: Dict[str, float] = {}
            for j, idx in enumerate(indices[i]):
                label = self._database_labels[idx]
                weight = float(similarities[j])
                class_scores[label] = class_scores.get(label, 0.0) + weight

            # Normalize to probabilities
            total = sum(class_scores.values())
            if total > 0:
                class_scores = {k: v / total for k, v in class_scores.items()}

            results.append(class_scores)

        return results[0] if single else results

    def predict_with_evidence(
        self,
        image: Image.Image,
        k: Optional[int] = None,
    ) -> Tuple[Dict[str, float], List[dict]]:
        """
        Classify an image and return evidence (nearest neighbors).

        This is the key differentiator of MedImageInsight — providing
        transparent, evidence-based predictions through similar reference images.

        Args:
            image: PIL Image to classify.
            k: Number of neighbors to return (default: self.k).

        Returns:
            Tuple of:
                - prediction: Dict of class_name → confidence
                - evidence: List of dicts with keys 'rank', 'label',
                  'similarity', 'path', 'metadata'
        """
        self._ensure_database()
        k = k or self.k

        query_embedding = self.model.encode_image(image, normalize=True)
        distances, indices = self._nn_index.kneighbors(
            query_embedding, n_neighbors=min(k, len(self._database_embeddings))
        )

        # Build evidence list
        evidence = []
        class_scores: Dict[str, float] = {}

        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            similarity = 1.0 - dist if self.metric == "cosine" else 1.0 / (1.0 + dist)
            label = self._database_labels[idx]

            entry = {
                "rank": rank + 1,
                "label": label,
                "similarity": float(similarity),
                "database_index": int(idx),
                "path": self._database_paths[idx] if self._database_paths else None,
                "metadata": (
                    self._database_metadata[idx] if self._database_metadata else None
                ),
            }
            evidence.append(entry)

            class_scores[label] = class_scores.get(label, 0.0) + float(similarity)

        # Normalize predictions
        total = sum(class_scores.values())
        if total > 0:
            class_scores = {k: v / total for k, v in class_scores.items()}

        return class_scores, evidence

    def predict_regression(
        self,
        image: Image.Image,
        values: Optional[List[float]] = None,
        k: int = 100,
    ) -> float:
        """
        KNN regression (e.g., bone age estimation).

        Uses k=100 with weighted averaging as described in the paper.

        Args:
            image: PIL Image.
            values: Numeric values for each database entry (e.g., age in months).
                    If None, attempts to parse labels as floats.
            k: Number of neighbors (default: 100 per paper).

        Returns:
            Weighted average of nearest neighbor values.
        """
        self._ensure_database()

        if values is None:
            try:
                values = [float(l) for l in self._database_labels]
            except (ValueError, TypeError):
                raise ValueError(
                    "Labels cannot be parsed as floats. "
                    "Provide explicit numeric 'values'."
                )

        query_embedding = self.model.encode_image(image, normalize=True)
        distances, indices = self._nn_index.kneighbors(
            query_embedding, n_neighbors=min(k, len(self._database_embeddings))
        )

        # Weighted average
        if self.metric == "cosine":
            weights = 1.0 - distances[0]
        else:
            weights = 1.0 / (1.0 + distances[0])

        neighbor_values = np.array([values[idx] for idx in indices[0]])
        prediction = np.average(neighbor_values, weights=weights)

        return float(prediction)

    def save_database(self, path: Union[str, Path]):
        """Save the database to disk for later reuse."""
        self._ensure_database()
        data = {
            "embeddings": self._database_embeddings,
            "labels": self._database_labels,
            "paths": self._database_paths,
            "metadata": self._database_metadata,
            "k": self.k,
            "metric": self.metric,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Database saved to {path}")

    def load_database(self, path: Union[str, Path]) -> "KNNClassifier":
        """Load a previously saved database."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.build_database(
            embeddings=data["embeddings"],
            labels=data["labels"],
            image_paths=data.get("paths"),
            metadata=data.get("metadata"),
        )
        logger.info(f"Database loaded from {path}")
        return self

    def _ensure_database(self):
        """Ensure the database is built before querying."""
        if self._database_embeddings is None or self._nn_index is None:
            raise RuntimeError(
                "Database not built. Call build_database() first."
            )

    def __repr__(self) -> str:
        n = self._database_embeddings.shape[0] if self._database_embeddings is not None else 0
        return f"KNNClassifier(k={self.k}, metric='{self.metric}', database_size={n})"
