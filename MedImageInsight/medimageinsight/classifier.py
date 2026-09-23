"""
Zero-Shot Medical Image Classifier.

Implements text-prompt based classification using MedImageInsight's
two-tower architecture. The text encoder generates a classification head
from text descriptions of each class, and images are classified by
computing cosine similarity against these text embeddings.

This mirrors the "Image-Text Search" mode described in the paper (Section 3.2.1).
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from medimageinsight.model import MedImageInsightModel

logger = logging.getLogger(__name__)

# Default prompt templates from the paper (Supplementary Table 10 of BiomedCLIP,
# adapted for MedImageInsight usage)
DEFAULT_TEMPLATES = [
    "a medical image of {}",
    "a photo of {}",
    "{} presented in medical image",
]


class ZeroShotClassifier:
    """
    Zero-shot medical image classifier using text-prompt classification heads.

    The classifier encodes text descriptions of each class using the text encoder,
    then classifies images by computing cosine similarity between image embeddings
    and these text embeddings, producing confidence scores suitable for ROC curves.

    Args:
        model: A loaded MedImageInsightModel instance.
        class_names: List of class label strings.
        templates: Prompt templates with '{}' placeholder for class name.

    Example:
        >>> model = MedImageInsightModel().load()
        >>> classifier = ZeroShotClassifier(
        ...     model=model,
        ...     class_names=["normal lung", "pneumonia"],
        ... )
        >>> image = Image.open("chest_xray.png")
        >>> predictions = classifier.predict(image)
        >>> print(predictions)
        {'normal lung': 0.72, 'pneumonia': 0.28}
    """

    def __init__(
        self,
        model: MedImageInsightModel,
        class_names: List[str],
        templates: Optional[List[str]] = None,
    ):
        self.model = model
        self.class_names = class_names
        self.templates = templates or DEFAULT_TEMPLATES
        self._text_embeddings = None

        # Pre-compute text embeddings for all classes
        self._build_classifier_head()

    def _build_classifier_head(self):
        """
        Build the classification head by encoding all class descriptions.

        For each class, we generate text prompts using all templates and
        average the resulting embeddings to create a robust class representation.
        """
        logger.info(f"Building classifier head for {len(self.class_names)} classes...")

        all_class_embeddings = []

        for class_name in self.class_names:
            # Generate prompts from templates
            prompts = [t.format(class_name) for t in self.templates]

            # Encode all prompts for this class
            embeddings = self.model.encode_text(prompts, normalize=True)

            # Average across templates
            class_embedding = np.mean(embeddings, axis=0, keepdims=True)

            # Re-normalize after averaging
            norm = np.linalg.norm(class_embedding, axis=-1, keepdims=True)
            class_embedding = class_embedding / (norm + 1e-8)

            all_class_embeddings.append(class_embedding)

        # Stack into (num_classes, embedding_dim)
        self._text_embeddings = np.concatenate(all_class_embeddings, axis=0)

        logger.info(
            f"Classifier head built: {self._text_embeddings.shape} "
            f"({len(self.class_names)} classes)"
        )

    def predict(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_scores: bool = True,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Classify image(s) using zero-shot text-prompt classification.

        Args:
            images: Single image or list of images.
            return_scores: If True, return per-class confidence scores.

        Returns:
            Dictionary mapping class names to confidence scores (softmax-normalized).
            Returns a list of dicts if multiple images are provided.
        """
        single = isinstance(images, Image.Image)
        if single:
            images = [images]

        # Encode images
        image_embeddings = self.model.encode_image(images, normalize=True)

        # Compute similarity (dot product on normalized vectors = cosine similarity)
        similarities = image_embeddings @ self._text_embeddings.T  # (N, num_classes)

        # Apply softmax to get probabilities
        # Temperature scaling (from CLIP-style models)
        temperature = 100.0
        logits = similarities * temperature
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Format results
        results = []
        for i in range(len(images)):
            scores = {
                name: float(probs[i, j])
                for j, name in enumerate(self.class_names)
            }
            results.append(scores)

        return results[0] if single else results

    def predict_top_k(
        self,
        image: Image.Image,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for a single image.

        Args:
            image: PIL Image to classify.
            k: Number of top predictions to return.

        Returns:
            List of (class_name, confidence) tuples, sorted by confidence.
        """
        scores = self.predict(image)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:k]

    def get_raw_similarities(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> np.ndarray:
        """
        Get raw cosine similarity scores (before softmax).

        Useful for custom thresholding or ROC curve generation.

        Args:
            images: Single image or list of images.

        Returns:
            numpy array of shape (N, num_classes) with raw similarity scores.
        """
        if isinstance(images, Image.Image):
            images = [images]

        image_embeddings = self.model.encode_image(images, normalize=True)
        return image_embeddings @ self._text_embeddings.T

    def __repr__(self) -> str:
        return (
            f"ZeroShotClassifier(classes={self.class_names}, "
            f"templates={len(self.templates)})"
        )
