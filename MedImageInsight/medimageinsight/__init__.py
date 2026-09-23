"""
MedImageInsight: An Open-Source Embedding Model for General Domain Medical Imaging

Implementation based on the paper by Codella et al. (Microsoft, 2024).
Paper: https://arxiv.org/abs/2410.06542

This package provides:
    - Zero-shot medical image classification via text prompts
    - Image-image search (KNN) for transparent, evidence-based classification
    - Embedding extraction for downstream tasks
    - ROC curve generation for regulatory compliance
    - DICOM and standard image format support
"""

__version__ = "1.0.0"
__author__ = "Mubeena"
__paper__ = "https://arxiv.org/abs/2410.06542"

from medimageinsight.model import MedImageInsightModel
from medimageinsight.classifier import ZeroShotClassifier
from medimageinsight.knn_search import KNNClassifier
from medimageinsight.embeddings import EmbeddingExtractor
from medimageinsight.preprocessing import MedicalImagePreprocessor

__all__ = [
    "MedImageInsightModel",
    "ZeroShotClassifier",
    "KNNClassifier",
    "EmbeddingExtractor",
    "MedicalImagePreprocessor",
]
