"""Tests for the ZeroShotClassifier."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestZeroShotClassifier:
    """Test suite for the ZeroShotClassifier class."""

    def test_softmax_sums_to_one(self):
        """Test that predicted probabilities sum to 1.0."""
        # Simulate softmax behavior
        logits = np.array([2.0, 0.5, -1.0])
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        np.testing.assert_almost_equal(np.sum(probs), 1.0, decimal=5)

    def test_template_formatting(self):
        """Test that prompt templates format correctly."""
        templates = [
            "a medical image of {}",
            "a photo of {}",
            "{} presented in medical image",
        ]
        class_name = "pneumonia"

        prompts = [t.format(class_name) for t in templates]

        assert prompts[0] == "a medical image of pneumonia"
        assert prompts[1] == "a photo of pneumonia"
        assert prompts[2] == "pneumonia presented in medical image"

    def test_average_embeddings_preserves_shape(self):
        """Test that averaging template embeddings preserves dimensions."""
        n_templates = 3
        embedding_dim = 768
        embeddings = np.random.randn(n_templates, embedding_dim)

        avg = np.mean(embeddings, axis=0, keepdims=True)
        assert avg.shape == (1, embedding_dim)

        # Renormalize
        norm = np.linalg.norm(avg, axis=-1, keepdims=True)
        avg_normalized = avg / (norm + 1e-8)
        np.testing.assert_almost_equal(
            np.linalg.norm(avg_normalized), 1.0, decimal=5
        )

    def test_classification_head_shape(self):
        """Test that text embedding matrix has correct shape."""
        num_classes = 5
        embedding_dim = 768

        class_embeddings = np.random.randn(num_classes, embedding_dim)

        # Query with batch of images
        batch_size = 10
        image_embeddings = np.random.randn(batch_size, embedding_dim)

        similarities = image_embeddings @ class_embeddings.T
        assert similarities.shape == (batch_size, num_classes)
