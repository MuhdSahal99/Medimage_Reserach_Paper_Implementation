"""Tests for the MedImageInsight model wrapper."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMedImageInsightModel:
    """Test suite for the MedImageInsightModel class."""

    def test_device_resolution_auto_cpu(self):
        """Test that 'auto' resolves to a valid device."""
        from medimageinsight.model import MedImageInsightModel

        model = MedImageInsightModel(device="auto")
        assert model.device.type in ("cpu", "cuda")

    def test_device_resolution_explicit_cpu(self):
        """Test explicit CPU device selection."""
        from medimageinsight.model import MedImageInsightModel

        model = MedImageInsightModel(device="cpu")
        assert model.device.type == "cpu"

    def test_repr_before_load(self):
        """Test string representation before loading."""
        from medimageinsight.model import MedImageInsightModel

        model = MedImageInsightModel()
        repr_str = repr(model)
        assert "not loaded" in repr_str
        assert "MedImageInsightModel" in repr_str

    def test_compute_similarity_shapes(self):
        """Test similarity computation with correct shapes."""
        from medimageinsight.model import MedImageInsightModel

        model = MedImageInsightModel()

        # Create mock embeddings
        img_emb = np.random.randn(5, 768).astype(np.float32)
        img_emb /= np.linalg.norm(img_emb, axis=-1, keepdims=True)

        txt_emb = np.random.randn(3, 768).astype(np.float32)
        txt_emb /= np.linalg.norm(txt_emb, axis=-1, keepdims=True)

        similarity = model.compute_similarity(img_emb, txt_emb)

        assert similarity.shape == (5, 3)
        assert np.all(similarity >= -1.01) and np.all(similarity <= 1.01)

    def test_similarity_self_is_one(self):
        """Test that similarity of a vector with itself is ~1.0."""
        from medimageinsight.model import MedImageInsightModel

        model = MedImageInsightModel()

        emb = np.random.randn(1, 768).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=-1, keepdims=True)

        similarity = model.compute_similarity(emb, emb)
        np.testing.assert_almost_equal(similarity[0, 0], 1.0, decimal=5)
