"""
MedImageInsight Model Loader and Wrapper.

Loads the MedImageInsight foundation model from HuggingFace and provides
a unified interface for image and text encoding.

Architecture (from paper):
    - Image Encoder: DaViT (Dual Attention Vision Transformer), 360M params
    - Text Encoder: Transformer-based, 252M params
    - Objective: UniCL (Unified Contrastive Learning)
    
Reference:
    Codella et al., "MedImageInsight: An Open-Source Embedding Model
    for General Domain Medical Imaging", arXiv:2410.06542, 2024.
"""

import os
import json
import logging
from typing import List, Optional, Union
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

# Default model identifier on HuggingFace
DEFAULT_MODEL_ID = "lion-ai/MedImageInsights"


class MedImageInsightModel:
    """
    Wrapper for the MedImageInsight foundation model.

    Provides methods to encode medical images and text into a shared
    embedding space for classification, search, and retrieval tasks.

    Args:
        model_id: HuggingFace model identifier or local path.
        device: Device to load model on ('cuda', 'cpu', or 'auto').
        dtype: Model precision (torch.float32 or torch.float16).
        cache_dir: Directory to cache downloaded model files.

    Example:
        >>> model = MedImageInsightModel()
        >>> image = Image.open("chest_xray.png")
        >>> embedding = model.encode_image(image)
        >>> print(embedding.shape)  # (1, embedding_dim)
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
    ):
        self.model_id = model_id
        self.dtype = dtype
        self.device = self._resolve_device(device)
        self.cache_dir = cache_dir

        self._model = None
        self._processor = None
        self._tokenizer = None
        self._config = None
        self._loaded = False

        logger.info(f"MedImageInsight initialized (device={self.device}, dtype={dtype})")

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def load(self) -> "MedImageInsightModel":
        """
        Download (if needed) and load the model into memory.

        Returns:
            self for method chaining.
        """
        if self._loaded:
            logger.info("Model already loaded, skipping.")
            return self

        logger.info(f"Loading model from '{self.model_id}'...")

        # Download model files from HuggingFace
        if os.path.isdir(self.model_id):
            model_dir = self.model_id
        else:
            model_dir = snapshot_download(
                repo_id=self.model_id,
                cache_dir=self.cache_dir,
            )

        logger.info(f"Model files located at: {model_dir}")

        # Load model components
        self._load_model_components(model_dir)
        self._loaded = True

        logger.info(
            f"Model loaded successfully on {self.device} "
            f"(image_encoder: {self._count_params('image')}M params, "
            f"text_encoder: {self._count_params('text')}M params)"
        )
        return self

    def _load_model_components(self, model_dir: str):
        """Load model, processor, and tokenizer from directory."""
        # Try to load using the simplified community structure
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer

            self._model = AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
                torch_dtype=self.dtype,
            ).to(self.device)
            self._model.eval()

            # Try loading processor/tokenizer
            try:
                self._processor = AutoProcessor.from_pretrained(
                    model_dir, trust_remote_code=True
                )
            except Exception:
                logger.warning("Could not load AutoProcessor, using manual preprocessing.")
                self._processor = None

            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_dir, trust_remote_code=True
                )
            except Exception:
                logger.warning("Could not load AutoTokenizer, using manual tokenization.")
                self._tokenizer = None

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(
                f"Could not load MedImageInsight from '{model_dir}'. "
                f"Ensure you have the correct model files. Error: {e}"
            )

        # Load config if available
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self._config = json.load(f)

    def _count_params(self, component: str) -> str:
        """Count parameters of a model component (returns string like '360')."""
        if self._model is None:
            return "?"
        try:
            total = sum(p.numel() for p in self._model.parameters())
            return f"{total / 1e6:.0f}"
        except Exception:
            return "?"

    def _ensure_loaded(self):
        """Ensure the model is loaded before inference."""
        if not self._loaded:
            self.load()

    @torch.no_grad()
    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image], torch.Tensor],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode one or more images into embedding vectors.

        Args:
            images: PIL Image(s) or a batch tensor of shape (B, C, H, W).
            normalize: Whether to L2-normalize the embeddings.

        Returns:
            numpy array of shape (N, embedding_dim).
        """
        self._ensure_loaded()

        if isinstance(images, Image.Image):
            images = [images]

        if isinstance(images, list):
            # Process through the model's processor
            if self._processor is not None:
                inputs = self._processor(images=images, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            else:
                # Manual preprocessing fallback
                from medimageinsight.preprocessing import MedicalImagePreprocessor
                preprocessor = MedicalImagePreprocessor()
                tensors = [preprocessor.preprocess(img) for img in images]
                inputs = {"pixel_values": torch.stack(tensors).to(self.device)}
        else:
            # Already a tensor
            inputs = {"pixel_values": images.to(self.device)}

        # Forward pass through image encoder
        outputs = self._model.get_image_features(**inputs)

        if isinstance(outputs, torch.Tensor):
            embeddings = outputs
        else:
            embeddings = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Encode one or more text strings into embedding vectors.

        Args:
            texts: Text string(s) to encode.
            normalize: Whether to L2-normalize the embeddings.

        Returns:
            numpy array of shape (N, embedding_dim).
        """
        self._ensure_loaded()

        if isinstance(texts, str):
            texts = [texts]

        if self._tokenizer is not None:
            inputs = self._tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        elif self._processor is not None:
            inputs = self._processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            raise RuntimeError("No tokenizer or processor available for text encoding.")

        outputs = self._model.get_text_features(**inputs)

        if isinstance(outputs, torch.Tensor):
            embeddings = outputs
        else:
            embeddings = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().numpy()

    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between image and text embeddings.

        Args:
            image_embeddings: (N, D) array of image embeddings.
            text_embeddings: (M, D) array of text embeddings.

        Returns:
            (N, M) similarity matrix.
        """
        # Embeddings are already L2-normalized, so dot product = cosine similarity
        return image_embeddings @ text_embeddings.T

    @property
    def embedding_dim(self) -> int:
        """Get the dimensionality of the embedding space."""
        self._ensure_loaded()
        if self._config and "hidden_size" in self._config:
            return self._config["hidden_size"]
        # Probe by encoding a dummy input
        try:
            dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
            emb = self.encode_image(dummy)
            return emb.shape[-1]
        except Exception:
            return -1

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"MedImageInsightModel(model_id='{self.model_id}', "
            f"device={self.device}, status={status})"
        )
