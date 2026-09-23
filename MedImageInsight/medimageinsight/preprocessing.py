"""
Medical Image Preprocessing Pipeline.

Supports loading and preprocessing medical images from various formats
including standard image files (PNG, JPEG) and DICOM (.dcm) files.

The preprocessing follows the MedImageInsight paper's approach:
    - Images resized to model input size (default 480x480)
    - Normalized using ImageNet statistics
    - Converted to RGB 3-channel tensors
"""

import logging
from typing import Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# Default image size used in MedImageInsight
DEFAULT_IMAGE_SIZE = 480

# ImageNet normalization (used as base since model inherits from Florence/DaViT)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MedicalImagePreprocessor:
    """
    Preprocessing pipeline for medical images.

    Handles image loading from various formats, resizing, normalization,
    and conversion to model-compatible tensors.

    Args:
        image_size: Target size for the model input (square).
        mean: Normalization mean (per channel).
        std: Normalization std (per channel).

    Example:
        >>> preprocessor = MedicalImagePreprocessor()
        >>> image = preprocessor.load("patient_scan.dcm")
        >>> tensor = preprocessor.preprocess(image)
        >>> print(tensor.shape)  # torch.Size([3, 480, 480])
    """

    def __init__(
        self,
        image_size: int = DEFAULT_IMAGE_SIZE,
        mean: Tuple[float, ...] = IMAGENET_MEAN,
        std: Tuple[float, ...] = IMAGENET_STD,
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std

        # Standard transform pipeline
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        # Augmentation transform for training
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomErasing(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def load(self, path: Union[str, Path]) -> Image.Image:
        """
        Load a medical image from file.

        Supports PNG, JPEG, BMP, TIFF, and DICOM (.dcm) formats.

        Args:
            path: Path to the image file.

        Returns:
            PIL Image in RGB mode.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        suffix = path.suffix.lower()

        if suffix == ".dcm":
            return self._load_dicom(path)
        elif suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
            return self._load_standard(path)
        else:
            # Try loading as standard image
            logger.warning(f"Unknown format '{suffix}', attempting standard load.")
            return self._load_standard(path)

    def _load_standard(self, path: Path) -> Image.Image:
        """Load a standard image file and convert to RGB."""
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _load_dicom(self, path: Path) -> Image.Image:
        """
        Load a DICOM file and convert to RGB PIL Image.

        Handles windowing for CT images and proper bit-depth conversion.
        """
        try:
            import pydicom
        except ImportError:
            raise ImportError(
                "pydicom is required for DICOM file support. "
                "Install it with: pip install pydicom"
            )

        ds = pydicom.dcmread(str(path))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Apply DICOM rescale if available (for CT Hounsfield Units)
        slope = getattr(ds, "RescaleSlope", 1.0)
        intercept = getattr(ds, "RescaleIntercept", 0.0)
        pixel_array = pixel_array * float(slope) + float(intercept)

        # Apply windowing if available
        window_center = getattr(ds, "WindowCenter", None)
        window_width = getattr(ds, "WindowWidth", None)

        if window_center is not None and window_width is not None:
            # Handle multi-valued window settings
            if isinstance(window_center, pydicom.multival.MultiValue):
                window_center = float(window_center[0])
            else:
                window_center = float(window_center)
            if isinstance(window_width, pydicom.multival.MultiValue):
                window_width = float(window_width[0])
            else:
                window_width = float(window_width)

            lower = window_center - window_width / 2
            upper = window_center + window_width / 2
            pixel_array = np.clip(pixel_array, lower, upper)

        # Normalize to 0-255 range
        pmin, pmax = pixel_array.min(), pixel_array.max()
        if pmax > pmin:
            pixel_array = (pixel_array - pmin) / (pmax - pmin) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)

        pixel_array = pixel_array.astype(np.uint8)

        # Convert to RGB (grayscale → 3 channels)
        if len(pixel_array.shape) == 2:
            pixel_array = np.stack([pixel_array] * 3, axis=-1)

        return Image.fromarray(pixel_array, mode="RGB")

    def preprocess(
        self, image: Image.Image, training: bool = False
    ) -> torch.Tensor:
        """
        Preprocess a PIL Image for model input.

        Args:
            image: PIL Image in RGB mode.
            training: If True, apply data augmentation.

        Returns:
            Tensor of shape (3, image_size, image_size).
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        transform = self.train_transform if training else self.transform
        return transform(image)

    def preprocess_batch(
        self,
        images: list,
        training: bool = False,
    ) -> torch.Tensor:
        """
        Preprocess a batch of images.

        Args:
            images: List of PIL Images.
            training: If True, apply data augmentation.

        Returns:
            Tensor of shape (N, 3, image_size, image_size).
        """
        tensors = [self.preprocess(img, training=training) for img in images]
        return torch.stack(tensors)

    def aggregate_3d_volume(
        self,
        slice_embeddings: np.ndarray,
        method: str = "median",
    ) -> np.ndarray:
        """
        Aggregate 2D slice embeddings into a single 3D volume embedding.

        As described in the paper (Section 3.2.5), median aggregation
        was selected after comparing median, mean, std, and max pooling.

        Args:
            slice_embeddings: (num_slices, embedding_dim) array.
            method: Aggregation method ('median', 'mean', 'max').

        Returns:
            (1, embedding_dim) aggregated embedding.
        """
        if method == "median":
            return np.median(slice_embeddings, axis=0, keepdims=True)
        elif method == "mean":
            return np.mean(slice_embeddings, axis=0, keepdims=True)
        elif method == "max":
            return np.max(slice_embeddings, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def __repr__(self) -> str:
        return (
            f"MedicalImagePreprocessor(image_size={self.image_size}, "
            f"mean={self.mean}, std={self.std})"
        )
