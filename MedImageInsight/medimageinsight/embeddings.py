"""
Embedding Extraction Utilities.

Provides batch embedding extraction from folders of medical images
with options to save/load embeddings to disk for reuse.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from medimageinsight.model import MedImageInsightModel
from medimageinsight.preprocessing import MedicalImagePreprocessor

logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".dcm"}


class EmbeddingExtractor:
    """
    Batch embedding extraction from medical image collections.

    Args:
        model: A loaded MedImageInsightModel instance.
        preprocessor: Optional MedicalImagePreprocessor instance.
        batch_size: Number of images to process at once.

    Example:
        >>> model = MedImageInsightModel().load()
        >>> extractor = EmbeddingExtractor(model)
        >>> embeddings, paths = extractor.extract_from_folder("./xray_images/")
        >>> extractor.save_embeddings(embeddings, paths, "xray_embeddings.npz")
    """

    def __init__(
        self,
        model: MedImageInsightModel,
        preprocessor: Optional[MedicalImagePreprocessor] = None,
        batch_size: int = 32,
    ):
        self.model = model
        self.preprocessor = preprocessor or MedicalImagePreprocessor()
        self.batch_size = batch_size

    def extract_from_folder(
        self,
        folder_path: Union[str, Path],
        recursive: bool = True,
        extensions: Optional[set] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract embeddings for all images in a folder.

        Args:
            folder_path: Path to the image folder.
            recursive: Whether to search subdirectories.
            extensions: Set of valid file extensions (default: IMAGE_EXTENSIONS).

        Returns:
            Tuple of (embeddings array, list of file paths).
        """
        folder = Path(folder_path)
        extensions = extensions or IMAGE_EXTENSIONS

        # Collect image paths
        if recursive:
            image_paths = [
                p for p in folder.rglob("*")
                if p.suffix.lower() in extensions and p.is_file()
            ]
        else:
            image_paths = [
                p for p in folder.iterdir()
                if p.suffix.lower() in extensions and p.is_file()
            ]

        image_paths.sort()
        logger.info(f"Found {len(image_paths)} images in '{folder}'")

        if not image_paths:
            return np.array([]), []

        return self.extract_from_paths([str(p) for p in image_paths])

    def extract_from_paths(
        self,
        image_paths: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract embeddings for a list of image file paths.

        Args:
            image_paths: List of absolute or relative image paths.

        Returns:
            Tuple of (embeddings array, list of valid file paths).
        """
        all_embeddings = []
        valid_paths = []

        for i in tqdm(range(0, len(image_paths), self.batch_size),
                      desc="Extracting embeddings"):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_images = []

            for path in batch_paths:
                try:
                    image = self.preprocessor.load(path)
                    batch_images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Failed to load '{path}': {e}")

            if batch_images:
                embeddings = self.model.encode_image(batch_images, normalize=True)
                all_embeddings.append(embeddings)

        if all_embeddings:
            return np.concatenate(all_embeddings, axis=0), valid_paths
        return np.array([]), []

    def extract_from_images(
        self,
        images: List[Image.Image],
    ) -> np.ndarray:
        """
        Extract embeddings from a list of PIL Images.

        Args:
            images: List of PIL Images.

        Returns:
            numpy array of shape (N, embedding_dim).
        """
        all_embeddings = []

        for i in tqdm(range(0, len(images), self.batch_size),
                      desc="Extracting embeddings"):
            batch = images[i : i + self.batch_size]
            embeddings = self.model.encode_image(batch, normalize=True)
            all_embeddings.append(embeddings)

        return np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])

    @staticmethod
    def save_embeddings(
        embeddings: np.ndarray,
        paths: List[str],
        output_path: Union[str, Path],
        labels: Optional[List[str]] = None,
    ):
        """
        Save embeddings and metadata to a compressed .npz file.

        Args:
            embeddings: (N, D) array of embeddings.
            paths: List of source file paths.
            output_path: Path to save the .npz file.
            labels: Optional list of labels.
        """
        data = {
            "embeddings": embeddings,
            "paths": np.array(paths, dtype=object),
        }
        if labels is not None:
            data["labels"] = np.array(labels, dtype=object)

        np.savez_compressed(str(output_path), **data)
        logger.info(
            f"Saved {embeddings.shape[0]} embeddings to '{output_path}' "
            f"({Path(output_path).stat().st_size / 1024:.1f} KB)"
        )

    @staticmethod
    def load_embeddings(
        path: Union[str, Path],
    ) -> Dict[str, np.ndarray]:
        """
        Load embeddings from a .npz file.

        Args:
            path: Path to the .npz file.

        Returns:
            Dictionary with keys 'embeddings', 'paths', and optionally 'labels'.
        """
        data = np.load(str(path), allow_pickle=True)
        result = {
            "embeddings": data["embeddings"],
            "paths": data["paths"].tolist(),
        }
        if "labels" in data:
            result["labels"] = data["labels"].tolist()

        logger.info(f"Loaded {result['embeddings'].shape[0]} embeddings from '{path}'")
        return result

    def __repr__(self) -> str:
        return f"EmbeddingExtractor(batch_size={self.batch_size})"
