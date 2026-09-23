"""
Example 02: Image-Image Search (KNN Classification)

Demonstrates the inherently explainable classification mode from
MedImageInsight — finding the most similar reference images and
using weighted voting for transparent, evidence-based predictions.

Usage:
    python examples/02_image_image_search.py --database_dir ./train_images/ --query ./test.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from medimageinsight import MedImageInsightModel, KNNClassifier
from medimageinsight.embeddings import EmbeddingExtractor
from medimageinsight.utils import plot_search_results


def create_demo_dataset():
    """Create a small synthetic dataset for demonstration."""
    import numpy as np

    print("  Creating synthetic demo dataset...")
    images = []
    labels = []

    # Generate simple colored images as placeholders
    class_colors = {
        "normal": (200, 200, 200),     # Light gray
        "abnormal_A": (180, 100, 100), # Reddish
        "abnormal_B": (100, 100, 180), # Bluish
    }

    for class_name, color in class_colors.items():
        for i in range(10):
            # Add slight variation
            r = min(255, max(0, color[0] + np.random.randint(-20, 20)))
            g = min(255, max(0, color[1] + np.random.randint(-20, 20)))
            b = min(255, max(0, color[2] + np.random.randint(-20, 20)))
            img = Image.new("RGB", (224, 224), color=(r, g, b))
            images.append(img)
            labels.append(class_name)

    return images, labels


def main():
    parser = argparse.ArgumentParser(
        description="KNN image-image search with MedImageInsight"
    )
    parser.add_argument("--database_dir", type=str, default=None,
                        help="Path to reference image database folder")
    parser.add_argument("--query", type=str, default=None,
                        help="Path to query image")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of nearest neighbors (default: 20)")
    parser.add_argument("--save_db", type=str, default=None,
                        help="Path to save the embedding database")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MedImageInsight — Image-Image Search (KNN)")
    print(f"  k = {args.k}")
    print(f"{'='*60}\n")

    # --- Load model ---
    print("[1/4] Loading MedImageInsight model...")
    model = MedImageInsightModel(device="auto")
    model.load()

    # --- Build reference database ---
    print("[2/4] Building reference database...")

    if args.database_dir:
        extractor = EmbeddingExtractor(model)
        embeddings, paths = extractor.extract_from_folder(args.database_dir)
        # Labels from subdirectory names
        labels = [Path(p).parent.name for p in paths]
    else:
        # Use synthetic demo data
        images, labels = create_demo_dataset()
        paths = None
        embeddings = None  # Let KNN extract them

    # --- Build KNN classifier ---
    knn = KNNClassifier(model=model, k=args.k)

    if embeddings is not None:
        knn.build_database(embeddings=embeddings, labels=labels, image_paths=paths)
    else:
        knn.build_database(images=images, labels=labels)

    print(f"      Database built: {knn}")

    # Save database if requested
    if args.save_db:
        knn.save_database(args.save_db)
        print(f"      Database saved to: {args.save_db}")

    # --- Query ---
    print("[3/4] Running search query...")

    if args.query:
        query_image = Image.open(args.query).convert("RGB")
    else:
        query_image = Image.new("RGB", (224, 224), color=(190, 190, 190))
        print("      (Using dummy query image)")

    # Predict with evidence
    prediction, evidence = knn.predict_with_evidence(query_image, k=min(args.k, 10))

    # --- Display results ---
    print(f"\n[4/4] Results:")
    print("-" * 50)

    print("\n  Prediction (weighted voting):")
    for class_name, score in sorted(prediction.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 40)
        print(f"    {class_name:30s} {score:.4f}  {bar}")

    print(f"\n  Top-{min(5, len(evidence))} Evidence (nearest neighbors):")
    for entry in evidence[:5]:
        print(
            f"    #{entry['rank']:2d}  "
            f"Label: {entry['label']:20s}  "
            f"Similarity: {entry['similarity']:.4f}  "
            f"{'(path: ' + entry['path'] + ')' if entry.get('path') else ''}"
        )

    print(f"\n{'='*60}")
    print("  The evidence above shows WHY the model made this prediction.")
    print("  This is the 'inherent transparency' feature of MedImageInsight.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
