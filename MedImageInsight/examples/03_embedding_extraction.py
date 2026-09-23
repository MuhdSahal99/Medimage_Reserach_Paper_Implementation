"""
Example 03: Embedding Extraction

Extract and save medical image embeddings for downstream tasks
such as clustering, visualization, or building search databases.

Usage:
    python examples/03_embedding_extraction.py --input_dir ./images/ --output embeddings.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from medimageinsight import MedImageInsightModel
from medimageinsight.embeddings import EmbeddingExtractor


def main():
    parser = argparse.ArgumentParser(
        description="Extract medical image embeddings with MedImageInsight"
    )
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing medical images")
    parser.add_argument("--output", type=str, default="embeddings.npz",
                        help="Output path for saved embeddings (.npz)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MedImageInsight — Embedding Extraction")
    print(f"{'='*60}\n")

    # --- Load model ---
    print("[1/3] Loading model...")
    model = MedImageInsightModel(device="auto")
    model.load()

    # --- Extract embeddings ---
    print("[2/3] Extracting embeddings...")
    extractor = EmbeddingExtractor(model, batch_size=args.batch_size)

    if args.input_dir:
        embeddings, paths = extractor.extract_from_folder(args.input_dir)
    else:
        # Demo mode: create synthetic images
        from PIL import Image

        print("  (No input directory provided — running in demo mode)")
        demo_images = [
            Image.new("RGB", (224, 224), color=(c, c, c))
            for c in range(50, 250, 20)
        ]
        embeddings = extractor.extract_from_images(demo_images)
        paths = [f"demo_image_{i}.png" for i in range(len(demo_images))]

    # --- Save ---
    print(f"\n[3/3] Saving embeddings...")
    extractor.save_embeddings(embeddings, paths, args.output)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Summary:")
    print(f"    Images processed: {embeddings.shape[0]}")
    print(f"    Embedding dim:    {embeddings.shape[1]}")
    print(f"    Output file:      {args.output}")
    print(f"    File size:        {Path(args.output).stat().st_size / 1024:.1f} KB")
    print(f"\n  Usage:")
    print(f'    data = np.load("{args.output}", allow_pickle=True)')
    print(f'    embeddings = data["embeddings"]  # shape: {embeddings.shape}')
    print(f'    paths = data["paths"]')
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
