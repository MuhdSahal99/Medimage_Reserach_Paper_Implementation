"""
Example 01: Zero-Shot Medical Image Classification

Demonstrates how to classify medical images using text prompts
without any task-specific training — a key capability of MedImageInsight.

Usage:
    python examples/01_zero_shot_classification.py --image path/to/image.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from medimageinsight import MedImageInsightModel, ZeroShotClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot medical image classification with MedImageInsight"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a medical image (PNG, JPEG, or DICOM)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="chest_xray",
        choices=["chest_xray", "dermatology", "oct", "ultrasound", "pathology"],
        help="Medical imaging domain for class labels",
    )
    args = parser.parse_args()

    # --- Define domain-specific class labels ---
    domain_classes = {
        "chest_xray": [
            "normal chest X-ray",
            "chest X-ray with pneumonia",
            "chest X-ray with pleural effusion",
            "chest X-ray with cardiomegaly",
            "chest X-ray with atelectasis",
        ],
        "dermatology": [
            "melanocytic nevus",
            "melanoma",
            "basal cell carcinoma",
            "actinic keratosis",
            "benign keratosis",
            "dermatofibroma",
            "vascular lesion",
            "squamous cell carcinoma",
        ],
        "oct": [
            "normal retinal OCT",
            "OCT with choroidal neovascularization",
            "OCT with diabetic macular edema",
            "OCT with drusen",
        ],
        "ultrasound": [
            "normal breast ultrasound",
            "benign breast mass on ultrasound",
            "malignant breast mass on ultrasound",
        ],
        "pathology": [
            "normal tissue histopathology",
            "metastatic tissue histopathology",
        ],
    }

    class_names = domain_classes[args.domain]
    print(f"\n{'='*60}")
    print(f"  MedImageInsight — Zero-Shot Classification")
    print(f"  Domain: {args.domain}")
    print(f"  Classes: {len(class_names)}")
    print(f"{'='*60}\n")

    # --- Load the model ---
    print("[1/3] Loading MedImageInsight model...")
    model = MedImageInsightModel(device="auto")
    model.load()
    print(f"      Model loaded: {model}")

    # --- Build classifier ---
    print("[2/3] Building zero-shot classifier...")
    classifier = ZeroShotClassifier(
        model=model,
        class_names=class_names,
    )
    print(f"      Classifier ready: {classifier}")

    # --- Classify image ---
    if args.image:
        image = Image.open(args.image).convert("RGB")
        source = args.image
    else:
        # Create a dummy test image if no image provided
        print("      (No image provided, using a dummy 224x224 gray image for demo)")
        image = Image.new("RGB", (224, 224), color=(128, 128, 128))
        source = "dummy_image"

    print(f"\n[3/3] Classifying: {source}")
    print("-" * 40)

    # Get predictions
    scores = classifier.predict(image)

    # Display results
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (class_name, confidence) in enumerate(sorted_scores, 1):
        bar = "█" * int(confidence * 40)
        print(f"  {rank}. {class_name:40s} {confidence:.4f}  {bar}")

    # Top prediction
    top_class, top_conf = sorted_scores[0]
    print(f"\n  → Prediction: {top_class} ({top_conf:.2%} confidence)")
    print(f"\n{'='*60}")
    print("  NOTE: This model is for RESEARCH PURPOSES ONLY.")
    print("  It is NOT intended for clinical diagnosis.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
