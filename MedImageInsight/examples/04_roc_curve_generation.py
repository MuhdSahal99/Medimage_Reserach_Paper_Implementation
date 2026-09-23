"""
Example 04: ROC Curve Generation

Demonstrates generation of ROC curves for regulatory compliance,
a key feature of MedImageInsight highlighted in the paper.

This generates publication-quality ROC curves with per-class AUC
and mean AUC, suitable for FDA/CE regulatory submissions.

Usage:
    python examples/04_roc_curve_generation.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from medimageinsight.utils import (
    compute_roc_curves,
    compute_mauc,
    plot_roc_curves,
    plot_confusion_matrix,
    print_classification_report,
    fairness_analysis,
)


def main():
    print(f"\n{'='*60}")
    print(f"  MedImageInsight — ROC Curve Generation")
    print(f"  (Simulated evaluation for demonstration)")
    print(f"{'='*60}\n")

    # --- Simulate evaluation data ---
    # In practice, you would replace this with actual model predictions
    np.random.seed(42)
    n_samples = 500
    n_classes = 4
    class_names = [
        "Normal",
        "Pneumonia",
        "Effusion",
        "Cardiomegaly",
    ]

    # Generate realistic-looking predictions
    y_true = np.random.randint(0, n_classes, n_samples)

    # Create scores that correlate with true labels (simulating a good model)
    y_scores = np.random.dirichlet(np.ones(n_classes) * 0.5, n_samples)
    for i in range(n_samples):
        y_scores[i, y_true[i]] += np.random.uniform(1.5, 3.0)
    # Normalize
    y_scores = y_scores / y_scores.sum(axis=1, keepdims=True)

    y_pred = np.argmax(y_scores, axis=1)

    # --- Compute ROC curves ---
    print("[1/5] Computing ROC curves...")
    roc_data = compute_roc_curves(y_true, y_scores, class_names)

    for class_name, data in roc_data.items():
        print(f"      {class_name:20s}: AUC = {data['auc']:.4f}")

    # --- Compute mAUC ---
    print("\n[2/5] Computing mean AUC (mAUC)...")
    mauc = compute_mauc(y_true, y_scores, average="macro")
    print(f"      mAUC (macro) = {mauc:.4f}")

    mauc_weighted = compute_mauc(y_true, y_scores, average="weighted")
    print(f"      mAUC (weighted) = {mauc_weighted:.4f}")

    # --- Plot ROC curves ---
    print("\n[3/5] Generating ROC curve plot...")
    plot_roc_curves(
        roc_data,
        title="MedImageInsight — Chest X-Ray Classification ROC",
        save_path="roc_curves.png",
        show=False,
    )
    print("      Saved to: roc_curves.png")

    # --- Classification report ---
    print("\n[4/5] Classification report:")
    print_classification_report(y_true, y_pred, class_names)

    # --- Confusion matrix ---
    print("[5/5] Generating confusion matrix...")
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        title="MedImageInsight — Confusion Matrix",
        save_path="confusion_matrix.png",
        show=False,
    )
    print("      Saved to: confusion_matrix.png")

    # --- Fairness analysis ---
    print("\n[Bonus] Fairness analysis by gender...")
    genders = np.random.choice(["Female", "Male"], n_samples, p=[0.45, 0.55])
    fairness_analysis(y_true, y_scores, genders, "Gender")

    print(f"\n{'='*60}")
    print("  All evaluation artifacts generated:")
    print("    • roc_curves.png       — Publication-quality ROC curves")
    print("    • confusion_matrix.png — Confusion matrix heatmap")
    print(f"\n  These are suitable for regulatory submissions (FDA/CE).")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
