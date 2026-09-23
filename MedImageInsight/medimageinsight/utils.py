"""
Evaluation Utilities: ROC Curves, Metrics, and Visualization.

Provides tools for generating publication-quality ROC curves, computing
mAUC (mean Area Under Curve), and visualizing classification results —
all key requirements for regulatory compliance as highlighted in the paper.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


def compute_roc_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: List[str],
) -> Dict[str, dict]:
    """
    Compute ROC curves for multi-class classification.

    Args:
        y_true: Ground truth labels (N,) — integer class indices.
        y_scores: Predicted scores (N, num_classes).
        class_names: List of class name strings.

    Returns:
        Dictionary mapping class names to {'fpr', 'tpr', 'auc'} dicts.
    """
    results = {}
    num_classes = len(class_names)

    for i, class_name in enumerate(class_names):
        # One-vs-rest binary labels for this class
        binary_true = (y_true == i).astype(int)

        if np.sum(binary_true) == 0 or np.sum(binary_true) == len(binary_true):
            logger.warning(f"Class '{class_name}' has only one label value, skipping ROC.")
            continue

        fpr, tpr, thresholds = roc_curve(binary_true, y_scores[:, i])
        roc_auc = auc(fpr, tpr)

        results[class_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
        }

    return results


def compute_mauc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    average: str = "macro",
) -> float:
    """
    Compute mean AUC (mAUC) — the primary metric used in the paper.

    Args:
        y_true: Ground truth labels (N,).
        y_scores: Predicted scores (N, num_classes).
        average: Averaging strategy ('macro', 'weighted', 'micro').

    Returns:
        Mean AUC score.
    """
    try:
        return float(roc_auc_score(y_true, y_scores, multi_class="ovr", average=average))
    except ValueError as e:
        logger.warning(f"Could not compute mAUC: {e}")
        return 0.0


def plot_roc_curves(
    roc_data: Dict[str, dict],
    title: str = "ROC Curves — MedImageInsight",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot publication-quality ROC curves (similar to Fig. 3 in the paper).

    Args:
        roc_data: Output from compute_roc_curves().
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save the figure to this path.
        show: Whether to display the plot.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Color palette for classes
    colors = plt.cm.Set2(np.linspace(0, 1, len(roc_data)))

    for (class_name, data), color in zip(roc_data.items(), colors):
        ax.plot(
            data["fpr"],
            data["tpr"],
            color=color,
            lw=2,
            label=f"{class_name} (AUC = {data['auc']:.3f})",
        )

    # Random classifier diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random (AUC = 0.500)")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Compute and display mean AUC
    mean_auc = np.mean([d["auc"] for d in roc_data.values()])
    ax.text(
        0.6, 0.15,
        f"Mean AUC = {mean_auc:.3f}",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"ROC curves saved to '{save_path}'")

    if show:
        plt.show()

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.
        show: Whether to display.

    Returns:
        matplotlib Figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(
                j, i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > thresh else "black",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_search_results(
    query_image,
    evidence: List[dict],
    figsize: Tuple[int, int] = (18, 4),
    max_results: int = 5,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize image-image search results with similarity scores.

    Displays the query image alongside its nearest neighbors,
    similar to Figure 2 in the paper.

    Args:
        query_image: PIL Image (the query).
        evidence: List of evidence dicts from KNNClassifier.predict_with_evidence().
        figsize: Figure size.
        max_results: Maximum number of results to display.
        save_path: Optional save path.
        show: Whether to display.

    Returns:
        matplotlib Figure.
    """
    n_results = min(max_results, len(evidence))
    fig, axes = plt.subplots(1, n_results + 1, figsize=figsize)

    # Query image
    axes[0].imshow(query_image)
    axes[0].set_title("Query Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    axes[0].set_frame_on(True)
    for spine in axes[0].spines.values():
        spine.set_edgecolor("blue")
        spine.set_linewidth(3)

    # Search results
    for i, entry in enumerate(evidence[:n_results]):
        ax = axes[i + 1]

        if entry.get("path"):
            try:
                result_img = Image.open(entry["path"])
                ax.imshow(result_img)
            except Exception:
                ax.text(0.5, 0.5, "Image\nNot Available", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, f"DB Index:\n{entry['database_index']}", ha="center", va="center")

        label = entry["label"]
        sim = entry["similarity"]
        ax.set_title(f"#{entry['rank']}: {label}\nSim: {sim:.3f}", fontsize=10)
        ax.axis("off")

        # Color border based on match
        for spine in ax.spines.values():
            spine.set_edgecolor("green" if sim > 0.5 else "orange")
            spine.set_linewidth(2)

    plt.suptitle("MedImageInsight — Image-Image Search Results", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> str:
    """
    Print a formatted classification report.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.

    Returns:
        Classification report string.
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    return report


def fairness_analysis(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    demographics: np.ndarray,
    demographic_name: str = "group",
) -> Dict[str, float]:
    """
    Perform fairness analysis by stratifying AUC across demographic groups.

    Mirrors Table 3 and Tables 9-10 from the paper.

    Args:
        y_true: Ground truth labels.
        y_scores: Predicted scores.
        demographics: Demographic group labels for each sample.
        demographic_name: Name of the demographic variable.

    Returns:
        Dictionary mapping group names to AUC scores.
    """
    groups = np.unique(demographics)
    results = {}

    for group in groups:
        mask = demographics == group
        if np.sum(mask) < 10:
            logger.warning(f"Group '{group}' has fewer than 10 samples, skipping.")
            continue

        group_true = y_true[mask]
        group_scores = y_scores[mask]

        try:
            if len(group_scores.shape) > 1 and group_scores.shape[1] > 2:
                group_auc = roc_auc_score(
                    group_true, group_scores, multi_class="ovr", average="macro"
                )
            else:
                group_auc = roc_auc_score(group_true, group_scores[:, 1])
            results[str(group)] = float(group_auc)
        except ValueError as e:
            logger.warning(f"Could not compute AUC for group '{group}': {e}")

    # Report
    print(f"\n{'='*50}")
    print(f"Fairness Analysis by {demographic_name}")
    print(f"{'='*50}")
    for group, auc_val in results.items():
        print(f"  {group:>20s}: AUC = {auc_val:.4f}")
    if results:
        vals = list(results.values())
        print(f"  {'Max Gap':>20s}: {max(vals) - min(vals):.4f}")
    print(f"{'='*50}\n")

    return results
