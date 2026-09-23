"""
Example 05: Fine-Tuning MedImageInsight

Demonstrates how to fine-tune MedImageInsight on a custom medical
imaging dataset, following the approach in the paper (Section 3.2.2):

    - Freeze early layers of image encoder (stage 1)
    - Freeze token/positional embeddings + first 3 blocks of text encoder
    - Learning rate: 1e-6, batch size: 1024, weight decay: 0.4
    - Training for 800 iterations

Usage:
    python examples/05_fine_tuning.py --data_dir ./custom_dataset/ --epochs 10
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from medimageinsight import MedImageInsightModel
from medimageinsight.preprocessing import MedicalImagePreprocessor


class MedicalImageDataset(Dataset):
    """Simple dataset for fine-tuning with folder structure: class_name/image.png"""

    def __init__(self, data_dir: str, preprocessor: MedicalImagePreprocessor, training: bool = True):
        self.preprocessor = preprocessor
        self.training = training
        self.samples = []
        self.class_names = []

        data_path = Path(data_dir)
        for class_dir in sorted(data_path.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
                class_idx = self.class_names.index(class_name)

                for img_path in class_dir.glob("*"):
                    if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".dcm"}:
                        self.samples.append((str(img_path), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.preprocessor.load(path)
        tensor = self.preprocessor.preprocess(image, training=self.training)
        return tensor, label


class LinearClassifier(nn.Module):
    """Linear classification head on top of frozen MedImageInsight embeddings."""

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, embeddings):
        return self.classifier(embeddings)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune MedImageInsight on custom medical imaging data"
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Dataset directory with class subfolders")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for classifier head")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--output", type=str, default="finetuned_classifier.pt",
                        help="Path to save fine-tuned classifier")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  MedImageInsight — Fine-Tuning Pipeline")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # --- Load base model ---
    print("\n[1/4] Loading MedImageInsight base model...")
    model = MedImageInsightModel(device=str(device))
    model.load()

    if args.data_dir:
        # --- Load dataset ---
        print(f"\n[2/4] Loading dataset from '{args.data_dir}'...")
        preprocessor = MedicalImagePreprocessor()
        dataset = MedicalImageDataset(args.data_dir, preprocessor, training=True)
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
        )

        num_classes = len(dataset.class_names)
        print(f"      Classes: {dataset.class_names}")
        print(f"      Samples: {len(dataset)}")
    else:
        # Demo mode
        print("\n[2/4] Running in demo mode (no dataset provided)...")
        num_classes = 3
        print("      Simulating 3-class classification task")

    # --- Build classifier head ---
    print(f"\n[3/4] Building classifier head ({num_classes} classes)...")
    embedding_dim = model.embedding_dim
    if embedding_dim <= 0:
        embedding_dim = 768  # fallback
    classifier = LinearClassifier(embedding_dim, num_classes).to(device)

    # --- Training loop ---
    print(f"\n[4/4] Training classifier head...")
    print(f"      Epochs: {args.epochs}")
    print(f"      Learning rate: {args.lr}")
    print(f"      Batch size: {args.batch_size}")
    print(f"      Strategy: Frozen backbone + trainable classifier head")
    print()

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.data_dir:
        for epoch in range(args.epochs):
            classifier.train()
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(dataloader, desc=f"  Epoch {epoch+1}/{args.epochs}")
            for batch_images, batch_labels in pbar:
                # Extract embeddings (frozen backbone)
                with torch.no_grad():
                    embeddings = model.encode_image(
                        [Image.fromarray(
                            (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        ) for img in batch_images]
                    )
                    embeddings = torch.from_numpy(embeddings).to(device)

                # Forward through classifier head
                logits = classifier(embeddings)
                labels = batch_labels.to(device)
                loss = criterion(logits, labels)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct/total:.2%}",
                })

            scheduler.step()
            avg_loss = total_loss / len(dataloader)
            accuracy = correct / total
            print(f"      Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.2%}")

        # Save
        torch.save({
            "classifier_state_dict": classifier.state_dict(),
            "num_classes": num_classes,
            "embedding_dim": embedding_dim,
        }, args.output)
        print(f"\n  Classifier saved to: {args.output}")

    else:
        print("  Skipping training (no dataset). Provide --data_dir to train.")
        print("\n  Expected folder structure:")
        print("    data_dir/")
        print("    ├── class_1/")
        print("    │   ├── image_001.png")
        print("    │   └── image_002.png")
        print("    ├── class_2/")
        print("    │   ├── image_003.png")
        print("    │   └── image_004.png")
        print("    └── class_3/")
        print("        └── image_005.png")

    print(f"\n{'='*60}")
    print("  Fine-tuning approach (from paper Section 3.2.2):")
    print("    • Freeze image encoder early layers (stage 1)")
    print("    • Freeze text encoder embeddings + first 3 blocks")
    print("    • This example uses a simpler approach: frozen backbone")
    print("      + trainable linear classifier head (transfer learning)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
