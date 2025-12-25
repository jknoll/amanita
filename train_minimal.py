"""
Minimal training script for BEiT model on FungiTastic dataset.
Bypasses fgvc library's model loading to avoid HuggingFace Hub model issues.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# Import fgvc components we can use
from fgvc.datasets import ImageDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_metadata(csv_path):
    """Load metadata CSV file."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"  Classes: {df['class_id'].nunique()}")
    print(f"  Class ID range: {df['class_id'].min()} to {df['class_id'].max()}")
    return df


def create_dataloaders(train_df, val_df, batch_size=16, num_workers=4, image_size=224):
    """Create training and validation dataloaders."""

    # ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # Training transforms with heavy augmentation
    train_transform = A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.08, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
        A.ToGray(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Validation transforms (no augmentation)
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # Create datasets
    train_dataset = ImageDataset(train_df, transform=train_transform)
    val_dataset = ImageDataset(val_df, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (images, targets, _) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.cpu().numpy())

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for images, targets, _ in pbar:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return avg_loss, accuracy, f1


def main():
    parser = argparse.ArgumentParser(description='Minimal BEiT training script')
    parser.add_argument('--train-path', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--val-path', type=str, required=True, help='Path to validation CSV')
    parser.add_argument('--test-path', type=str, required=True, help='Path to test CSV')
    parser.add_argument('--model', type=str,
                        default='hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224',
                        help='Model name')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--save-dir', type=str, default='./checkpoints', help='Save directory')

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load metadata
    print("Loading metadata...")
    train_df = load_metadata(args.train_path)
    val_df = load_metadata(args.val_path)
    test_df = load_metadata(args.test_path)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = timm.create_model(args.model, pretrained=True)

    # Get model info
    num_classes = model.head.out_features if hasattr(model, 'head') else model.fc.out_features
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model classes: {num_classes}")
    print(f"  Parameters: {num_params:.1f}M")

    model = model.to(device)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_df, val_df,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    best_f1 = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)

        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

        # Validate
        val_loss, val_acc, val_f1 = validate(
            model, val_loader, criterion, device
        )

        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        # Scheduler step
        scheduler.step(val_f1)

        # Save best model
        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"✓ Saved best model (F1: {best_f1:.4f})")

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
