"""
Full training script for multi-task BEiT model.

Implements complete training loop with:
- Multi-task loss function with class balancing
- Per-rank and hierarchical metrics
- Checkpoint saving
- Learning rate scheduling
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from models.beit_multitask import create_beit_multitask
from dataset_multitask import create_multitask_dataloaders
from losses import MultiTaskLoss
from metrics import MultiTaskMetrics, format_metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-task BEiT model')

    # Data arguments
    parser.add_argument('--train-path', type=str, required=True,
                        help='Path to training CSV')
    parser.add_argument('--val-path', type=str, required=True,
                        help='Path to validation CSV')
    parser.add_argument('--test-path', type=str, required=True,
                        help='Path to test CSV')
    parser.add_argument('--mappings-path', type=str, default='taxonomic_mappings.json',
                        help='Path to taxonomic mappings JSON')
    parser.add_argument('--class-weights-path', type=str, default='class_weights.pt',
                        help='Path to class weights file')

    # Model arguments
    parser.add_argument('--model-name', type=str,
                        default='hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224',
                        help='Base model name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Load pretrained weights')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use-class-weights', action='store_true', default=False,
                        help='Use class balancing weights in loss')

    # Output arguments
    parser.add_argument('--save-dir', type=str, default='./checkpoints_multitask',
                        help='Directory to save checkpoints')

    return parser.parse_args()


def load_data_and_mappings(args):
    """Load datasets and taxonomic mappings."""
    print("Loading data and mappings...")

    # Load mappings
    with open(args.mappings_path, 'r') as f:
        mappings = json.load(f)

    num_classes = mappings['metadata']['num_classes']
    print(f"  Number of classes per rank:")
    for rank, n in num_classes.items():
        print(f"    {rank:8s}: {n:5d}")

    # Load metadata
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)

    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")

    return mappings, num_classes, train_df, val_df, test_df


def create_model_optimizer_criterion(num_classes, args, device):
    """Create multi-task model, optimizer, criterion, and scheduler."""
    print("\nCreating model...")

    model = create_beit_multitask(
        pretrained=args.pretrained,
        num_classes_dict=num_classes,
        model_name=args.model_name
    )

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Loss function
    print("\nCreating loss function...")
    if args.use_class_weights:
        print("  Loading class weights from:", args.class_weights_path)
        class_weights = torch.load(args.class_weights_path)
        criterion = MultiTaskLoss(class_weights=class_weights, device=device)
    else:
        print("  Using unweighted loss")
        criterion = MultiTaskLoss(device=device)

    return model, optimizer, scheduler, criterion


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.

    Returns:
        dict: Training metrics
    """
    model.train()

    metrics_tracker = MultiTaskMetrics()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc="Training")

    for images, labels, _ in progress_bar:
        # Move to device
        images = images.to(device)
        labels = {rank: labels[rank].to(device) for rank in labels}

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)

        # Compute loss
        loss, loss_dict = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        metrics_tracker.update(outputs, labels)

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss/num_batches:.4f}"
        })

    # Compute final metrics
    avg_loss = total_loss / num_batches
    metrics = metrics_tracker.compute()

    return {
        'loss': avg_loss,
        **metrics
    }


def validate(model, val_loader, criterion, device):
    """
    Validate the model.

    Returns:
        dict: Validation metrics
    """
    model.eval()

    metrics_tracker = MultiTaskMetrics()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(val_loader, desc="Validation")

    with torch.no_grad():
        for images, labels, _ in progress_bar:
            # Move to device
            images = images.to(device)
            labels = {rank: labels[rank].to(device) for rank in labels}

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss, loss_dict = criterion(outputs, labels)

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            metrics_tracker.update(outputs, labels)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })

    # Compute final metrics
    avg_loss = total_loss / num_batches
    metrics = metrics_tracker.compute()

    return {
        'loss': avg_loss,
        **metrics
    }


def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"  Checkpoint saved: {save_path}")


def main():
    """Main training loop."""
    args = parse_args()

    print("=" * 60)
    print("Multi-Task BEiT Training")
    print("=" * 60)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Load data
    mappings, num_classes, train_df, val_df, test_df = load_data_and_mappings(args)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_multitask_dataloaders(
        train_df, val_df,
        mappings,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    # Create model, optimizer, criterion
    model, optimizer, scheduler, criterion = create_model_optimizer_criterion(
        num_classes, args, device
    )
    model = model.to(device)

    # Training loop
    print("\n" + "=" * 60)
    print(f"Starting training for {args.epochs} epochs")
    print("=" * 60)

    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)

        print("\nTrain Metrics:")
        print(f"  Loss: {train_metrics['loss']:.4f}")
        print(f"  Avg Accuracy: {train_metrics['avg_acc']:.2%}")
        print(f"  Avg F1: {train_metrics['avg_f1']:.2%}")
        print(f"  Hierarchical Acc: {train_metrics['hierarchical_acc']:.2%}")

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        print("\nValidation Metrics:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Avg Accuracy: {val_metrics['avg_acc']:.2%}")
        print(f"  Avg F1: {val_metrics['avg_f1']:.2%}")
        print(f"  Hierarchical Acc: {val_metrics['hierarchical_acc']:.2%}")

        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_metrics['avg_f1'])

        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pt')
        save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, checkpoint_path)

        # Save best model
        if val_metrics['avg_f1'] > best_val_f1:
            best_val_f1 = val_metrics['avg_f1']
            best_epoch = epoch
            best_path = os.path.join(args.save_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, best_path)
            print(f"  ★ New best model! (F1: {best_val_f1:.2%})")

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val F1: {best_val_f1:.2%}")
    print("=" * 60)


if __name__ == '__main__':
    main()
