"""
Training script skeleton for multi-task BEiT model.

This is a skeleton for Milestone 2. The actual loss function and training
logic will be implemented in Milestone 3.
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

from models.beit_multitask import create_beit_multitask
from dataset_multitask import create_multitask_dataloaders


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


def create_model_and_optimizer(num_classes, args):
    """Create multi-task model and optimizer."""
    print("\nCreating model...")

    model = create_beit_multitask(
        pretrained=args.pretrained,
        num_classes_dict=num_classes,
        model_name=args.model_name
    )

    # Optimizer (only for new heads initially, can fine-tune backbone later)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Scheduler (will be based on validation performance)
    # Placeholder - will be implemented in Milestone 3
    scheduler = None

    return model, optimizer, scheduler


def train_epoch(model, train_loader, optimizer, device):
    """
    Train for one epoch.

    NOTE: This is a skeleton. The actual loss function will be
    implemented in Milestone 3.
    """
    model.train()

    print("  [Skeleton] Training epoch - loss function not implemented yet")

    # TODO (Milestone 3):
    # - Implement multi-task loss function
    # - Combine losses from all 6 taxonomic ranks
    # - Consider hierarchical constraints
    # - Track per-rank metrics

    return {
        'loss': 0.0,  # Placeholder
        'metrics': {}  # Placeholder
    }


def validate(model, val_loader, device):
    """
    Validate the model.

    NOTE: This is a skeleton. The actual metrics will be
    implemented in Milestone 3.
    """
    model.eval()

    print("  [Skeleton] Validation - metrics not implemented yet")

    # TODO (Milestone 3):
    # - Compute per-rank accuracy
    # - Compute hierarchical accuracy
    # - Track F1 scores for each rank
    # - Consider weighted metrics

    return {
        'loss': 0.0,  # Placeholder
        'metrics': {}  # Placeholder
    }


def main():
    """Main training loop."""
    args = parse_args()

    print("=" * 60)
    print("Multi-Task BEiT Training - Skeleton")
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

    # Create model
    model, optimizer, scheduler = create_model_and_optimizer(num_classes, args)
    model = model.to(device)

    # Training loop
    print("\n" + "=" * 60)
    print("NOTE: This is a skeleton for Milestone 2")
    print("Loss function and metrics will be implemented in Milestone 3")
    print("=" * 60)

    print(f"\nWould train for {args.epochs} epochs with:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: AdamW")

    print("\nTODO for Milestone 3:")
    print("  1. Implement multi-task loss function")
    print("  2. Implement per-rank metrics (accuracy, F1)")
    print("  3. Implement hierarchical accuracy metric")
    print("  4. Add checkpoint saving logic")
    print("  5. Add learning rate scheduling")
    print("  6. Add progress tracking (tqdm, wandb)")

    print("\n" + "=" * 60)
    print("✓ Skeleton verified - ready for Milestone 3")
    print("=" * 60)


if __name__ == '__main__':
    main()
