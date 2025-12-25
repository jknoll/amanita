"""
ISC-compatible training script for multi-task BEiT model on Strong Compute cluster.

This script is adapted for distributed training on the Strong Compute ISC cluster
using cycling_utils for interruptible and resumable training.

Based on: https://github.com/StrongResearch/isc-demos
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ISC cycling_utils imports
from cycling_utils import (
    InterruptableDistributedSampler,
    MetricsTracker,
    AtomicDirectory,
    atomic_torch_save,
)

from models.beit_multitask import create_beit_multitask
from dataset_multitask import FungiTasticMultiTask
from losses import MultiTaskLoss
from metrics import MultiTaskMetrics, format_metrics


class Timer:
    """Simple timer for tracking execution time of different steps."""

    def __init__(self, is_master=True):
        self.is_master = is_master
        self.start_time = time.time()
        self.last_time = self.start_time

    def report(self, step_name):
        """Report time elapsed for a step."""
        if not self.is_master:
            return
        current_time = time.time()
        step_time = current_time - self.last_time
        total_time = current_time - self.start_time
        print(f"[{total_time:.2f}s] {step_name} (took {step_time:.2f}s)")
        self.last_time = current_time


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train multi-task BEiT on ISC cluster')

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
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use-class-weights', action='store_true', default=False,
                        help='Use class balancing weights in loss')

    # Distributed training arguments (ISC-specific)
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--world-size', type=int, default=1,
                        help='Number of GPUs')
    parser.add_argument('--is-master', action='store_true', default=False,
                        help='Whether this is the master process')

    return parser.parse_args()


def setup_distributed(args):
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])

    args.is_master = args.rank == 0 if hasattr(args, 'rank') else True

    # Initialize process group for distributed training
    if args.world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)

    return args


def load_data_and_mappings(args, timer):
    """Load datasets and taxonomic mappings."""
    if args.is_master:
        print("\nLoading data and mappings...")

    # Load mappings
    with open(args.mappings_path, 'r') as f:
        mappings = json.load(f)

    num_classes = mappings['metadata']['num_classes']

    if args.is_master:
        print(f"  Number of classes per rank:")
        for rank, n in num_classes.items():
            print(f"    {rank:8s}: {n:5d}")

    # Load metadata
    train_df = pd.read_csv(args.train_path)
    val_df = pd.read_csv(args.val_path)
    test_df = pd.read_csv(args.test_path)

    if args.is_master:
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")

    timer.report("Loaded data and mappings")

    return mappings, num_classes, train_df, val_df, test_df


def create_datasets_and_samplers(train_df, val_df, mappings, args, timer):
    """Create datasets and distributed samplers."""
    # Create datasets
    train_dataset = FungiTasticMultiTask(
        train_df,
        mappings,
        is_training=True
    )

    val_dataset = FungiTasticMultiTask(
        val_df,
        mappings,
        is_training=False
    )

    timer.report("Created datasets")

    # Create distributed samplers (ISC-specific)
    train_sampler = InterruptableDistributedSampler(train_dataset)
    val_sampler = InterruptableDistributedSampler(val_dataset)

    timer.report("Initialized samplers")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True
    )

    timer.report("Initialized dataloaders")

    return train_loader, val_loader, train_sampler, val_sampler


def create_model_optimizer_criterion(num_classes, args, device, timer):
    """Create multi-task model, optimizer, criterion, and scheduler."""
    if args.is_master:
        print("\nCreating model...")

    model = create_beit_multitask(
        pretrained=args.pretrained,
        num_classes_dict=num_classes,
        model_name=args.model_name
    )

    model = model.to(device)

    # Wrap model for distributed training
    if args.world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    timer.report("Created and wrapped model")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=args.is_master
    )

    # Loss function
    if args.is_master:
        print("\nCreating loss function...")

    if args.use_class_weights:
        if args.is_master:
            print("  Loading class weights from:", args.class_weights_path)
        class_weights = torch.load(args.class_weights_path)
        criterion = MultiTaskLoss(class_weights=class_weights, device=device)
    else:
        if args.is_master:
            print("  Using unweighted loss")
        criterion = MultiTaskLoss(device=device)

    timer.report("Created optimizer and criterion")

    return model, optimizer, scheduler, criterion


def train_epoch(model, train_loader, train_sampler, criterion, optimizer,
                device, epoch, args, writer, metrics_tracker):
    """
    Train for one epoch with ISC-compatible distributed training.

    Returns:
        dict: Training metrics
    """
    model.train()
    train_sampler.set_epoch(epoch)  # Important for proper shuffling in distributed training

    local_metrics = MultiTaskMetrics()
    total_loss = 0.0
    num_batches = 0

    # Only show progress bar on master process
    if args.is_master:
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
    else:
        progress_bar = train_loader

    for batch_idx, (images, labels, _) in enumerate(progress_bar):
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
        local_metrics.update(outputs, labels)

        # Update progress bar (master only)
        if args.is_master:
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss/num_batches:.4f}"
            })

        # Log to tensorboard (master only)
        if args.is_master and writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/batch_loss', loss.item(), global_step)

    # Compute final metrics
    avg_loss = total_loss / num_batches
    local_metrics_dict = local_metrics.compute()

    # Aggregate metrics across all GPUs using MetricsTracker
    metrics_dict = {
        'loss': avg_loss,
        **local_metrics_dict
    }

    if args.world_size > 1:
        aggregated_metrics = metrics_tracker.aggregate_metrics(metrics_dict)
        return aggregated_metrics
    else:
        return metrics_dict


def validate(model, val_loader, val_sampler, criterion, device, epoch,
             args, writer, metrics_tracker):
    """
    Validate the model with ISC-compatible distributed validation.

    Returns:
        dict: Validation metrics
    """
    model.eval()
    val_sampler.set_epoch(epoch)

    local_metrics = MultiTaskMetrics()
    total_loss = 0.0
    num_batches = 0

    # Only show progress bar on master process
    if args.is_master:
        progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch}")
    else:
        progress_bar = val_loader

    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(progress_bar):
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
            local_metrics.update(outputs, labels)

            # Update progress bar (master only)
            if args.is_master:
                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{total_loss/num_batches:.4f}"
                })

    # Compute final metrics
    avg_loss = total_loss / num_batches
    local_metrics_dict = local_metrics.compute()

    metrics_dict = {
        'loss': avg_loss,
        **local_metrics_dict
    }

    # Aggregate metrics across all GPUs
    if args.world_size > 1:
        aggregated_metrics = metrics_tracker.aggregate_metrics(metrics_dict)
        return aggregated_metrics
    else:
        return metrics_dict


def save_checkpoint(model, optimizer, scheduler, train_sampler, val_sampler,
                   epoch, metrics, saver, args):
    """Save model checkpoint using ISC AtomicDirectory."""
    # Only master process saves
    if not args.is_master:
        return

    # Get model state dict (unwrap DDP if needed)
    if args.world_size > 1:
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_sampler_state': train_sampler.state_dict(),
        'val_sampler_state': val_sampler.state_dict(),
        'metrics': metrics
    }

    # Save using atomic save (ISC-specific)
    checkpoint_path = f'checkpoint_epoch_{epoch}.pt'
    atomic_torch_save(saver, checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")


def main():
    """Main training loop for ISC cluster."""
    args = parse_args()

    # Setup distributed training
    args = setup_distributed(args)

    timer = Timer(is_master=args.is_master)

    if args.is_master:
        print("=" * 60)
        print("Multi-Task BEiT Training (ISC Cluster)")
        print("=" * 60)
        print(f"World size: {args.world_size}")
        print(f"Rank: {args.rank if hasattr(args, 'rank') else 0}")

    # Device setup
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    if args.is_master:
        print(f"\nDevice: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(args.local_rank)}")

    timer.report("Setup complete")

    # ISC-specific: Setup TensorBoard writer (master only)
    writer = None
    if args.is_master and 'LOSSY_ARTIFACT_PATH' in os.environ:
        log_dir = os.environ['LOSSY_ARTIFACT_PATH']
        writer = SummaryWriter(log_dir=log_dir)
        timer.report("Initialized TensorBoard writer")

    # ISC-specific: Setup checkpoint saver
    if 'CHECKPOINT_ARTIFACT_PATH' in os.environ:
        output_directory = os.environ['CHECKPOINT_ARTIFACT_PATH']
    else:
        output_directory = './checkpoints_multitask_isc'
        os.makedirs(output_directory, exist_ok=True)

    saver = AtomicDirectory(output_directory=output_directory, is_master=args.is_master)
    timer.report("Initialized checkpoint saver")

    # ISC-specific: Setup metrics tracker for distributed aggregation
    metrics_tracker = MetricsTracker(is_master=args.is_master)
    timer.report("Initialized metrics tracker")

    # Load data
    mappings, num_classes, train_df, val_df, test_df = load_data_and_mappings(args, timer)

    # Create datasets and samplers
    train_loader, val_loader, train_sampler, val_sampler = create_datasets_and_samplers(
        train_df, val_df, mappings, args, timer
    )

    if args.is_master:
        print(f"  Train batches per GPU: {len(train_loader)}")
        print(f"  Val batches per GPU:   {len(val_loader)}")

    # Create model, optimizer, criterion
    model, optimizer, scheduler, criterion = create_model_optimizer_criterion(
        num_classes, args, device, timer
    )

    # Training loop
    if args.is_master:
        print("\n" + "=" * 60)
        print(f"Starting training for {args.epochs} epochs")
        print("=" * 60)

    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        if args.is_master:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print("-" * 60)

        # Train
        train_metrics = train_epoch(
            model, train_loader, train_sampler, criterion, optimizer,
            device, epoch, args, writer, metrics_tracker
        )

        if args.is_master:
            print("\nTrain Metrics:")
            print(f"  Loss: {train_metrics['loss']:.4f}")
            print(f"  Avg Accuracy: {train_metrics['avg_acc']:.2%}")
            print(f"  Avg F1: {train_metrics['avg_f1']:.2%}")
            print(f"  Hierarchical Acc: {train_metrics['hierarchical_acc']:.2%}")

            # Log to tensorboard
            if writer is not None:
                writer.add_scalar('epoch/train_loss', train_metrics['loss'], epoch)
                writer.add_scalar('epoch/train_acc', train_metrics['avg_acc'], epoch)
                writer.add_scalar('epoch/train_f1', train_metrics['avg_f1'], epoch)

        # Validate
        val_metrics = validate(
            model, val_loader, val_sampler, criterion, device, epoch,
            args, writer, metrics_tracker
        )

        if args.is_master:
            print("\nValidation Metrics:")
            print(f"  Loss: {val_metrics['loss']:.4f}")
            print(f"  Avg Accuracy: {val_metrics['avg_acc']:.2%}")
            print(f"  Avg F1: {val_metrics['avg_f1']:.2%}")
            print(f"  Hierarchical Acc: {val_metrics['hierarchical_acc']:.2%}")

            # Log to tensorboard
            if writer is not None:
                writer.add_scalar('epoch/val_loss', val_metrics['loss'], epoch)
                writer.add_scalar('epoch/val_acc', val_metrics['avg_acc'], epoch)
                writer.add_scalar('epoch/val_f1', val_metrics['avg_f1'], epoch)

        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_metrics['avg_f1'])

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, train_sampler, val_sampler,
            epoch, val_metrics, saver, args
        )

        # Save best model
        if args.is_master and val_metrics['avg_f1'] > best_val_f1:
            best_val_f1 = val_metrics['avg_f1']
            best_epoch = epoch

            # Save best model
            if args.world_size > 1:
                model_state = model.module.state_dict()
            else:
                model_state = model.state_dict()

            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'metrics': val_metrics
            }
            atomic_torch_save(saver, best_checkpoint, 'best_model.pt')
            print(f"  ★ New best model! (F1: {best_val_f1:.2%})")

    if args.is_master:
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"  Best epoch: {best_epoch}")
        print(f"  Best val F1: {best_val_f1:.2%}")
        print("=" * 60)

        if writer is not None:
            writer.close()

    # Cleanup distributed training
    if args.world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
