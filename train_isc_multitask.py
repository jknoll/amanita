#!/usr/bin/env python3
"""
ISC-compatible training script for multi-task BEiT model on Strong Compute cluster.

This script is adapted for distributed training on the Strong Compute ISC cluster
using cycling_utils for interruptible and resumable training.

**LOCAL/CLUSTER DETECTION:**
- Detects cluster mode if RANK or WORLD_SIZE environment variables are set (set by torchrun)
- Falls back to local single-GPU mode otherwise
- In local mode: uses standard PyTorch components, no cycling_utils required
- In cluster mode: uses ISC cycling_utils for distributed training

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
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ============================================================
# DETECTION MECHANISM: Check for distributed environment variables
# ============================================================
IS_DISTRIBUTED = 'RANK' in os.environ or 'WORLD_SIZE' in os.environ

# Conditionally import ISC cycling_utils only in cluster mode
if IS_DISTRIBUTED:
    try:
        from cycling_utils import (
            InterruptableDistributedSampler,
            MetricsTracker,
            AtomicDirectory,
            atomic_torch_save,
        )
        CYCLING_UTILS_AVAILABLE = True
    except ImportError:
        print("WARNING: cycling_utils not available, falling back to local mode")
        IS_DISTRIBUTED = False
        CYCLING_UTILS_AVAILABLE = False
else:
    CYCLING_UTILS_AVAILABLE = False

from models.beit_multitask import create_beit_multitask
from dataset_multitask import FungiTasticMultiTask
from losses import MultiTaskLoss
from metrics import MultiTaskMetrics, format_metrics

# Print mode at import time
if IS_DISTRIBUTED:
    print("🌐 CLUSTER MODE: Distributed training with cycling_utils")
else:
    print("💻 LOCAL MODE: Single-GPU training")


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
    parser.add_argument('--image-root', type=str, default=None,
                        help='Root directory for images (required if CSV has no image_path column)')

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
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume from latest checkpoint if available')
    parser.add_argument('--checkpoint-path', type=str, default=None,
                        help='Explicit path to checkpoint file to resume from (overrides --resume)')

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
    if IS_DISTRIBUTED:
        # Cluster mode: setup distributed training
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.rank = int(os.environ['RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        else:
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0

        args.is_master = args.rank == 0

        # Initialize process group for distributed training
        if args.world_size > 1:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(args.local_rank)
    else:
        # Local mode: single GPU
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.is_master = True

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
    # Import transforms
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    # Create transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])

    # Create datasets
    train_dataset = FungiTasticMultiTask(train_df, train_transform, mappings, args.image_root, split='train')
    val_dataset = FungiTasticMultiTask(val_df, val_transform, mappings, args.image_root, split='val')

    timer.report("Created datasets")

    # Create samplers based on mode
    if IS_DISTRIBUTED and CYCLING_UTILS_AVAILABLE:
        # Cluster mode: use ISC samplers
        train_sampler = InterruptableDistributedSampler(train_dataset)
        val_sampler = InterruptableDistributedSampler(val_dataset)
    else:
        # Local mode: use standard PyTorch samplers
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)

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

    # Set epoch for distributed sampler
    if IS_DISTRIBUTED and hasattr(train_sampler, 'set_epoch'):
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

    # Return local metrics (master process will report)
    metrics_dict = {
        'loss': avg_loss,
        **local_metrics_dict
    }

    return metrics_dict


def validate(model, val_loader, val_sampler, criterion, device, epoch,
             args, writer, metrics_tracker):
    """
    Validate the model with ISC-compatible distributed validation.

    Returns:
        dict: Validation metrics
    """
    model.eval()

    # Set epoch for distributed sampler
    if IS_DISTRIBUTED and hasattr(val_sampler, 'set_epoch'):
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

    # Return local metrics (master process will report)
    metrics_dict = {
        'loss': avg_loss,
        **local_metrics_dict
    }

    return metrics_dict


def save_checkpoint(model, optimizer, scheduler, train_sampler, val_sampler,
                   epoch, metrics, saver, args):
    """Save model checkpoint (ISC atomic save or standard torch.save)."""
    checkpoint_filename = f'checkpoint_epoch_{epoch}.pt'

    # Save using appropriate method
    if hasattr(saver, 'prepare_checkpoint_directory'):
        # ISC cluster mode: ALL ranks must participate in collective operations
        checkpoint_dir = saver.prepare_checkpoint_directory()

        if args.is_master:
            # Only master creates and saves the checkpoint
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
                'metrics': metrics
            }

            # Add sampler state only if using distributed samplers
            if IS_DISTRIBUTED and hasattr(train_sampler, 'state_dict'):
                checkpoint['train_sampler_state'] = train_sampler.state_dict()
                checkpoint['val_sampler_state'] = val_sampler.state_dict()

            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            atomic_torch_save(checkpoint, checkpoint_path)

        # ALL ranks must call symlink_latest (has barrier inside)
        saver.symlink_latest(checkpoint_dir)

        if args.is_master:
            print(f"  Checkpoint saved: {checkpoint_filename}")
    else:
        # Local mode: only master saves (single GPU, no distributed)
        if not args.is_master:
            return

        # Get model state dict
        model_state = model.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics
        }

        checkpoint_path = os.path.join(saver, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_filename}")


def load_checkpoint(model, optimizer, scheduler, train_sampler, val_sampler, saver, args):
    """
    Load checkpoint if available.

    Returns:
        tuple: (start_epoch, best_val_f1, best_epoch) or (0, 0.0, 0) if no checkpoint
    """
    checkpoint_path = None

    # Check for explicit checkpoint path first
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        if args.is_master:
            print(f"Using explicit checkpoint path: {checkpoint_path}")
    else:
        # Find latest checkpoint automatically
        if hasattr(saver, 'prepare_checkpoint_directory'):
            # ISC mode: Look for latest symlink or AtomicDirectory checkpoints
            checkpoint_dir = None

            # Try 'latest' symlink first
            latest_link = os.path.join(saver.output_directory, 'latest')
            if os.path.islink(latest_link) and os.path.exists(latest_link):
                checkpoint_dir = latest_link

            # If latest symlink doesn't work, look for AtomicDirectory.latest_checkpoint
            if checkpoint_dir is None:
                atomic_latest = os.path.join(saver.output_directory, 'AtomicDirectory.latest_checkpoint')
                if os.path.islink(atomic_latest) and os.path.exists(atomic_latest):
                    checkpoint_dir = atomic_latest

            # If symlinks are broken, find the latest AtomicDirectory_checkpoint_* directory
            if checkpoint_dir is None:
                atomic_dirs = []
                if os.path.isdir(saver.output_directory):
                    for name in os.listdir(saver.output_directory):
                        if name.startswith('AtomicDirectory_checkpoint_'):
                            full_path = os.path.join(saver.output_directory, name)
                            if os.path.isdir(full_path):
                                # Extract checkpoint number for sorting
                                try:
                                    checkpoint_num = int(name.split('_')[-1])
                                    atomic_dirs.append((checkpoint_num, full_path))
                                except ValueError:
                                    pass

                    if atomic_dirs:
                        # Sort by checkpoint number and take the highest
                        atomic_dirs.sort(reverse=True)
                        checkpoint_dir = atomic_dirs[0][1]
                        if args.is_master:
                            print(f"Found latest checkpoint directory: {os.path.basename(checkpoint_dir)}")

            # Now look for checkpoint files in the directory
            if checkpoint_dir and os.path.isdir(checkpoint_dir):
                # Find checkpoint file (prefer checkpoint_epoch_*.pt, fallback to best_model.pt)
                checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
                if checkpoint_files:
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_files[0])
                elif os.path.exists(os.path.join(checkpoint_dir, 'best_model.pt')):
                    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
        else:
            # Local mode: Look for latest checkpoint in directory
            if os.path.isdir(saver):
                checkpoint_files = sorted([f for f in os.listdir(saver) if f.startswith('checkpoint_epoch_') and f.endswith('.pt')])
                if checkpoint_files:
                    checkpoint_path = os.path.join(saver, checkpoint_files[-1])
                elif os.path.exists(os.path.join(saver, 'best_model.pt')):
                    checkpoint_path = os.path.join(saver, 'best_model.pt')

    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        if args.is_master:
            print("No checkpoint found. Starting from scratch.")
        return 0, 0.0, 0

    if args.is_master:
        print(f"\nLoading checkpoint from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Restore model state
    if args.world_size > 1:
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    # Restore optimizer and scheduler
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore sampler states if available
    if IS_DISTRIBUTED and 'train_sampler_state' in checkpoint:
        if hasattr(train_sampler, 'load_state_dict'):
            train_sampler.load_state_dict(checkpoint['train_sampler_state'])
        if hasattr(val_sampler, 'load_state_dict'):
            val_sampler.load_state_dict(checkpoint['val_sampler_state'])

    start_epoch = checkpoint['epoch'] + 1  # Resume from next epoch

    # Try to get best metrics from checkpoint
    metrics = checkpoint.get('metrics', {})
    best_val_f1 = metrics.get('avg_f1', 0.0)

    if args.is_master:
        print(f"  Resuming from epoch {checkpoint['epoch']}")
        print(f"  Best val F1 so far: {best_val_f1:.2%}")

    return start_epoch, best_val_f1, checkpoint['epoch']


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

    # Setup checkpoint saver
    if 'CHECKPOINT_ARTIFACT_PATH' in os.environ:
        output_directory = os.environ['CHECKPOINT_ARTIFACT_PATH']
    else:
        output_directory = './checkpoints_multitask_isc'
        os.makedirs(output_directory, exist_ok=True)

    if IS_DISTRIBUTED and CYCLING_UTILS_AVAILABLE:
        saver = AtomicDirectory(output_directory=output_directory, is_master=args.is_master)
        timer.report("Initialized checkpoint saver (ISC)")
    else:
        saver = output_directory  # Just use the directory path in local mode
        timer.report("Initialized checkpoint saver (local)")

    # ISC-specific: Setup metrics tracker for distributed aggregation
    if IS_DISTRIBUTED and CYCLING_UTILS_AVAILABLE:
        metrics_tracker = MetricsTracker()
        timer.report("Initialized metrics tracker")
    else:
        metrics_tracker = None  # Not needed in local mode

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

    # Test checkpoint saving early to fail fast
    if args.is_master:
        print("\nTesting checkpoint save mechanism...")

    test_checkpoint = {'epoch': 0, 'test': True}
    test_filename = 'test_checkpoint.pt'

    if hasattr(saver, 'prepare_checkpoint_directory'):
        # ISC cluster mode - prepare_checkpoint_directory is collective
        test_dir = saver.prepare_checkpoint_directory()

        if args.is_master:
            # Only master saves the file
            test_path = os.path.join(test_dir, test_filename)
            atomic_torch_save(test_checkpoint, test_path)

        # ALL ranks must call symlink_latest (has barrier inside)
        saver.symlink_latest(test_dir)
    else:
        # Local mode - only runs on single GPU
        if args.is_master:
            test_path = os.path.join(saver, test_filename)
            torch.save(test_checkpoint, test_path)

    if args.is_master:
        print("  ✓ Checkpoint save test successful")
        timer.report("Tested checkpoint saving")

    # Load checkpoint if resuming
    start_epoch = 1
    best_val_f1 = 0.0
    best_epoch = 0

    if args.resume:
        start_epoch, best_val_f1, best_epoch = load_checkpoint(
            model, optimizer, scheduler, train_sampler, val_sampler, saver, args
        )
        if args.is_master:
            timer.report("Loaded checkpoint")

    # Training loop
    if args.is_master:
        print("\n" + "=" * 60)
        if start_epoch == 1:
            print(f"Starting training for {args.epochs} epochs")
        else:
            print(f"Resuming training from epoch {start_epoch}/{args.epochs}")
        print("=" * 60)

    for epoch in range(start_epoch, args.epochs + 1):
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

        # Save best model - check if this is the best
        is_new_best = args.is_master and val_metrics['avg_f1'] > best_val_f1

        if is_new_best:
            best_val_f1 = val_metrics['avg_f1']
            best_epoch = epoch

        # In ISC mode, ALL ranks must participate even if not new best
        if hasattr(saver, 'prepare_checkpoint_directory'):
            # ISC cluster mode: ALL ranks participate in collective operations
            best_dir = saver.prepare_checkpoint_directory()

            if is_new_best:
                # Only master saves when it's actually a new best
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

                best_filename = 'best_model.pt'
                best_path = os.path.join(best_dir, best_filename)
                atomic_torch_save(best_checkpoint, best_path)

            # ALL ranks call symlink_latest
            saver.symlink_latest(best_dir)

            if is_new_best:
                print(f"  ★ New best model! (F1: {best_val_f1:.2%})")
        else:
            # Local mode: only master saves if it's new best
            if is_new_best:
                model_state = model.state_dict()

                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'metrics': val_metrics
                }

                best_filename = 'best_model.pt'
                best_path = os.path.join(saver, best_filename)
                torch.save(best_checkpoint, best_path)
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
