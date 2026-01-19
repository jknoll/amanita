#!/usr/bin/env python3
"""
Training Log Analysis Script for FungiTastic Multi-Task Training

Usage:
    python analyze_training_logs.py /shared/artifacts/exp-NAME/logs/rank_0.txt
    python analyze_training_logs.py /shared/artifacts/exp-NAME/checkpoints/
"""

import sys
import re
import os
from pathlib import Path
import torch


def parse_training_log(log_path):
    """Parse training log file and extract metrics."""

    with open(log_path, 'r') as f:
        content = f.read()

    # Extract epoch-wise metrics
    epochs = []

    # Pattern for epoch headers
    epoch_pattern = re.compile(r'Epoch (\d+)/(\d+)')

    # Pattern for train metrics
    train_loss_pattern = re.compile(r'Train Metrics:\s+Loss: ([\d.]+)')
    train_acc_pattern = re.compile(r'Avg Accuracy: ([\d.]+)%')
    train_f1_pattern = re.compile(r'Avg F1: ([\d.]+)%')

    # Pattern for val metrics
    val_loss_pattern = re.compile(r'Val Metrics:\s+Loss: ([\d.]+)', re.MULTILINE)

    # Split into epoch sections
    epoch_sections = re.split(r'Epoch \d+/\d+', content)
    epoch_matches = list(epoch_pattern.finditer(content))

    for i, (section, match) in enumerate(zip(epoch_sections[1:], epoch_matches)):
        epoch_num = int(match.group(1))
        total_epochs = int(match.group(2))

        # Extract train metrics
        train_loss = train_loss_pattern.search(section)
        train_acc = train_acc_pattern.search(section)
        train_f1 = train_f1_pattern.search(section)

        # Find validation section (after train metrics)
        val_sections = section.split('Train Metrics:')
        if len(val_sections) > 1:
            val_section = val_sections[1]
            val_loss = re.search(r'Loss: ([\d.]+)', val_section)
            val_acc = re.search(r'Avg Accuracy: ([\d.]+)%', val_section)
            val_f1 = re.search(r'Avg F1: ([\d.]+)%', val_section)
        else:
            val_loss = val_acc = val_f1 = None

        # Check if this was a best model
        is_best = '★ New best model!' in section

        epoch_data = {
            'epoch': epoch_num,
            'train_loss': float(train_loss.group(1)) if train_loss else None,
            'train_acc': float(train_acc.group(1)) if train_acc else None,
            'train_f1': float(train_f1.group(1)) if train_f1 else None,
            'val_loss': float(val_loss.group(1)) if val_loss else None,
            'val_acc': float(val_acc.group(1)) if val_acc else None,
            'val_f1': float(val_f1.group(1)) if val_f1 else None,
            'is_best': is_best
        }

        epochs.append(epoch_data)

    return epochs


def analyze_checkpoint(checkpoint_path):
    """Analyze a checkpoint file to extract epoch and metrics."""

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        epoch = checkpoint.get('epoch', 'NOT FOUND')
        metrics = checkpoint.get('metrics', {})

        info = {
            'epoch': epoch,
            'val_f1': metrics.get('avg_f1', None),
            'val_accuracy': metrics.get('avg_accuracy', None),
            'val_loss': metrics.get('loss', None),
        }

        return info
    except Exception as e:
        return {'error': str(e)}


def scan_checkpoint_directory(checkpoint_dir):
    """Scan checkpoint directory and analyze all checkpoints."""

    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = []

    # Find all AtomicDirectory_checkpoint_* directories
    for atomic_dir in sorted(checkpoint_dir.glob('AtomicDirectory_checkpoint_*')):
        if atomic_dir.is_dir():
            atomic_num = int(atomic_dir.name.split('_')[-1])

            # Look for checkpoint files
            for ckpt_file in atomic_dir.glob('*.pt'):
                info = analyze_checkpoint(ckpt_file)
                info['atomic_num'] = atomic_num
                info['filename'] = ckpt_file.name
                info['path'] = str(ckpt_file)
                checkpoints.append(info)

    return checkpoints


def print_epoch_table(epochs):
    """Print formatted table of epoch results."""

    print("\n" + "="*100)
    print("EPOCH-WISE TRAINING RESULTS")
    print("="*100)

    header = f"{'Epoch':>6} | {'Train Loss':>11} | {'Train Acc':>10} | {'Train F1':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'Val F1':>7} | {'Best':>4}"
    print(header)
    print("-" * 100)

    for e in epochs:
        train_loss = f"{e['train_loss']:.4f}" if e['train_loss'] is not None else "N/A"
        train_acc = f"{e['train_acc']:.2f}%" if e['train_acc'] is not None else "N/A"
        train_f1 = f"{e['train_f1']:.2f}%" if e['train_f1'] is not None else "N/A"
        val_loss = f"{e['val_loss']:.4f}" if e['val_loss'] is not None else "N/A"
        val_acc = f"{e['val_acc']:.2f}%" if e['val_acc'] is not None else "N/A"
        val_f1 = f"{e['val_f1']:.2f}%" if e['val_f1'] is not None else "N/A"
        best = "★" if e['is_best'] else ""

        print(f"{e['epoch']:6d} | {train_loss:>11} | {train_acc:>10} | {train_f1:>9} | {val_loss:>9} | {val_acc:>8} | {val_f1:>7} | {best:>4}")

    # Summary statistics
    if epochs:
        best_val_f1 = max([e['val_f1'] for e in epochs if e['val_f1'] is not None], default=None)
        best_epoch = next((e['epoch'] for e in epochs if e['val_f1'] == best_val_f1), None)

        print("="*100)
        print(f"Best Validation F1: {best_val_f1:.2f}% (Epoch {best_epoch})")
        print(f"Total Epochs: {len(epochs)}")
        print("="*100 + "\n")


def print_checkpoint_table(checkpoints):
    """Print formatted table of checkpoint information."""

    print("\n" + "="*100)
    print("CHECKPOINT ANALYSIS")
    print("="*100)

    header = f"{'Atomic #':>9} | {'Filename':>20} | {'Epoch':>6} | {'Val F1':>8} | {'Val Acc':>9} | {'Val Loss':>9}"
    print(header)
    print("-" * 100)

    for ckpt in checkpoints:
        if 'error' in ckpt:
            print(f"Error: {ckpt['error']}")
            continue

        atomic_num = ckpt['atomic_num']
        filename = ckpt['filename'][:20]  # Truncate if needed
        epoch = ckpt['epoch'] if ckpt['epoch'] != 'NOT FOUND' else 'N/A'
        val_f1 = f"{ckpt['val_f1']:.4f}" if ckpt['val_f1'] is not None else "N/A"
        val_acc = f"{ckpt['val_accuracy']:.4f}" if ckpt['val_accuracy'] is not None else "N/A"
        val_loss = f"{ckpt['val_loss']:.4f}" if ckpt['val_loss'] is not None else "N/A"

        print(f"{atomic_num:9d} | {filename:>20} | {epoch:6} | {val_f1:>8} | {val_acc:>9} | {val_loss:>9}")

    print("="*100)

    # Note about AtomicDirectory numbers
    print("\nNOTE: AtomicDirectory numbers are sequential counters (NOT epoch numbers).")
    print("      The actual epoch is stored inside the checkpoint file.\n")


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Analyze log file:        python analyze_training_logs.py /path/to/rank_0.txt")
        print("  Analyze checkpoints:     python analyze_training_logs.py /path/to/checkpoints/")
        print("  Analyze experiment dir:  python analyze_training_logs.py /shared/artifacts/exp-NAME/")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"Error: Path does not exist: {input_path}")
        sys.exit(1)

    # Determine what to analyze
    if input_path.is_file():
        # It's a log file
        print(f"Analyzing log file: {input_path}")
        epochs = parse_training_log(input_path)
        print_epoch_table(epochs)

    elif input_path.is_dir():
        # Check if it's a checkpoint directory or experiment directory
        if (input_path / 'checkpoints').exists():
            # It's an experiment directory
            print(f"Analyzing experiment: {input_path.name}")

            # Analyze logs if they exist
            log_file = input_path / 'logs' / 'rank_0.txt'
            if log_file.exists():
                print(f"\nParsing log file: {log_file}")
                epochs = parse_training_log(log_file)
                print_epoch_table(epochs)

            # Analyze checkpoints
            checkpoint_dir = input_path / 'checkpoints'
            print(f"\nScanning checkpoints: {checkpoint_dir}")
            checkpoints = scan_checkpoint_directory(checkpoint_dir)
            print_checkpoint_table(checkpoints)

        else:
            # Assume it's a checkpoint directory
            print(f"Scanning checkpoint directory: {input_path}")
            checkpoints = scan_checkpoint_directory(input_path)
            print_checkpoint_table(checkpoints)

    else:
        print(f"Error: Invalid path type: {input_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
