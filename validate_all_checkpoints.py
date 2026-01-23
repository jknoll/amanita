#!/usr/bin/env python3
"""
Validate all checkpoints and generate comprehensive performance report.

This script:
1. Enumerates all checkpoint files across experiments
2. Validates each checkpoint on the FungiTastic validation set
3. Collects metrics for all taxonomic ranks
4. Generates performance plots
5. Creates an HTML report identifying best checkpoint and neighbors
"""

import os
import re
import json
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local imports
from models.beit_multitask import BEiTMultiTask, create_beit_multitask
from dataset_multitask import FungiTasticMultiTask

# Configuration
CONFIG = {
    # Paths
    'artifact_root': '/media/j/Extra FAT/Amanita-Validation',
    'val_csv_path': '/media/j/Extra FAT/FungiTastic/dataset/FungiTastic/metadata/FungiTastic/FungiTastic-ClosedSet-Val.csv',
    'image_root': '/media/j/Extra FAT/FungiTastic/dataset/FungiTastic/FungiTastic/',
    'taxonomic_mappings_path': '/home/j/Documents/git/amanita/taxonomic_mappings.json',
    'output_dir': '/home/j/Documents/git/amanita/checkpoint_validation_results',

    # Model settings
    'image_size': 224,
    'num_classes_dict': {
        'phylum': 7,
        'class': 28,
        'order': 95,
        'family': 308,
        'genus': 918,
        'species': 2786
    },

    # Dataloader settings
    'batch_size': 64,
    'num_workers': 8,
}

TAXONOMIC_RANKS = ['phylum', 'class', 'order', 'family', 'genus', 'species']


def find_all_checkpoints(artifact_root):
    """Find all checkpoint files in the artifact directory."""
    checkpoints = []

    artifact_path = Path(artifact_root)
    for exp_dir in artifact_path.iterdir():
        if not exp_dir.is_dir() or exp_dir.name.startswith('.'):
            continue

        # Extract experiment name
        exp_name = exp_dir.name

        # Look for checkpoints directory
        checkpoint_dir = exp_dir / 'checkpoints'
        if not checkpoint_dir.exists():
            continue

        # Find all checkpoint subdirectories
        for ckpt_subdir in checkpoint_dir.iterdir():
            if not ckpt_subdir.is_dir():
                continue

            # Extract checkpoint number from directory name
            match = re.search(r'checkpoint_(\d+)', ckpt_subdir.name)
            if not match:
                continue

            ckpt_num = int(match.group(1))

            # Find .pt files (prefer best_model.pt, fall back to test_checkpoint.pt)
            pt_files = list(ckpt_subdir.glob('*.pt'))
            pt_files = [f for f in pt_files if not f.name.startswith('._')]

            if not pt_files:
                continue

            # Prefer best_model.pt
            pt_file = None
            for f in pt_files:
                if f.name == 'best_model.pt':
                    pt_file = f
                    break
            if pt_file is None:
                pt_file = pt_files[0]

            checkpoints.append({
                'experiment': exp_name,
                'checkpoint_num': ckpt_num,
                'path': str(pt_file),
                'filename': pt_file.name,
            })

    # Sort by experiment name, then checkpoint number
    checkpoints.sort(key=lambda x: (x['experiment'], x['checkpoint_num']))

    return checkpoints


def load_model(checkpoint_path, num_classes_dict, device):
    """Load a model from checkpoint."""
    # Create model architecture
    model = create_beit_multitask(
        pretrained=False,
        num_classes_dict=num_classes_dict
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get state dict
    state_dict = checkpoint['model_state_dict']

    # Handle DDP checkpoint format (strip 'module.' prefix if present)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Load weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Extract metadata from checkpoint
    metadata = {
        'epoch': checkpoint.get('epoch', None),
        'best_val_acc': checkpoint.get('best_val_acc', None),
    }

    return model, metadata


def validate_checkpoint(model, val_loader, device):
    """Run validation on a single checkpoint."""
    predictions = {rank: [] for rank in TAXONOMIC_RANKS}
    labels = {rank: [] for rank in TAXONOMIC_RANKS}
    confidences = {rank: [] for rank in TAXONOMIC_RANKS}
    top5_predictions = {rank: [] for rank in TAXONOMIC_RANKS}

    with torch.no_grad():
        for images, batch_labels, _ in tqdm(val_loader, desc="Validating", leave=False):
            images = images.to(device)
            outputs = model(images)

            for rank in TAXONOMIC_RANKS:
                logits = outputs[rank]
                probs = F.softmax(logits, dim=1)

                # Top-1
                conf, preds = probs.max(dim=1)
                predictions[rank].extend(preds.cpu().numpy())
                confidences[rank].extend(conf.cpu().numpy())
                labels[rank].extend(batch_labels[rank].numpy())

                # Top-5
                top5_conf, top5_pred = probs.topk(5, dim=1)
                top5_predictions[rank].extend(top5_pred.cpu().numpy())

    # Convert to numpy
    for rank in TAXONOMIC_RANKS:
        predictions[rank] = np.array(predictions[rank])
        labels[rank] = np.array(labels[rank])
        confidences[rank] = np.array(confidences[rank])
        top5_predictions[rank] = np.array(top5_predictions[rank])

    # Calculate metrics
    metrics = {}
    for rank in TAXONOMIC_RANKS:
        preds = predictions[rank]
        lbls = labels[rank]
        confs = confidences[rank]
        top5_preds = top5_predictions[rank]

        metrics[rank] = {
            'accuracy': accuracy_score(lbls, preds),
            'precision': precision_score(lbls, preds, average='macro', zero_division=0),
            'recall': recall_score(lbls, preds, average='macro', zero_division=0),
            'f1': f1_score(lbls, preds, average='macro', zero_division=0),
        }

        # Top-5 accuracy
        top5_correct = np.any(top5_preds == lbls.reshape(-1, 1), axis=1)
        metrics[rank]['top5_accuracy'] = top5_correct.mean()

        # Confidence stats
        correct_mask = preds == lbls
        metrics[rank]['avg_confidence'] = confs.mean()
        metrics[rank]['avg_confidence_correct'] = confs[correct_mask].mean() if correct_mask.sum() > 0 else 0
        metrics[rank]['avg_confidence_incorrect'] = confs[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0

    # Hierarchical accuracy
    all_correct = np.ones(len(labels['species']), dtype=bool)
    for rank in TAXONOMIC_RANKS:
        all_correct &= (predictions[rank] == labels[rank])
    metrics['hierarchical_accuracy'] = all_correct.mean()

    # Average accuracy across ranks
    metrics['avg_accuracy'] = np.mean([metrics[r]['accuracy'] for r in TAXONOMIC_RANKS])

    return metrics


def create_dataloader(config, taxonomic_mappings):
    """Create validation dataloader."""
    val_df = pd.read_csv(config['val_csv_path'])
    print(f"Loaded {len(val_df)} validation samples")

    val_transform = A.Compose([
        A.Resize(config['image_size'], config['image_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_dataset = FungiTasticMultiTask(
        df=val_df,
        transform=val_transform,
        taxonomic_mappings=taxonomic_mappings,
        image_root=config['image_root'],
        split='val'
    )

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        labels = {}
        for rank in TAXONOMIC_RANKS:
            labels[rank] = torch.tensor([item[1][rank] for item in batch], dtype=torch.long)
        filepaths = [item[2] for item in batch]
        return images, labels, filepaths

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn
    )

    return val_loader


def generate_plots(all_results, output_dir):
    """Generate performance plots across checkpoints."""

    # Group results by experiment
    experiments = defaultdict(list)
    for result in all_results:
        experiments[result['experiment']].append(result)

    # Sort each experiment by checkpoint number
    for exp_name in experiments:
        experiments[exp_name].sort(key=lambda x: x['checkpoint_num'])

    # Plot 1: Accuracy across checkpoints for each experiment
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for idx, rank in enumerate(TAXONOMIC_RANKS):
        ax = axes[idx]

        for exp_idx, (exp_name, exp_results) in enumerate(sorted(experiments.items())):
            if len(exp_results) < 2:
                continue

            ckpt_nums = [r['checkpoint_num'] for r in exp_results]
            accuracies = [r['metrics'][rank]['accuracy'] for r in exp_results]

            ax.plot(ckpt_nums, accuracies, 'o-',
                   label=exp_name[:30], color=colors[exp_idx], alpha=0.7)

        ax.set_xlabel('Checkpoint Number')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{rank.capitalize()} Accuracy')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    # Add legend to last subplot
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
               ncol=min(4, len(experiments)), fontsize=8)

    plt.suptitle('Accuracy Across Checkpoints by Taxonomic Rank', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_by_rank.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Average accuracy and species accuracy comparison
    fig, ax = plt.subplots(figsize=(14, 8))

    for exp_idx, (exp_name, exp_results) in enumerate(sorted(experiments.items())):
        if len(exp_results) < 2:
            continue

        ckpt_nums = [r['checkpoint_num'] for r in exp_results]
        avg_acc = [r['metrics']['avg_accuracy'] for r in exp_results]
        species_acc = [r['metrics']['species']['accuracy'] for r in exp_results]

        ax.plot(ckpt_nums, species_acc, 'o-',
               label=f'{exp_name[:25]} (species)', color=colors[exp_idx])
        ax.plot(ckpt_nums, avg_acc, 's--',
               label=f'{exp_name[:25]} (avg)', color=colors[exp_idx], alpha=0.5)

    ax.set_xlabel('Checkpoint Number')
    ax.set_ylabel('Accuracy')
    ax.set_title('Species vs Average Accuracy Across Checkpoints')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'species_vs_avg_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Best checkpoint per experiment
    fig, ax = plt.subplots(figsize=(14, 8))

    best_per_exp = []
    for exp_name, exp_results in sorted(experiments.items()):
        best = max(exp_results, key=lambda x: x['metrics']['species']['accuracy'])
        best_per_exp.append({
            'experiment': exp_name,
            'checkpoint': best['checkpoint_num'],
            'species_acc': best['metrics']['species']['accuracy'],
            'avg_acc': best['metrics']['avg_accuracy'],
            'hierarchical_acc': best['metrics']['hierarchical_accuracy'],
        })

    best_df = pd.DataFrame(best_per_exp)
    best_df = best_df.sort_values('species_acc', ascending=True)

    y_pos = range(len(best_df))

    ax.barh(y_pos, best_df['species_acc'], height=0.4, label='Species', alpha=0.8)
    ax.barh([y + 0.4 for y in y_pos], best_df['avg_acc'], height=0.4, label='Average', alpha=0.6)

    ax.set_yticks([y + 0.2 for y in y_pos])
    ax.set_yticklabels([f"{exp[:30]} (ckpt {ckpt})"
                        for exp, ckpt in zip(best_df['experiment'], best_df['checkpoint'])],
                       fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_title('Best Checkpoint Performance per Experiment')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'best_per_experiment.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Heatmap of all checkpoints
    # Create a matrix of species accuracy
    all_exp_names = sorted(experiments.keys())
    max_ckpt = max(r['checkpoint_num'] for r in all_results) + 1

    heatmap_data = np.full((len(all_exp_names), max_ckpt), np.nan)

    for result in all_results:
        exp_idx = all_exp_names.index(result['experiment'])
        ckpt_num = result['checkpoint_num']
        heatmap_data[exp_idx, ckpt_num] = result['metrics']['species']['accuracy']

    fig, ax = plt.subplots(figsize=(16, 10))

    # Mask NaN values
    masked_data = np.ma.masked_invalid(heatmap_data)

    im = ax.imshow(masked_data, aspect='auto', cmap='viridis', vmin=0, vmax=1)

    ax.set_yticks(range(len(all_exp_names)))
    ax.set_yticklabels([exp[:35] for exp in all_exp_names], fontsize=8)
    ax.set_xlabel('Checkpoint Number')
    ax.set_ylabel('Experiment')
    ax.set_title('Species Accuracy Heatmap Across All Checkpoints')

    plt.colorbar(im, ax=ax, label='Species Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plots saved to {output_dir}")


def generate_html_report(all_results, output_dir):
    """Generate comprehensive HTML report."""

    # Find best checkpoint overall
    best_result = max(all_results, key=lambda x: x['metrics']['species']['accuracy'])

    # Find prev and next checkpoints from the same experiment
    same_exp_results = [r for r in all_results if r['experiment'] == best_result['experiment']]
    same_exp_results.sort(key=lambda x: x['checkpoint_num'])

    best_idx = next(i for i, r in enumerate(same_exp_results)
                    if r['checkpoint_num'] == best_result['checkpoint_num'])

    prev_ckpt = same_exp_results[best_idx - 1] if best_idx > 0 else None
    next_ckpt = same_exp_results[best_idx + 1] if best_idx < len(same_exp_results) - 1 else None

    # Identify missing checkpoints (gaps in the sequence)
    existing_ckpts = set(r['checkpoint_num'] for r in same_exp_results)
    max_ckpt = max(existing_ckpts)
    missing_before = set(range(best_result['checkpoint_num'])) - existing_ckpts
    missing_after = set(range(best_result['checkpoint_num'] + 1, max_ckpt + 1)) - existing_ckpts

    # Calculate statistics
    total_checkpoints = len(all_results)
    experiments = set(r['experiment'] for r in all_results)

    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Checkpoint Validation Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; background-color: #f8f9fa; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #e1e1e1; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        tr:hover {{ background-color: #f1f1f1; }}
        .best {{ background-color: #d4edda !important; border-left: 4px solid #28a745; }}
        .prev {{ background-color: #fff3cd !important; border-left: 4px solid #ffc107; }}
        .next {{ background-color: #cce5ff !important; border-left: 4px solid #007bff; }}
        .metric-good {{ color: #27ae60; font-weight: bold; }}
        .metric-medium {{ color: #f39c12; font-weight: bold; }}
        .metric-poor {{ color: #e74c3c; font-weight: bold; }}
        .summary-box {{ background: white; padding: 25px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 20px 0; }}
        .key-finding {{ background-color: #e8f6f3; border-left: 4px solid #1abc9c; padding: 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
        .bracket-search {{ background-color: #e3f2fd; border-left: 4px solid #2196f3; padding: 20px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
        .warning {{ background-color: #fdf2e9; border-left: 4px solid #e67e22; padding: 15px; margin: 20px 0; }}
        .image-container {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
        .stat-value {{ font-size: 2em; font-weight: bold; color: #3498db; }}
        .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
        code {{ background-color: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-family: 'Consolas', monospace; }}
    </style>
</head>
<body>
    <h1>Checkpoint Validation Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="stat-grid">
        <div class="stat-card">
            <div class="stat-value">{total_checkpoints}</div>
            <div class="stat-label">Total Checkpoints</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(experiments)}</div>
            <div class="stat-label">Experiments</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best_result['metrics']['species']['accuracy']:.1%}</div>
            <div class="stat-label">Best Species Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{best_result['metrics']['avg_accuracy']:.1%}</div>
            <div class="stat-label">Best Avg Accuracy</div>
        </div>
    </div>

    <div class="key-finding">
        <h3>Best Checkpoint Identified</h3>
        <p><strong>Experiment:</strong> <code>{best_result['experiment']}</code></p>
        <p><strong>Checkpoint:</strong> {best_result['checkpoint_num']}</p>
        <p><strong>Path:</strong> <code>{best_result['path']}</code></p>
        <p><strong>Species Accuracy:</strong> <span class="metric-good">{best_result['metrics']['species']['accuracy']:.2%}</span></p>
        <p><strong>Hierarchical Accuracy:</strong> {best_result['metrics']['hierarchical_accuracy']:.2%}</p>
    </div>

    <div class="bracket-search">
        <h3>Bracket Search Results</h3>
        <p>For optimal checkpoint selection, consider syncing these additional checkpoints from the ISC cluster:</p>
        <table>
            <tr>
                <th>Position</th>
                <th>Checkpoint</th>
                <th>Species Accuracy</th>
                <th>Status</th>
            </tr>
            <tr class="prev">
                <td>Previous</td>
                <td>{prev_ckpt['checkpoint_num'] if prev_ckpt else 'N/A'}</td>
                <td>{f"{prev_ckpt['metrics']['species']['accuracy']:.2%}" if prev_ckpt else 'N/A'}</td>
                <td>{'Available locally' if prev_ckpt else 'Not available'}</td>
            </tr>
            <tr class="best">
                <td><strong>BEST</strong></td>
                <td><strong>{best_result['checkpoint_num']}</strong></td>
                <td><strong>{best_result['metrics']['species']['accuracy']:.2%}</strong></td>
                <td>Available locally</td>
            </tr>
            <tr class="next">
                <td>Next</td>
                <td>{next_ckpt['checkpoint_num'] if next_ckpt else 'N/A'}</td>
                <td>{f"{next_ckpt['metrics']['species']['accuracy']:.2%}" if next_ckpt else 'N/A'}</td>
                <td>{'Available locally' if next_ckpt else 'Not available'}</td>
            </tr>
        </table>
"""

    # Add missing checkpoints info
    if missing_before or missing_after:
        nearby_missing = sorted([c for c in missing_before if c >= best_result['checkpoint_num'] - 5] +
                               [c for c in missing_after if c <= best_result['checkpoint_num'] + 5])
        if nearby_missing:
            html += f"""
        <p><strong>Missing checkpoints near best (consider syncing):</strong></p>
        <ul>
"""
            for ckpt in nearby_missing:
                html += f"            <li>Checkpoint {ckpt}</li>\n"
            html += """        </ul>
"""

    html += """    </div>

    <h2>Performance Visualizations</h2>

    <h3>Accuracy by Taxonomic Rank</h3>
    <div class="image-container">
        <img src="accuracy_by_rank.png" alt="Accuracy by Rank">
    </div>

    <h3>Species vs Average Accuracy</h3>
    <div class="image-container">
        <img src="species_vs_avg_accuracy.png" alt="Species vs Average Accuracy">
    </div>

    <h3>Best Checkpoint per Experiment</h3>
    <div class="image-container">
        <img src="best_per_experiment.png" alt="Best per Experiment">
    </div>

    <h3>Accuracy Heatmap</h3>
    <div class="image-container">
        <img src="accuracy_heatmap.png" alt="Accuracy Heatmap">
    </div>

    <h2>All Checkpoint Results</h2>
    <table>
        <tr>
            <th>Experiment</th>
            <th>Checkpoint</th>
            <th>Species Acc</th>
            <th>Top-5 Species</th>
            <th>Genus Acc</th>
            <th>Family Acc</th>
            <th>Avg Acc</th>
            <th>Hierarchical</th>
        </tr>
"""

    # Sort results by species accuracy descending
    sorted_results = sorted(all_results, key=lambda x: x['metrics']['species']['accuracy'], reverse=True)

    for result in sorted_results:
        m = result['metrics']
        row_class = ""
        if result == best_result:
            row_class = "best"
        elif result == prev_ckpt:
            row_class = "prev"
        elif result == next_ckpt:
            row_class = "next"

        species_acc = m['species']['accuracy']
        acc_class = 'metric-good' if species_acc > 0.5 else 'metric-medium' if species_acc > 0.3 else 'metric-poor'

        html += f"""        <tr class="{row_class}">
            <td>{result['experiment'][:35]}</td>
            <td>{result['checkpoint_num']}</td>
            <td class="{acc_class}">{species_acc:.2%}</td>
            <td>{m['species']['top5_accuracy']:.2%}</td>
            <td>{m['genus']['accuracy']:.2%}</td>
            <td>{m['family']['accuracy']:.2%}</td>
            <td>{m['avg_accuracy']:.2%}</td>
            <td>{m['hierarchical_accuracy']:.2%}</td>
        </tr>
"""

    html += """    </table>

    <h2>Detailed Metrics for Best Checkpoint</h2>
    <table>
        <tr>
            <th>Rank</th>
            <th>Top-1 Accuracy</th>
            <th>Top-5 Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1 (Macro)</th>
            <th>Avg Confidence</th>
        </tr>
"""

    for rank in TAXONOMIC_RANKS:
        m = best_result['metrics'][rank]
        acc_class = 'metric-good' if m['accuracy'] > 0.8 else 'metric-medium' if m['accuracy'] > 0.5 else 'metric-poor'
        html += f"""        <tr>
            <td><strong>{rank.capitalize()}</strong></td>
            <td class="{acc_class}">{m['accuracy']:.2%}</td>
            <td>{m['top5_accuracy']:.2%}</td>
            <td>{m['precision']:.2%}</td>
            <td>{m['recall']:.2%}</td>
            <td>{m['f1']:.2%}</td>
            <td>{m['avg_confidence']:.3f}</td>
        </tr>
"""

    html += f"""    </table>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
        <p>Generated by validate_all_checkpoints.py</p>
        <p>Artifact root: {CONFIG['artifact_root']}</p>
    </footer>
</body>
</html>
"""

    report_path = os.path.join(output_dir, 'checkpoint_report.html')
    with open(report_path, 'w') as f:
        f.write(html)

    print(f"HTML report saved to: {report_path}")
    return report_path


def main():
    """Main validation pipeline."""
    print("="*80)
    print("CHECKPOINT VALIDATION PIPELINE")
    print("="*80)

    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)

    # Find all checkpoints
    print("\n1. Finding checkpoints...")
    checkpoints = find_all_checkpoints(CONFIG['artifact_root'])
    print(f"   Found {len(checkpoints)} checkpoints across experiments")

    if not checkpoints:
        print("ERROR: No checkpoints found!")
        return

    # Load taxonomic mappings
    print("\n2. Loading taxonomic mappings...")
    with open(CONFIG['taxonomic_mappings_path'], 'r') as f:
        taxonomic_mappings = json.load(f)

    # Create dataloader (once, reused for all checkpoints)
    print("\n3. Creating validation dataloader...")
    val_loader = create_dataloader(CONFIG, taxonomic_mappings)
    print(f"   {len(val_loader)} batches, batch size {CONFIG['batch_size']}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n4. Using device: {device}")

    # Validate each checkpoint
    print("\n5. Validating checkpoints...")
    all_results = []

    for i, ckpt_info in enumerate(checkpoints):
        print(f"\n   [{i+1}/{len(checkpoints)}] {ckpt_info['experiment']} checkpoint {ckpt_info['checkpoint_num']}")

        try:
            # Load model
            model, metadata = load_model(ckpt_info['path'], CONFIG['num_classes_dict'], device)

            # Validate
            metrics = validate_checkpoint(model, val_loader, device)

            # Store results
            result = {
                **ckpt_info,
                'metrics': metrics,
                'metadata': metadata,
            }
            all_results.append(result)

            print(f"      Species: {metrics['species']['accuracy']:.2%}, "
                  f"Avg: {metrics['avg_accuracy']:.2%}, "
                  f"Hierarchical: {metrics['hierarchical_accuracy']:.2%}")

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"      ERROR: {e}")
            continue

    if not all_results:
        print("\nERROR: No checkpoints were successfully validated!")
        return

    # Save raw results
    print("\n6. Saving results...")
    results_path = os.path.join(CONFIG['output_dir'], 'validation_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"   Results saved to: {results_path}")

    # Generate plots
    print("\n7. Generating plots...")
    generate_plots(all_results, CONFIG['output_dir'])

    # Generate HTML report
    print("\n8. Generating HTML report...")
    report_path = generate_html_report(all_results, CONFIG['output_dir'])

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

    best_result = max(all_results, key=lambda x: x['metrics']['species']['accuracy'])
    print(f"\nBest checkpoint:")
    print(f"  Experiment: {best_result['experiment']}")
    print(f"  Checkpoint: {best_result['checkpoint_num']}")
    print(f"  Species accuracy: {best_result['metrics']['species']['accuracy']:.2%}")
    print(f"  Path: {best_result['path']}")

    print(f"\nReport: {report_path}")
    print(f"Results: {results_path}")


if __name__ == '__main__':
    main()
