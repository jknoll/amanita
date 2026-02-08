#!/usr/bin/env python3
"""
Regenerate specimen visualization images with microscopy filtering.

This script regenerates the well_recognized_specimens.png and poorly_recognized_specimens.png
images, filtering out spore/microscopy images that don't represent typical mushroom photographs.

Usage:
    python regenerate_specimen_images.py --results-pkl validation_results/validation_results.pkl

Or with a validation CSV that has captions:
    python regenerate_specimen_images.py --val-csv path/to/FungiTastic-Val.csv --image-root path/to/images
"""

import os
import sys
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def is_microscopy_image(filepath, caption=None):
    """
    Detect if an image is likely a microscopy/spore image rather than a macroscopic photo.

    Uses multiple heuristics:
    1. Caption-based detection (if captions available)
    2. Filename patterns
    3. Image analysis (aspect ratio, color distribution)

    Returns True if image should be excluded.
    """
    # Check caption for microscopy keywords
    if caption and isinstance(caption, str):
        caption_lower = caption.lower()
        microscopy_keywords = [
            'spore', 'microscop', 'magnif', 'slide', 'cell', 'hypha', 'hyphae',
            'basidi', 'ascus', 'asci', 'cystid', 'gill section', 'cross section',
            'μm', 'micron', '400x', '1000x', 'oil immersion', 'microscope',
            'spore print', 'circular spore', 'elliptical spore'
        ]
        for keyword in microscopy_keywords:
            if keyword in caption_lower:
                return True

    # Check filename patterns (some datasets use prefixes for microscopy)
    if filepath:
        filename = os.path.basename(filepath).lower()
        micro_patterns = ['micro', 'spore', 'slide', 'section', 'scope']
        for pattern in micro_patterns:
            if pattern in filename:
                return True

    # Optional: Image-based analysis
    if filepath and os.path.exists(filepath):
        try:
            img = Image.open(filepath)
            img_array = np.array(img)

            # Check if image has very high white/cream background coverage
            # (typical of microscope slides)
            if len(img_array.shape) == 3:
                # Calculate percentage of near-white pixels
                white_threshold = 240
                near_white = np.all(img_array > white_threshold, axis=2)
                white_ratio = near_white.sum() / (img_array.shape[0] * img_array.shape[1])

                # If more than 60% is near-white, likely a microscopy image
                if white_ratio > 0.6:
                    return True

        except Exception:
            pass

    return False


def filter_non_microscopy_indices(indices, filepaths, df=None, verbose=True):
    """
    Filter indices to exclude microscopy images.

    Args:
        indices: Array of indices to filter
        filepaths: List of all filepaths (indexed by results indices)
        df: Optional DataFrame with captions column
        verbose: Print filtering details

    Returns:
        Filtered array of indices
    """
    filtered = []
    removed_count = 0

    for idx in indices:
        filepath = filepaths[idx] if idx < len(filepaths) else None

        # Try to get caption if available
        caption = None
        if df is not None and 'captions' in df.columns and filepath:
            try:
                filename = os.path.basename(filepath)
                matching = df[df['filename'] == filename]
                if len(matching) > 0:
                    caption = matching.iloc[0]['captions']
            except Exception:
                pass

        if is_microscopy_image(filepath, caption):
            removed_count += 1
            if verbose:
                print(f"  Filtering: {os.path.basename(filepath) if filepath else 'unknown'}")
        else:
            filtered.append(idx)

    if verbose:
        print(f"Filtered {removed_count} microscopy images from {len(indices)} total")

    return np.array(filtered)


def display_specimens(indices, filepaths, predictions, labels, confidences,
                      id_to_name, title, n_samples=12, n_cols=4, output_path=None):
    """
    Display a grid of specimen images with predictions.

    Args:
        indices: Array of indices to display
        filepaths: List of all filepaths
        predictions: Dict of predictions per rank
        labels: Dict of labels per rank
        confidences: Dict of confidences per rank
        id_to_name: Mapping from ID to name for species
        title: Plot title
        n_samples: Number of samples to show
        n_cols: Number of columns in grid
        output_path: Optional path to save figure

    Returns:
        matplotlib figure
    """
    if len(indices) == 0:
        print(f"No specimens found for: {title}")
        return None

    # Sample indices if too many
    np.random.seed(42)  # For reproducibility
    if len(indices) > n_samples:
        sample_indices = np.random.choice(indices, n_samples, replace=False)
    else:
        sample_indices = indices[:n_samples]

    n_rows = (len(sample_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.array(axes).flatten()

    for ax_idx, sample_idx in enumerate(sample_indices):
        ax = axes[ax_idx]

        # Load and display image
        filepath = filepaths[sample_idx]
        try:
            img = Image.open(filepath)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center', fontsize=12)
            print(f"Warning: Could not load {filepath}: {e}")

        # Get prediction info
        pred_id = predictions['species'][sample_idx]
        true_id = labels['species'][sample_idx]
        conf = confidences['species'][sample_idx]

        pred_name = id_to_name['species'].get(str(pred_id), f"ID_{pred_id}")
        true_name = id_to_name['species'].get(str(true_id), f"ID_{true_id}")

        is_correct = pred_id == true_id
        color = 'green' if is_correct else 'red'

        ax.set_title(f"True: {true_name[:25]}\nPred: {pred_name[:25]}\nConf: {conf:.3f}",
                     fontsize=8, color=color)
        ax.axis('off')

    # Hide empty axes
    for ax_idx in range(len(sample_indices), len(axes)):
        axes[ax_idx].axis('off')

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def regenerate_from_pickle(pkl_path, output_dir, val_csv_path=None):
    """
    Regenerate specimen images from cached pickle results.

    Args:
        pkl_path: Path to validation_results.pkl
        output_dir: Output directory for images
        val_csv_path: Optional path to validation CSV with captions
    """
    print(f"Loading results from: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    results = data['results']
    taxonomic_mappings = data['taxonomic_mappings']
    config = data['config']

    id_to_name = taxonomic_mappings['id_to_name']

    # Load validation CSV if provided (for captions)
    val_df = None
    if val_csv_path and os.path.exists(val_csv_path):
        print(f"Loading validation CSV: {val_csv_path}")
        val_df = pd.read_csv(val_csv_path)

    # Get indices
    rank = 'species'
    preds = results['predictions'][rank]
    lbls = results['labels'][rank]
    confs = results['confidences'][rank]

    correct_mask = preds == lbls
    high_conf_threshold = config.get('high_confidence_threshold', 0.9)
    low_conf_threshold = config.get('low_confidence_threshold', 0.5)

    well_recognized_mask = correct_mask & (confs >= high_conf_threshold)
    poorly_recognized_mask = ~correct_mask

    well_recognized_indices = np.where(well_recognized_mask)[0]
    poorly_recognized_indices = np.where(poorly_recognized_mask)[0]

    print(f"\nBefore filtering:")
    print(f"  Well-recognized: {len(well_recognized_indices)}")
    print(f"  Poorly-recognized: {len(poorly_recognized_indices)}")

    # Filter microscopy images
    print("\nFiltering microscopy images...")
    well_recognized_filtered = filter_non_microscopy_indices(
        well_recognized_indices, results['filepaths'], val_df
    )
    poorly_recognized_filtered = filter_non_microscopy_indices(
        poorly_recognized_indices, results['filepaths'], val_df
    )

    print(f"\nAfter filtering:")
    print(f"  Well-recognized: {len(well_recognized_filtered)}")
    print(f"  Poorly-recognized: {len(poorly_recognized_filtered)}")

    # Generate images
    os.makedirs(output_dir, exist_ok=True)

    print("\nGenerating well-recognized specimens image...")
    display_specimens(
        well_recognized_filtered,
        results['filepaths'],
        results['predictions'],
        results['labels'],
        results['confidences'],
        id_to_name,
        'Well-Recognized Specimens (High Confidence Correct)',
        output_path=os.path.join(output_dir, 'well_recognized_specimens.png')
    )

    print("\nGenerating poorly-recognized specimens image...")
    display_specimens(
        poorly_recognized_filtered,
        results['filepaths'],
        results['predictions'],
        results['labels'],
        results['confidences'],
        id_to_name,
        'Poorly-Recognized Specimens (Incorrect Predictions)',
        output_path=os.path.join(output_dir, 'poorly_recognized_specimens.png')
    )

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description='Regenerate specimen images with microscopy filtering')
    parser.add_argument('--results-pkl', type=str,
                        default='validation_results/validation_results.pkl',
                        help='Path to validation results pickle file')
    parser.add_argument('--val-csv', type=str,
                        help='Path to validation CSV with captions (optional)')
    parser.add_argument('--output-dir', type=str,
                        default='validation_results',
                        help='Output directory for regenerated images')
    args = parser.parse_args()

    if os.path.exists(args.results_pkl):
        regenerate_from_pickle(args.results_pkl, args.output_dir, args.val_csv)
    else:
        print(f"Error: Results file not found: {args.results_pkl}")
        print("\nTo regenerate specimen images, you need either:")
        print("  1. The validation_results.pkl file from a previous validation run")
        print("  2. Run the full validation_notebook.ipynb with the dataset and model")
        sys.exit(1)


if __name__ == '__main__':
    main()
