"""
Compute class weights for handling severe class imbalance in taxonomic ranks.

Uses inverse frequency weighting to give more importance to rare classes.
"""

import json
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def compute_class_weights_per_rank(train_df, taxonomic_mappings, save_path='class_weights.pt'):
    """
    Compute class weights for each taxonomic rank to handle imbalance.

    Uses sklearn's 'balanced' mode: weight = n_samples / (n_classes * class_count)

    Args:
        train_df: Training DataFrame with taxonomic columns
        taxonomic_mappings: Dictionary with name_to_id mappings
        save_path: Path to save the weights

    Returns:
        dict: Class weights for each rank as tensors
    """
    print("Computing class weights for each taxonomic rank...")
    print("=" * 70)

    class_weights = {}
    name_to_id = taxonomic_mappings['name_to_id']

    ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

    for rank in ranks:
        print(f"\n{rank.upper()}:")

        # Get labels (convert names to IDs, skip missing values)
        labels = []
        for name in train_df[rank]:
            if pd.notna(name) and name in name_to_id[rank]:
                labels.append(name_to_id[rank][name])

        labels = np.array(labels)

        # Get all classes for this rank
        n_classes = len(name_to_id[rank])
        classes = np.arange(n_classes)

        print(f"  Total samples: {len(labels)}")
        print(f"  Number of classes: {n_classes}")

        # Compute balanced class weights
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )

        # Convert to tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        class_weights[rank] = weights_tensor

        # Statistics
        print(f"  Weight statistics:")
        print(f"    Min:    {weights.min():.6f}")
        print(f"    Max:    {weights.max():.6f}")
        print(f"    Mean:   {weights.mean():.6f}")
        print(f"    Median: {np.median(weights):.6f}")
        print(f"    Ratio (max/min): {weights.max()/weights.min():.1f}x")

        # Show examples of highest/lowest weighted classes
        id_to_name = taxonomic_mappings['id_to_name'][rank]
        max_idx = int(weights.argmax())
        min_idx = int(weights.argmin())

        print(f"  Highest weight: {id_to_name[str(max_idx)]} (weight={weights[max_idx]:.4f})")
        print(f"  Lowest weight:  {id_to_name[str(min_idx)]} (weight={weights[min_idx]:.4f})")

    # Save weights
    print(f"\n{'=' * 70}")
    print(f"Saving class weights to: {save_path}")
    torch.save(class_weights, save_path)
    print("✓ Class weights saved successfully!")

    return class_weights


def load_class_weights(path='class_weights.pt'):
    """Load precomputed class weights."""
    return torch.load(path)


def analyze_class_imbalance(train_df, taxonomic_mappings):
    """
    Analyze class imbalance to help choose weighting strategy.
    """
    print("\nClass Imbalance Analysis:")
    print("=" * 70)

    ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

    for rank in ranks:
        counts = train_df[rank].value_counts()
        n_classes = len(counts)

        print(f"\n{rank.upper()}:")
        print(f"  Classes: {n_classes}")
        print(f"  Imbalance ratio: {counts.max() / counts.min():.1f}x")
        print(f"  Most common: {counts.index[0]} ({counts.iloc[0]:,} samples)")
        print(f"  Least common: {counts.index[-1]} ({counts.iloc[-1]:,} samples)")

        # Show distribution
        percentiles = [50, 75, 90, 95, 99]
        print(f"  Sample distribution (percentiles):")
        for p in percentiles:
            val = np.percentile(counts.values, p)
            print(f"    {p}th: {val:.0f} samples/class")


if __name__ == '__main__':
    # Load data
    print("Loading training data...")
    train_df = pd.read_csv('/data/uds-fern-absorbed-dugong-251223/full_300px/metadata/FungiTastic/FungiTastic-Train.csv')

    # Load mappings
    with open('taxonomic_mappings.json', 'r') as f:
        taxonomic_mappings = json.load(f)

    print(f"Training samples: {len(train_df):,}\n")

    # Analyze imbalance
    analyze_class_imbalance(train_df, taxonomic_mappings)

    # Compute weights
    print("\n" + "=" * 70)
    class_weights = compute_class_weights_per_rank(train_df, taxonomic_mappings)

    print("\n" + "=" * 70)
    print("✓ Class weight computation complete!")
    print("=" * 70)

    # Test loading
    print("\nTesting weight loading...")
    loaded_weights = load_class_weights()
    print(f"✓ Successfully loaded weights for {len(loaded_weights)} ranks")

    for rank, weights in loaded_weights.items():
        print(f"  {rank:8s}: shape={weights.shape}, dtype={weights.dtype}")
