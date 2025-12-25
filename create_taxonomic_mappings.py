"""
Create taxonomic label mappings for multi-task classification.

Analyzes the FungiTastic dataset and creates mappings from taxonomic names
to class indices for all 6 taxonomic ranks: phylum, class, order, family, genus, species.
"""

import json
import pandas as pd
from pathlib import Path


def create_taxonomic_mappings(metadata_path, output_path='taxonomic_mappings.json'):
    """
    Create and save taxonomic label mappings.

    Args:
        metadata_path: Path to training metadata CSV
        output_path: Path to save mappings JSON file

    Returns:
        dict: Nested dictionary with mappings for each taxonomic rank
    """
    print(f"Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)

    print(f"Total samples: {len(df)}")

    # Define taxonomic ranks to process
    taxonomic_ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

    # Create mappings for each rank
    mappings = {}
    reverse_mappings = {}  # For easy lookup: id -> name

    print("\nCreating mappings for each taxonomic rank:")
    print("=" * 60)

    for rank in taxonomic_ranks:
        # Get unique values (sorted alphabetically for consistency)
        unique_values = sorted(df[rank].dropna().unique())
        n_unique = len(unique_values)

        # Create mapping: name -> id
        rank_mapping = {name: idx for idx, name in enumerate(unique_values)}

        # Create reverse mapping: id -> name
        rank_reverse = {idx: name for name, idx in rank_mapping.items()}

        mappings[rank] = rank_mapping
        reverse_mappings[rank] = rank_reverse

        print(f"{rank.capitalize():12s}: {n_unique:5d} unique values")

        # Show first 5 examples
        print(f"  Examples: {list(unique_values[:5])}")

    # Check for missing values
    print("\n" + "=" * 60)
    print("Checking for missing values:")
    print("=" * 60)

    for rank in taxonomic_ranks:
        n_missing = df[rank].isna().sum()
        if n_missing > 0:
            print(f"  WARNING: {rank} has {n_missing} missing values")
        else:
            print(f"  {rank:12s}: ✓ No missing values")

    # For species, we also need to handle the category_id mapping
    # Get unique species with their category_ids
    species_to_category = df[['species', 'category_id']].drop_duplicates()
    species_to_category_dict = dict(zip(species_to_category['species'], species_to_category['category_id']))

    # Add category_id mapping to species
    mappings['species_to_category_id'] = species_to_category_dict

    print("\n" + "=" * 60)
    print(f"Species to category_id mapping: {len(species_to_category_dict)} entries")

    # Create complete mapping structure
    complete_mappings = {
        'name_to_id': mappings,
        'id_to_name': reverse_mappings,
        'metadata': {
            'source': str(metadata_path),
            'total_samples': len(df),
            'num_classes': {rank: len(mappings[rank]) for rank in taxonomic_ranks}
        }
    }

    # Save to JSON
    output_path = Path(output_path)
    print(f"\nSaving mappings to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(complete_mappings, f, indent=2)

    print("✓ Mappings saved successfully!")

    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for rank in taxonomic_ranks:
        print(f"  {rank.capitalize():12s}: {len(mappings[rank]):5d} classes")

    return complete_mappings


def load_taxonomic_mappings(path='taxonomic_mappings.json'):
    """Load taxonomic mappings from JSON file."""
    with open(path, 'r') as f:
        mappings = json.load(f)
    return mappings


def verify_mappings(mappings, metadata_path):
    """Verify that mappings are correct by checking sample conversions."""
    print("\nVerifying mappings...")
    print("=" * 60)

    df = pd.read_csv(metadata_path)

    # Check a few random samples
    sample_indices = [0, 100, 1000, 5000]

    name_to_id = mappings['name_to_id']
    id_to_name = mappings['id_to_name']

    for idx in sample_indices:
        if idx >= len(df):
            continue

        row = df.iloc[idx]
        print(f"\nSample {idx}:")

        for rank in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
            name = row[rank]
            if pd.isna(name):
                print(f"  {rank:8s}: MISSING VALUE")
                continue

            mapped_id = name_to_id[rank][name]
            reverse_name = id_to_name[rank][mapped_id]

            status = "✓" if name == reverse_name else "✗"
            print(f"  {rank:8s}: {name} -> {mapped_id} -> {reverse_name} {status}")


if __name__ == '__main__':
    # Path to training metadata
    metadata_path = "/data/uds-fern-absorbed-dugong-251223/full_300px/metadata/FungiTastic/FungiTastic-Train.csv"

    # Create mappings
    mappings = create_taxonomic_mappings(metadata_path)

    # Verify mappings
    verify_mappings(mappings, metadata_path)

    print("\n" + "=" * 60)
    print("✓ Taxonomic mappings created and verified!")
    print("=" * 60)
