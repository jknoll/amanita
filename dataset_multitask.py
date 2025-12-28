"""
Multi-task dataset class for hierarchical taxonomic classification.

Extends the standard ImageDataset to return labels for all 6 taxonomic ranks.
"""

import os
import json
import pandas as pd
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


class FungiTasticMultiTask(Dataset):
    """
    Multi-task dataset for FungiTastic taxonomic classification.

    Returns labels for all 6 taxonomic ranks: phylum, class, order, family, genus, species.
    """

    def __init__(self, df, transform, taxonomic_mappings, image_root=None, split='train'):
        """
        Initialize multi-task dataset.

        Args:
            df: DataFrame with image paths and taxonomic information
            transform: Albumentations transform pipeline
            taxonomic_mappings: Dictionary with name_to_id mappings for each rank
            image_root: Root directory for images (only needed if df doesn't have 'image_path' column)
            split: Dataset split ('train', 'val', or 'test') - used to construct image paths
        """
        self.df = df
        self.transform = transform
        self.name_to_id = taxonomic_mappings['name_to_id']
        self.image_root = image_root
        self.split = split

        # Check if we need to construct image paths
        self.has_image_path = 'image_path' in df.columns
        if not self.has_image_path:
            if image_root is None:
                raise ValueError(
                    "DataFrame does not have 'image_path' column and no image_root was provided. "
                    "Please provide image_root parameter."
                )
            print(f"Dataset will construct image paths from: {image_root}/{split}/300p/")

        # Taxonomic ranks in order
        self.taxonomic_ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

        print(f"Created multi-task dataset with {len(df)} samples")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get item at index.

        Returns:
            tuple: (image, labels_dict, filepath)
                - image: Transformed image tensor
                - labels_dict: Dictionary mapping rank names to label indices
                - filepath: Path to the image file
        """
        row = self.df.iloc[idx]

        # Get image path - either from column or construct from filename
        if self.has_image_path:
            image_path = row['image_path']
        else:
            # Construct path from image_root, split, and filename
            # FungiTastic structure: <image_root>/<split>/300p/<filename>
            filename = row['filename']
            image_path = os.path.join(self.image_root, self.split, '300p', filename)

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']

        # Create labels dictionary
        labels = {}
        for rank in self.taxonomic_ranks:
            taxonomic_name = row[rank]

            # Handle missing values (rare in training data, but possible)
            if pd.isna(taxonomic_name):
                # Use a special index for missing values (will need special handling in loss)
                labels[rank] = -1
            else:
                # Map taxonomic name to class ID
                labels[rank] = self.name_to_id[rank][taxonomic_name]

        return image, labels, image_path


def create_multitask_dataloaders(
    train_df,
    val_df,
    taxonomic_mappings,
    batch_size=16,
    num_workers=4,
    image_size=224,
    image_root=None
):
    """
    Create training and validation dataloaders for multi-task learning.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        taxonomic_mappings: Taxonomic name-to-ID mappings
        batch_size: Batch size
        num_workers: Number of data loading workers
        image_size: Input image size
        image_root: Root directory for images (only needed if df doesn't have 'image_path' column)

    Returns:
        tuple: (train_loader, val_loader)
    """
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
    train_dataset = FungiTasticMultiTask(train_df, train_transform, taxonomic_mappings, image_root, split='train')
    val_dataset = FungiTasticMultiTask(val_df, val_transform, taxonomic_mappings, image_root, split='val')

    # Custom collate function to handle dictionary labels
    def collate_fn(batch):
        """Collate batch with dictionary labels."""
        images = torch.stack([item[0] for item in batch])

        # Collect labels for each rank
        labels = {}
        for rank in train_dataset.taxonomic_ranks:
            labels[rank] = torch.tensor([item[1][rank] for item in batch], dtype=torch.long)

        filepaths = [item[2] for item in batch]

        return images, labels, filepaths

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader


# Add numpy import
import numpy as np


if __name__ == '__main__':
    """Test multi-task dataset loading."""
    print("Testing multi-task dataset...")
    print("=" * 60)

    # Load taxonomic mappings
    with open('taxonomic_mappings.json', 'r') as f:
        taxonomic_mappings = json.load(f)

    # Load minimal subset
    train_df = pd.read_csv('minimal_subsets/FungiTastic-Train-Minimal.csv')
    val_df = pd.read_csv('minimal_subsets/FungiTastic-Val-Minimal.csv')

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_multitask_dataloaders(
        train_df, val_df,
        taxonomic_mappings,
        batch_size=8,
        num_workers=2
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Test loading a batch
    print("\n" + "=" * 60)
    print("Testing batch loading...")
    print("=" * 60)

    for images, labels, filepaths in train_loader:
        print(f"\nBatch loaded successfully!")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Images min/max: {images.min():.3f} / {images.max():.3f}")

        print(f"\n  Label shapes:")
        for rank, label_tensor in labels.items():
            print(f"    {rank:8s}: {label_tensor.shape}, dtype={label_tensor.dtype}")

        print(f"\n  Sample labels (first item in batch):")
        id_to_name = taxonomic_mappings['id_to_name']
        for rank in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
            label_id = labels[rank][0].item()
            if label_id >= 0:
                name = id_to_name[rank][str(label_id)]
                print(f"    {rank:8s}: {label_id:4d} -> {name}")
            else:
                print(f"    {rank:8s}: MISSING")

        print(f"\n  Sample filepath: {filepaths[0]}")

        break  # Only test one batch

    print("\n" + "=" * 60)
    print("✓ Multi-task dataset test successful!")
    print("=" * 60)
