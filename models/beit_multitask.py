"""
Multi-task BEiT model for hierarchical taxonomic classification.

Extends the pretrained BEiT model to predict all 6 taxonomic ranks:
Phylum, Class, Order, Family, Genus, Species
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Optional


class BEiTMultiTask(nn.Module):
    """
    Multi-task BEiT model for hierarchical fungi classification.

    Replaces the single classification head with 6 separate heads,
    one for each taxonomic rank. All heads share the same backbone features.
    """

    def __init__(self, base_model, num_classes_dict: Dict[str, int]):
        """
        Initialize multi-task BEiT model.

        Args:
            base_model: Pretrained BEiT model (with original single head)
            num_classes_dict: Dictionary mapping rank names to number of classes
                Example: {
                    'phylum': 7,
                    'class': 28,
                    'order': 95,
                    'family': 308,
                    'genus': 918,
                    'species': 2786
                }
        """
        super().__init__()

        # Store the backbone (everything except the head)
        self.backbone = base_model

        # Get feature dimension from the original head
        self.feature_dim = base_model.head.in_features
        print(f"Feature dimension: {self.feature_dim}")

        # Remove the original single classification head
        # Replace with identity so forward_features still works
        self.backbone.head = nn.Identity()

        # Create separate classification heads for each taxonomic rank
        self.heads = nn.ModuleDict({
            'phylum': nn.Linear(self.feature_dim, num_classes_dict['phylum']),
            'class': nn.Linear(self.feature_dim, num_classes_dict['class']),
            'order': nn.Linear(self.feature_dim, num_classes_dict['order']),
            'family': nn.Linear(self.feature_dim, num_classes_dict['family']),
            'genus': nn.Linear(self.feature_dim, num_classes_dict['genus']),
            'species': nn.Linear(self.feature_dim, num_classes_dict['species'])
        })

        # Initialize the new heads with small random weights
        for head in self.heads.values():
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

        # Store metadata
        self.num_classes_dict = num_classes_dict
        self.taxonomic_ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

        print(f"Created multi-task BEiT model:")
        for rank, n_classes in num_classes_dict.items():
            print(f"  {rank:8s}: {n_classes:5d} classes")

    def forward_features(self, x):
        """
        Extract features from backbone (uses BEiT's forward_features + pooling).

        Args:
            x: Input images (batch_size, 3, 224, 224)

        Returns:
            features: Extracted features (batch_size, feature_dim)
        """
        # Get all patch features from backbone
        x = self.backbone.forward_features(x)  # (batch, 197, 768)

        # Select cls token (first token) - same as BEiT's forward_head
        x = x[:, 0]  # (batch, 768)

        # Apply normalization and dropout (same as BEiT's forward_head)
        x = self.backbone.fc_norm(x)
        x = self.backbone.head_drop(x)

        return x

    def forward(self, x):
        """
        Forward pass through multi-task model.

        Args:
            x: Input images (batch_size, 3, 224, 224)

        Returns:
            dict: Dictionary mapping rank names to logits
                Example: {
                    'phylum': tensor(batch_size, 7),
                    'class': tensor(batch_size, 28),
                    ...
                }
        """
        # Extract shared features from backbone
        features = self.forward_features(x)

        # Classify at each taxonomic level
        outputs = {
            rank: self.heads[rank](features)
            for rank in self.taxonomic_ranks
        }

        return outputs

    def get_num_params(self):
        """Get total number of parameters and breakdown by component."""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        heads_params = sum(p.numel() for p in self.heads.parameters())
        total_params = backbone_params + heads_params

        return {
            'total': total_params,
            'backbone': backbone_params,
            'heads': heads_params,
            'per_head': {rank: sum(p.numel() for p in head.parameters())
                        for rank, head in self.heads.items()}
        }


def create_beit_multitask(
    pretrained: bool = True,
    num_classes_dict: Optional[Dict[str, int]] = None,
    model_name: str = 'hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224'
) -> BEiTMultiTask:
    """
    Create BEiT multi-task model for taxonomic classification.

    Args:
        pretrained: Whether to load pretrained weights
        num_classes_dict: Dict with number of classes per rank
                         If None, uses default FungiTastic counts
        model_name: Name of the base model to load

    Returns:
        BEiTMultiTask model with pretrained backbone and new classification heads
    """
    # Default class counts from FungiTastic dataset
    if num_classes_dict is None:
        num_classes_dict = {
            'phylum': 7,
            'class': 28,
            'order': 95,
            'family': 308,
            'genus': 918,
            'species': 2786  # Updated from 2829 based on actual data
        }

    print(f"Creating BEiT multi-task model...")
    print(f"  Base model: {model_name}")
    print(f"  Pretrained: {pretrained}")

    # Load base model with pretrained weights
    base_model = timm.create_model(model_name, pretrained=pretrained)

    print(f"  Base model loaded: {sum(p.numel() for p in base_model.parameters()) / 1e6:.1f}M params")

    # Create multi-task wrapper
    model = BEiTMultiTask(base_model, num_classes_dict)

    # Print parameter summary
    param_info = model.get_num_params()
    print(f"\nParameter summary:")
    print(f"  Total:     {param_info['total'] / 1e6:.1f}M")
    print(f"  Backbone:  {param_info['backbone'] / 1e6:.1f}M (pretrained)")
    print(f"  Heads:     {param_info['heads'] / 1e6:.1f}M (new)")

    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing BEiT multi-task model creation...")
    print("=" * 60)

    model = create_beit_multitask(pretrained=True)

    print("\n" + "=" * 60)
    print("Testing forward pass with dummy input...")
    print("=" * 60)

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)

    print("\nOutput shapes:")
    for rank, output in outputs.items():
        print(f"  {rank:8s}: {output.shape}")

    print("\n✓ Model creation and forward pass successful!")
