"""
Multi-task loss function for hierarchical taxonomic classification.

Implements weighted sum of cross-entropy losses with per-class balancing
to handle severe class imbalance across taxonomic ranks.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for hierarchical taxonomic classification.

    Combines cross-entropy losses from all 6 taxonomic ranks with:
    1. Rank-level weighting (to balance importance across ranks)
    2. Class-level weighting (to handle severe imbalance within ranks)
    """

    def __init__(
        self,
        rank_weights: Optional[Dict[str, float]] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        device: str = 'cpu'
    ):
        """
        Initialize multi-task loss.

        Args:
            rank_weights: Dict mapping rank names to importance weights
                         If None, uses equal weights
            class_weights: Dict mapping rank names to class weight tensors
                          If None, no class balancing
            device: Device to move class weights to
        """
        super().__init__()

        self.ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

        # Rank weights (relative importance of each taxonomic level)
        if rank_weights is None:
            # Default: normalized inverse of number of classes
            # Gives more weight to harder tasks (more classes)
            default_weights = {
                'phylum':  0.30,  # 7 classes - easy but important
                'class':   0.25,  # 28 classes
                'order':   0.15,  # 95 classes
                'family':  0.10,  # 308 classes
                'genus':   0.10,  # 918 classes
                'species': 0.10   # 2786 classes - hardest
            }
            self.rank_weights = default_weights
        else:
            self.rank_weights = rank_weights

        # Normalize rank weights to sum to 1
        total_weight = sum(self.rank_weights.values())
        self.rank_weights = {k: v/total_weight for k, v in self.rank_weights.items()}

        # Class weights for handling imbalance
        self.class_weights = {}
        if class_weights is not None:
            for rank in self.ranks:
                if rank in class_weights:
                    self.class_weights[rank] = class_weights[rank].to(device)
                else:
                    self.class_weights[rank] = None
        else:
            self.class_weights = {rank: None for rank in self.ranks}

        # Create criterion for each rank
        self.criteria = nn.ModuleDict({
            rank: nn.CrossEntropyLoss(weight=self.class_weights[rank], ignore_index=-1)
            for rank in self.ranks
        })

        self.device = device

        # Print configuration
        print("MultiTaskLoss Configuration:")
        print("  Rank weights (normalized):")
        for rank, weight in self.rank_weights.items():
            print(f"    {rank:8s}: {weight:.3f}")

        if any(w is not None for w in self.class_weights.values()):
            print("  Class balancing: ENABLED")
            for rank in self.ranks:
                if self.class_weights[rank] is not None:
                    w = self.class_weights[rank]
                    print(f"    {rank:8s}: min={w.min():.4f}, max={w.max():.4f}")
        else:
            print("  Class balancing: DISABLED")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ) -> tuple:
        """
        Compute multi-task loss.

        Args:
            outputs: Dict mapping rank names to logits
                    Example: {'phylum': tensor(B, 7), 'class': tensor(B, 28), ...}
            labels: Dict mapping rank names to target class indices
                   Example: {'phylum': tensor(B), 'class': tensor(B), ...}

        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: Scalar tensor (weighted sum of all losses)
                - loss_dict: Dict of per-rank losses for logging
        """
        total_loss = 0.0
        loss_dict = {}

        for rank in self.ranks:
            # Get predictions and targets for this rank
            logits = outputs[rank]
            targets = labels[rank]

            # Compute cross-entropy loss (with class weights if provided)
            loss = self.criteria[rank](logits, targets)

            # Weight by rank importance
            weighted_loss = self.rank_weights[rank] * loss

            # Accumulate
            total_loss = total_loss + weighted_loss

            # Store for logging (unweighted loss)
            loss_dict[f'{rank}_loss'] = loss.item()

        # Also store total loss
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def get_rank_weights(self):
        """Return current rank weights."""
        return self.rank_weights.copy()

    def set_rank_weights(self, new_weights: Dict[str, float]):
        """
        Update rank weights (e.g., for curriculum learning).

        Args:
            new_weights: New weights dict (will be normalized)
        """
        total = sum(new_weights.values())
        self.rank_weights = {k: v/total for k, v in new_weights.items()}

        print("Updated rank weights:")
        for rank, weight in self.rank_weights.items():
            print(f"  {rank:8s}: {weight:.3f}")


if __name__ == '__main__':
    """Test MultiTaskLoss."""
    print("Testing MultiTaskLoss...")
    print("=" * 60)

    # Create dummy data
    batch_size = 8
    outputs = {
        'phylum':  torch.randn(batch_size, 7, requires_grad=True),
        'class':   torch.randn(batch_size, 28, requires_grad=True),
        'order':   torch.randn(batch_size, 95, requires_grad=True),
        'family':  torch.randn(batch_size, 308, requires_grad=True),
        'genus':   torch.randn(batch_size, 918, requires_grad=True),
        'species': torch.randn(batch_size, 2786, requires_grad=True)
    }

    labels = {
        'phylum':  torch.randint(0, 7, (batch_size,)),
        'class':   torch.randint(0, 28, (batch_size,)),
        'order':   torch.randint(0, 95, (batch_size,)),
        'family':  torch.randint(0, 308, (batch_size,)),
        'genus':   torch.randint(0, 918, (batch_size,)),
        'species': torch.randint(0, 2786, (batch_size,))
    }

    # Test without class weights
    print("\n1. Testing without class weights:")
    print("-" * 60)
    criterion = MultiTaskLoss()
    loss, loss_dict = criterion(outputs, labels)

    print(f"\nTotal loss: {loss.item():.4f}")
    print("Per-rank losses:")
    for rank in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
        print(f"  {rank:8s}: {loss_dict[f'{rank}_loss']:.4f}")

    # Test with class weights
    print("\n" + "=" * 60)
    print("2. Testing with class weights:")
    print("-" * 60)

    # Load class weights
    class_weights = torch.load('class_weights.pt')
    criterion_weighted = MultiTaskLoss(class_weights=class_weights)
    loss_weighted, loss_dict_weighted = criterion_weighted(outputs, labels)

    print(f"\nTotal loss: {loss_weighted.item():.4f}")
    print("Per-rank losses:")
    for rank in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
        print(f"  {rank:8s}: {loss_dict_weighted[f'{rank}_loss']:.4f}")

    # Test backward
    print("\n" + "=" * 60)
    print("3. Testing backward pass:")
    print("-" * 60)
    loss.backward()
    print("✓ Backward pass successful")

    print("\n" + "=" * 60)
    print("✓ MultiTaskLoss tests passed!")
    print("=" * 60)
