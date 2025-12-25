"""
Multi-task metrics for hierarchical taxonomic classification.

Implements per-rank accuracy, F1 scores, and hierarchical accuracy
to evaluate model performance across all taxonomic levels.
"""

import torch
from typing import Dict
from sklearn.metrics import f1_score
import numpy as np


class MultiTaskMetrics:
    """
    Multi-task metrics for hierarchical taxonomic classification.

    Tracks:
    1. Per-rank accuracy (for each of 6 taxonomic levels)
    2. Per-rank F1 score (macro-averaged to handle class imbalance)
    3. Hierarchical accuracy (all ranks must be correct)
    """

    def __init__(self):
        """Initialize multi-task metrics tracker."""
        self.ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']
        self.reset()

    def reset(self):
        """Reset all metrics."""
        # Store predictions and labels for each rank
        self.predictions = {rank: [] for rank in self.ranks}
        self.labels = {rank: [] for rank in self.ranks}

        # Store hierarchical correctness (all ranks correct)
        self.hierarchical_correct = []
        self.hierarchical_total = 0

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor]
    ):
        """
        Update metrics with batch predictions.

        Args:
            outputs: Dict mapping rank names to logits
                    Example: {'phylum': tensor(B, 7), 'class': tensor(B, 28), ...}
            labels: Dict mapping rank names to target class indices
                   Example: {'phylum': tensor(B), 'class': tensor(B), ...}
        """
        batch_size = labels['phylum'].size(0)

        # Get predictions for each rank
        preds = {}
        for rank in self.ranks:
            # Convert logits to class predictions
            pred = torch.argmax(outputs[rank], dim=1)
            preds[rank] = pred

            # Store for per-rank metrics (convert to numpy)
            self.predictions[rank].extend(pred.cpu().numpy())
            self.labels[rank].extend(labels[rank].cpu().numpy())

        # Compute hierarchical accuracy (all ranks must be correct)
        for i in range(batch_size):
            all_correct = all(
                preds[rank][i] == labels[rank][i]
                for rank in self.ranks
            )
            self.hierarchical_correct.append(all_correct)

        self.hierarchical_total += batch_size

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            dict: Dictionary of metrics
                - '<rank>_acc': Per-rank accuracy
                - '<rank>_f1': Per-rank F1 score (macro)
                - 'hierarchical_acc': Hierarchical accuracy
                - 'avg_acc': Average accuracy across all ranks
                - 'avg_f1': Average F1 across all ranks
        """
        metrics = {}

        # Per-rank metrics
        acc_values = []
        f1_values = []

        for rank in self.ranks:
            preds = np.array(self.predictions[rank])
            lbls = np.array(self.labels[rank])

            # Accuracy
            acc = (preds == lbls).mean()
            metrics[f'{rank}_acc'] = acc
            acc_values.append(acc)

            # F1 score (macro-averaged to handle imbalance)
            # Use zero_division=0 to handle classes with no predictions
            f1 = f1_score(lbls, preds, average='macro', zero_division=0)
            metrics[f'{rank}_f1'] = f1
            f1_values.append(f1)

        # Hierarchical accuracy
        hierarchical_acc = sum(self.hierarchical_correct) / self.hierarchical_total
        metrics['hierarchical_acc'] = hierarchical_acc

        # Average metrics
        metrics['avg_acc'] = np.mean(acc_values)
        metrics['avg_f1'] = np.mean(f1_values)

        return metrics

    def compute_and_reset(self) -> Dict[str, float]:
        """
        Compute metrics and reset.

        Convenience method for end-of-epoch metric computation.
        """
        metrics = self.compute()
        self.reset()
        return metrics


def format_metrics(metrics: Dict[str, float], prefix: str = '') -> str:
    """
    Format metrics dictionary into readable string.

    Args:
        metrics: Dictionary of metrics from compute()
        prefix: Optional prefix (e.g., 'Train' or 'Val')

    Returns:
        str: Formatted metrics string
    """
    ranks = ['phylum', 'class', 'order', 'family', 'genus', 'species']

    lines = []
    if prefix:
        lines.append(f"{prefix} Metrics:")

    # Per-rank metrics
    lines.append("  Per-rank Accuracy:")
    for rank in ranks:
        acc = metrics[f'{rank}_acc']
        lines.append(f"    {rank:8s}: {acc:6.2%}")

    lines.append("  Per-rank F1 Score:")
    for rank in ranks:
        f1 = metrics[f'{rank}_f1']
        lines.append(f"    {rank:8s}: {f1:6.2%}")

    # Summary metrics
    lines.append("  Summary:")
    lines.append(f"    Avg Accuracy:      {metrics['avg_acc']:6.2%}")
    lines.append(f"    Avg F1:            {metrics['avg_f1']:6.2%}")
    lines.append(f"    Hierarchical Acc:  {metrics['hierarchical_acc']:6.2%}")

    return '\n'.join(lines)


if __name__ == '__main__':
    """Test MultiTaskMetrics."""
    print("Testing MultiTaskMetrics...")
    print("=" * 60)

    # Create dummy data
    batch_size = 16
    num_batches = 5

    # Initialize metrics
    metrics_tracker = MultiTaskMetrics()

    print(f"\nSimulating {num_batches} batches of {batch_size} samples each...")

    for batch_idx in range(num_batches):
        # Create dummy outputs (logits)
        outputs = {
            'phylum':  torch.randn(batch_size, 7),
            'class':   torch.randn(batch_size, 28),
            'order':   torch.randn(batch_size, 95),
            'family':  torch.randn(batch_size, 308),
            'genus':   torch.randn(batch_size, 918),
            'species': torch.randn(batch_size, 2786)
        }

        # Create dummy labels
        labels = {
            'phylum':  torch.randint(0, 7, (batch_size,)),
            'class':   torch.randint(0, 28, (batch_size,)),
            'order':   torch.randint(0, 95, (batch_size,)),
            'family':  torch.randint(0, 308, (batch_size,)),
            'genus':   torch.randint(0, 918, (batch_size,)),
            'species': torch.randint(0, 2786, (batch_size,))
        }

        # Update metrics
        metrics_tracker.update(outputs, labels)

    # Compute final metrics
    print(f"\nComputing metrics...")
    metrics = metrics_tracker.compute()

    print("\n" + "=" * 60)
    print(format_metrics(metrics))
    print("=" * 60)

    # Test reset
    print("\nTesting reset...")
    metrics_tracker.reset()
    assert all(len(v) == 0 for v in metrics_tracker.predictions.values())
    assert all(len(v) == 0 for v in metrics_tracker.labels.values())
    print("✓ Reset successful")

    # Test compute_and_reset
    print("\nTesting compute_and_reset...")
    for batch_idx in range(3):
        outputs = {
            'phylum':  torch.randn(batch_size, 7),
            'class':   torch.randn(batch_size, 28),
            'order':   torch.randn(batch_size, 95),
            'family':  torch.randn(batch_size, 308),
            'genus':   torch.randn(batch_size, 918),
            'species': torch.randn(batch_size, 2786)
        }
        labels = {
            'phylum':  torch.randint(0, 7, (batch_size,)),
            'class':   torch.randint(0, 28, (batch_size,)),
            'order':   torch.randint(0, 95, (batch_size,)),
            'family':  torch.randint(0, 308, (batch_size,)),
            'genus':   torch.randint(0, 918, (batch_size,)),
            'species': torch.randint(0, 2786, (batch_size,))
        }
        metrics_tracker.update(outputs, labels)

    metrics2 = metrics_tracker.compute_and_reset()
    assert all(len(v) == 0 for v in metrics_tracker.predictions.values())
    print("✓ Compute and reset successful")

    print("\n" + "=" * 60)
    print("✓ MultiTaskMetrics tests passed!")
    print("=" * 60)
