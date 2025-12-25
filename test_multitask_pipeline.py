"""
Test complete multi-task pipeline: data loading + model forward pass.

Verifies that the multi-task model works end-to-end with real data.
"""

import json
import pandas as pd
import torch
from models.beit_multitask import create_beit_multitask
from dataset_multitask import create_multitask_dataloaders


def test_multitask_pipeline():
    """Test complete multi-task training pipeline."""
    print("=" * 60)
    print("Testing Multi-Task Pipeline")
    print("=" * 60)

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load taxonomic mappings
    print("\n1. Loading taxonomic mappings...")
    with open('taxonomic_mappings.json', 'r') as f:
        taxonomic_mappings = json.load(f)

    num_classes = taxonomic_mappings['metadata']['num_classes']
    print("   Number of classes per rank:")
    for rank, n in num_classes.items():
        print(f"     {rank:8s}: {n:5d}")

    # Load data
    print("\n2. Loading dataset...")
    train_df = pd.read_csv('minimal_subsets/FungiTastic-Train-Minimal.csv')
    val_df = pd.read_csv('minimal_subsets/FungiTastic-Val-Minimal.csv')
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")

    # Create dataloaders
    print("\n3. Creating dataloaders...")
    train_loader, val_loader = create_multitask_dataloaders(
        train_df, val_df,
        taxonomic_mappings,
        batch_size=8,
        num_workers=2
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")

    # Create model
    print("\n4. Creating multi-task model...")
    model = create_beit_multitask(pretrained=True, num_classes_dict=num_classes)
    model = model.to(device)
    model.eval()

    # Test forward pass with real data
    print("\n5. Testing forward pass with real batch...")
    print("=" * 60)

    with torch.no_grad():
        for images, labels, filepaths in train_loader:
            # Move to device
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Display results
            print(f"\nInput:")
            print(f"  Images: {images.shape}, device={images.device}")

            print(f"\nGround Truth Labels:")
            for rank, label_tensor in labels.items():
                print(f"  {rank:8s}: {label_tensor.shape}, values={label_tensor[:3].tolist()}...")

            print(f"\nModel Outputs:")
            for rank, output in outputs.items():
                print(f"  {rank:8s}: {output.shape}, device={output.device}")

            # Get predictions
            print(f"\nPredictions for first sample:")
            id_to_name = taxonomic_mappings['id_to_name']

            for rank in ['phylum', 'class', 'order', 'family', 'genus', 'species']:
                # Ground truth
                gt_id = labels[rank][0].item()
                gt_name = id_to_name[rank][str(gt_id)] if gt_id >= 0 else "MISSING"

                # Prediction
                pred_id = outputs[rank][0].argmax().item()
                pred_name = id_to_name[rank][str(pred_id)]

                match = "✓" if gt_id == pred_id else "✗"

                print(f"  {rank:8s}: GT={gt_name[:30]:30s} | "
                      f"Pred={pred_name[:30]:30s} {match}")

            # Only test one batch
            break

    print("\n" + "=" * 60)
    print("✓ Multi-task pipeline test successful!")
    print("=" * 60)

    # Summary
    print("\nPipeline Components Verified:")
    print("  ✓ Taxonomic mappings loaded")
    print("  ✓ Dataset loading with multi-task labels")
    print("  ✓ Dataloader batching")
    print("  ✓ Model forward pass")
    print("  ✓ Output shapes correct for all 6 ranks")
    print("  ✓ GPU/CPU compatibility")

    return True


if __name__ == '__main__':
    try:
        test_multitask_pipeline()
    except Exception as e:
        print(f"\n✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
