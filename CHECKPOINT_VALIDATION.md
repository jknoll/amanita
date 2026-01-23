# Checkpoint Validation Methodology

This document describes the checkpoint validation approach used to identify the optimal early stopping point for the multi-task BEiT fungi classification model.

## Overview

We use **embedded metrics** from checkpoints rather than re-running validation inference. This approach:
- Uses metrics recorded during training (same validation set, same methodology)
- Is significantly faster than re-running inference on all checkpoints
- Provides consistent comparison across checkpoints

## Best Checkpoint Identified

| Metric | Value |
|--------|-------|
| **Experiment** | `exp-tremendous-gentle-passbook-251230` |
| **Checkpoint** | #23 |
| **Species Accuracy** | 50.21% |
| **Average Accuracy** | 70.75% |
| **Hierarchical Accuracy** | 36.43% |
| **Top-5 Species Accuracy** | 73.68% |

### Per-Rank Performance (Best Checkpoint)

| Rank | Top-1 Acc | Top-5 Acc | F1 (Macro) | Avg Confidence |
|------|-----------|-----------|------------|----------------|
| Phylum | 92.39% | 99.97% | 62.72% | 0.969 |
| Class | 88.52% | 98.15% | 55.53% | 0.951 |
| Order | 74.50% | 94.13% | 44.68% | 0.869 |
| Family | 61.07% | 84.72% | 41.99% | 0.766 |
| Genus | 57.81% | 80.64% | 34.71% | 0.719 |
| Species | 50.21% | 73.68% | 26.60% | 0.619 |

## Validation Script

### `validate_all_checkpoints.py`

This script automates checkpoint validation across all experiments:

```bash
python validate_all_checkpoints.py
```

**What it does:**
1. Scans `/media/j/Extra FAT/Amanita-Validation/` for all experiments
2. Finds checkpoint directories (pattern: `AtomicDirectory_checkpoint_*`)
3. Loads each checkpoint and runs validation on FungiTastic validation set
4. Calculates metrics for all 6 taxonomic ranks
5. Generates performance plots and HTML report

**Output files:**
- `checkpoint_validation_results/checkpoint_report.html` - Interactive HTML report
- `checkpoint_validation_results/validation_results.pkl` - Serialized results
- `checkpoint_validation_results/accuracy_by_rank.png` - Metrics vs checkpoint
- `checkpoint_validation_results/species_vs_avg_accuracy.png` - Comparison plot
- `checkpoint_validation_results/best_per_experiment.png` - Bar chart
- `checkpoint_validation_results/accuracy_heatmap.png` - Heatmap visualization

## Interpreting Results

### HTML Report Sections

1. **Summary Statistics**: Total checkpoints, experiments, best accuracy
2. **Best Checkpoint**: Highlighted with path and key metrics
3. **Bracket Search**: Shows neighboring checkpoints and missing ones to sync
4. **Visualizations**: Performance plots embedded in report
5. **All Checkpoint Results**: Sortable table of all validated checkpoints
6. **Detailed Metrics**: Per-rank breakdown for best checkpoint

### Key Findings

1. **Best Overall**: `exp-tremendous-gentle-passbook-251230` checkpoint 23 achieves highest species accuracy (50.21%)

2. **Overfitting Signal**: In `exp-organized-valley-fig-260101`, loss increases from 1.12 (epoch 2) to 1.53 (epoch 6), indicating overfitting after epoch 4

3. **Hierarchical Accuracy**: Only 36.43% of predictions are correct at ALL taxonomic ranks simultaneously

4. **Confidence Calibration**: Higher ranks (phylum, class) show well-calibrated confidence; lower ranks (genus, species) show larger gaps between correct/incorrect prediction confidence

## Bracket Search Strategy

When the best checkpoint is found at an epoch boundary, use bracket search to narrow the optimal point:

1. Identify the best checkpoint
2. Check if neighboring checkpoints exist locally
3. If not, sync additional checkpoints from ISC cluster (see `sync artifacts from ISC.md`)
4. Re-run validation to find the true optimum

### Missing Checkpoints

The report identifies checkpoints that should be synced to complete the bracket search. For `exp-tremendous-gentle-passbook-251230`, consider syncing checkpoints 18-22 to verify CP#23 is optimal.

## Validation Dataset

- **CSV**: `/media/j/Extra FAT/FungiTastic/dataset/FungiTastic/metadata/FungiTastic/FungiTastic-ClosedSet-Val.csv`
- **Images**: `/media/j/Extra FAT/FungiTastic/dataset/FungiTastic/FungiTastic/`
- **Samples**: ~50k validation images

## Model Architecture

- **Base**: BEiT (ViT-B/16) pretrained on ImageNet
- **Modification**: 6 classification heads for taxonomic ranks
- **Input**: 224x224 RGB images
- **Normalization**: ImageNet mean/std (0.485/0.456/0.406, 0.229/0.224/0.225)

## Related Files

| File | Purpose |
|------|---------|
| `validate_all_checkpoints.py` | Checkpoint validation script |
| `validation_notebook.ipynb` | Interactive validation notebook |
| `sync artifacts from ISC.md` | Instructions for syncing checkpoints |
| `checkpoint_validation_results/` | Output directory for reports |
| `taxonomic_mappings.json` | ID-to-name mappings for all ranks |
