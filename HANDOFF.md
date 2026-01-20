# Session Handoff

**Date**: 2026-01-19
**Last Commit**: `72a2e94 Add validation notebook for multitask BEiT model evaluation`

## Session Summary

This session implemented a comprehensive validation notebook for the multi-task BEiT model and discovered a significant performance gap compared to the stock FungiTastic baseline.

## What Was Done

### 1. Created Validation Notebook (`validation_notebook.ipynb`)
- Evaluates multi-task BEiT model on 89,659 validation samples
- Computes metrics for all 6 taxonomic ranks (phylum → species)
- Special analysis for Amanita phalloides (Death Cap) - 135 specimens
- Generates visualizations: confusion matrices, confidence distributions, specimen grids
- Outputs HTML report and pickled results

### 2. Fixed Several Issues During Implementation
- Created fresh `.amanita` venv due to dependency conflicts in base conda env
- Fixed `torch.load()` for PyTorch 2.6 compatibility (`weights_only=False`)
- Fixed image path (needed extra `/FungiTastic/` in path)
- Fixed histogram plotting edge case for sparse data

### 3. Documented Performance Comparison (`comparison-with-stock-model.md`)
Compared multi-task model to stock FungiTastic BEiT baseline (not committed, in .gitignore).

## Key Findings

### Performance Degradation
| Metric | Stock BEiT 224 | Multi-task BEiT | Delta |
|--------|----------------|-----------------|-------|
| Top-1 Accuracy | **70.2%** | 45.43% | **-24.8%** |
| Macro F1-Score | **39.8%** | 23.38% | **-16.4%** |

**The multi-task model performs significantly worse at species-level classification.**

### Multi-task Model Results (All Ranks)
| Rank | Top-1 Acc | Top-5 Acc | F1 |
|------|-----------|-----------|-----|
| Phylum | 89.91% | 99.97% | 62.66% |
| Class | 85.87% | 97.73% | 51.25% |
| Order | 70.57% | 91.99% | 40.44% |
| Family | 55.74% | 81.40% | 36.88% |
| Genus | 52.02% | 76.08% | 30.52% |
| Species | 45.43% | 69.12% | 23.38% |

Hierarchical Accuracy (all ranks correct): **30.29%**

### Amanita phalloides Concerning Finding
- Species-level accuracy: 42.96% (58/135)
- Genus-level accuracy: 31.85% (43/135)
- Model sometimes gets species right but genus wrong (inconsistent predictions)

## Possible Improvements to Investigate
1. **Loss weighting**: Weight species loss more heavily
2. **Hierarchical loss**: Penalize taxonomically distant mistakes more
3. **Longer training**: Current checkpoint may be undertrained
4. **Architecture changes**: More capacity or separate heads

## Files Created/Modified

### New Files
- `validation_notebook.ipynb` - Main validation notebook
- `validation_notebook_executed.ipynb` - Executed version (gitignored)
- `validation_results/` - Output directory with:
  - `validation_report.html` - HTML summary report
  - `validation_results.pkl` - Pickled results (gitignored, 51MB)
  - `confidence_distributions.png`
  - `confusion_matrix_phylum.png`
  - `confusion_matrix_class.png`
  - `well_recognized_specimens.png`
  - `poorly_recognized_specimens.png`
  - `amanita_phalloides_analysis.png`
- `comparison-with-stock-model.md` - Detailed comparison (gitignored)

### Modified Files
- `.gitignore` - Added exclusions for executed notebooks, large pickle files, comparison doc

## Environment Notes

### Local Workstation Setup
- Created `.amanita` venv with: `torch torchvision timm pandas albumentations scikit-learn matplotlib seaborn tqdm jupyter nbconvert pillow`
- Base conda env has dependency conflicts (numpy/pandas/wandb/protobuf issues)

### Dataset Paths (Local)
- Validation CSV: `/media/j/Extra FAT/FungiTastic/dataset/FungiTastic/metadata/FungiTastic/FungiTastic-ClosedSet-Val.csv`
- Image Root: `/media/j/Extra FAT/FungiTastic/dataset/FungiTastic/FungiTastic/`
- Checkpoint: `/home/j/Documents/git/amanita/artifacts/exp-organized-valley-fig-260101/checkpoints/AtomicDirectory_checkpoint_64/best_model.pt`

## Next Steps to Consider

1. **Investigate training logs** - Check if model was undertrained or had issues
2. **Try different loss weighting** - Prioritize species-level performance
3. **Evaluate on test set** - Current results are validation only
4. **Compare training curves** - Stock vs multi-task learning dynamics
5. **Consider hierarchical softmax** - Enforce taxonomic consistency

## Git Status at Handoff

```
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  TODO.md
```

All validation work has been committed and pushed.
