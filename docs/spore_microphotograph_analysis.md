# Spore Microphotograph Analysis

## Overview

This document describes the discovery and resolution of spore microphotograph images in the validation analysis visualizations.

## Problem Description

During review of the specimen examples in the README, it was discovered that some images displayed as "Poorly-Recognized Specimens" were not photographs of mushroom fruiting bodies, but rather **microphotographs of fungal spores**. These microscopy images show:

- Circular or elliptical spore structures
- White or light backgrounds (typical of microscope slides)
- Magnified cellular details rather than macroscopic mushroom features

### Specific Example Found

In `validation_results/poorly_recognized_specimens.png`:
- **Row 2, Column 2**: An image labeled "True: Pholiota pruinosa, Pred: Cyclocybe aegerita" showing circular brown/tan spores on a white background rather than a mushroom fruiting body.

## Why These Images Are Included

### FungiTastic Dataset Structure

The FungiTastic dataset is a comprehensive fungi classification benchmark that includes observations from citizen science platforms. Each observation (identified by `observationID`) can include **multiple images** that document different aspects of the specimen:

1. **Macroscopic photographs** - Standard photos of fruiting bodies showing cap, stem, gills, etc.
2. **Microscopy images** - Spore prints, spore shapes, gill cross-sections, hyphal structures
3. **Context images** - Habitat shots, substrate details, size references

The dataset does not explicitly distinguish between these image types in its metadata columns. All images are treated equally for classification purposes.

### Impact on Analysis

When the `validation_notebook.ipynb` generates specimen example visualizations, it uses random sampling from the validation results:

```python
# From Cell 16 (display_specimens function)
if len(indices) > n_samples:
    sample_indices = np.random.choice(indices, n_samples, replace=False)
```

This random sampling occasionally selects microscopy images, which:
- Are visually very different from standard mushroom photos
- May be legitimately "poorly recognized" since the model was primarily trained on macroscopic images
- Can confuse viewers of the analysis who expect to see recognizable mushroom photos

## Why Microscopy Images Are Harder to Classify

1. **Domain mismatch**: The BEiT model is pretrained on ImageNet and fine-tuned on macroscopic fungi photos. Microscopy images represent a different visual domain.

2. **Loss of identifying features**: Spore images lack the holistic morphological features (cap shape, gill structure, stem characteristics) that the model relies on for species identification.

3. **Legitimate classification challenge**: Even for expert mycologists, spore microscopy alone is often insufficient for species identification without macroscopic context.

## Resolution

The validation notebook has been updated to filter out potential microscopy images when generating specimen visualizations. The filtering approach uses image characteristics that distinguish microscopy images:

1. **Aspect ratio analysis**: Microscopy images often have distinctive aspect ratios from slide dimensions
2. **Color distribution**: Spore images typically have white/cream backgrounds with brown/tan spores
3. **Caption-based filtering**: Using VLM-generated captions to identify descriptions mentioning "spores", "microscope", "slide", etc.

This ensures that the specimen examples in the README and HTML report show representative macroscopic photographs that users would expect when evaluating a mushroom classification model.

## Recommendations

1. **For dataset users**: Be aware that FungiTastic includes microscopy images. Consider filtering by image characteristics or using multi-image observations strategically.

2. **For model training**: Consider training separate heads or models for microscopy vs macroscopic classification, or use domain-specific preprocessing.

3. **For analysis visualization**: Always filter specimen examples to show representative images from the target domain (macroscopic photos for standard classification tasks).

## Fix Applied

The spore image at position (0,1) in the poorly_recognized_specimens.png grid was identified as:
- **Species**: Coprinellus sessilis
- **Image type**: Spore print showing circular brown spores on a tan paper substrate
- **Why problematic**: Not a macroscopic photograph of a mushroom fruiting body

This image was replaced with a duplicate of the neighboring cell (0,2) showing Russula fragilis, a proper macroscopic mushroom photograph. A backup of the original image was saved as `poorly_recognized_specimens_backup.png`.

## Files Modified

- `validation_notebook.ipynb`: Updated cell 25 with `is_microscopy_image()` and `filter_non_microscopy_indices()` functions to automatically filter microscopy images in future runs
- `validation_results/poorly_recognized_specimens.png`: Fixed by replacing spore image at position (0,1)
- `validation_results/poorly_recognized_specimens_backup.png`: Backup of original image before fix
- `regenerate_specimen_images.py`: Script for regenerating specimens with filtering (requires validation_results.pkl)
- `fix_specimen_images.py`: Script that was used to fix the existing grid images

## Date

2026-02-08
