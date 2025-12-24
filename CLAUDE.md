# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **amanita** project repository that works with the **FungiTastic** dataset - a fungi classification benchmark consisting of ~350k multimodal observations across 5k fine-grained species. The dataset contains photographs and additional metadata including meteorological data, satellite images, and segmentation masks.

The repository includes:
- Dataset tools and loaders for the FungiTastic dataset
- Multiple baseline implementations for different computer vision tasks
- Pre-trained model inference scripts
- Metadata extraction tools

## Environment Setup

Determine if we are in the local (user workstation) environment, or on the Strong Compute ISC and running the isc-demos image, by inspecting the /opt/ directory for a /venv/. The presence of the /opt/venv indicates the latter, and we should activate and use that environment.

```bash
# Create and activate virtual environment
python -m venv .amanita
source .amanita/bin/activate

# Install basic dependencies (for inference)
pip install -r requirements.txt

# For specific baselines, install their requirements:
pip install -r FungiTastic/baselines/closed_set/requirements.txt
pip install -r FungiTastic/baselines/few_shot/requirements.txt
pip install -r FungiTastic/baselines/segmentation/requirements.txt
```

## Dataset Download

The FungiTastic dataset must be downloaded separately:

```bash
# Install wget if needed
sudo apt-get update && sudo apt-get install wget

# Download dataset (adjust subset and size as needed)
cd FungiTastic/dataset/
python download.py --metadata --images --subset "m" --size "300" --save_path "./"

# Available options:
# --subset: "m" (Mini), "fs" (FewShot), or "full"
# --size: "300", "500", "720", "fullsize"
# --metadata, --images, --masks, --climatic, --satellite
```

## Key Architecture Components

### Dataset Classes

The dataset classes are located in `FungiTastic/dataset/`:

1. **FungiTastic** (`fungi.py`): Base dataset class for image classification
   - Supports multiple subsets: Mini, FewShot, all
   - Supports multiple splits: train, val, test, dna
   - Supports multiple tasks: closed-set, open-set
   - Returns: (image, category_id, file_path)

2. **MaskFungiTastic** (`mask_fungi.py`): Extended class for segmentation tasks
   - Three segmentation modes: binary, semantic, instance
   - Returns: (image, masks, category_id, file_path, labels)

3. **FeatureFungiTastic** (`feature_fungi.py`): For pre-extracted features

All dataset classes inherit from `fgvc.datasets.ImageDataset` (from the FGVC library).

### Model Architecture

Models are loaded via the `timm` library (PyTorch Image Models):

```python
import timm
model = timm.create_model("hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224", pretrained=True)
```

Pre-trained models are available on [HuggingFace Hub](https://huggingface.co/collections/BVRA/fungitastic-66a227ce0520be533dc6403b).

### Class Index Mapping

**CRITICAL**: The model outputs sequential class indices (0, 1, 2, ..., N-1) but the dataset uses `category_id` values that may not be sequential. When performing inference:

1. Load the training metadata CSV to get category_id ↔ scientificName mapping
2. Sort category_ids to align with model output indices
3. Map predicted index → sorted category_id → species name

See `MODEL_INFO.md` for detailed mapping examples.

## Baseline Implementations

The repository contains baseline implementations for multiple tasks in `FungiTastic/baselines/`:

### 1. Closed-Set Classification (`closed_set/`)

Standard supervised classification on known species.

```bash
# Training
cd FungiTastic/baselines/closed_set
python train.py \
    --train-path <path_to_train_metadata.csv> \
    --val-path <path_to_val_metadata.csv> \
    --test-path <path_to_test_metadata.csv> \
    --config-path configs/FungiTastic-224.yaml \
    --wandb-entity <entity> \
    --wandb-project <project>
```

Configuration files in `configs/` and `sweep_configs/` define:
- Model architecture (via timm)
- Data augmentations (e.g., "vit_heavy")
- Loss functions (CrossEntropyLoss, FocalLoss, SeeSawLoss)
- Training hyperparameters

### 2. Few-Shot Classification (`few_shot/`)

Feature-based classification for rare species with limited samples.

**Two-stage pipeline:**

1. Feature extraction:
```bash
cd FungiTastic/baselines/few_shot
python feature_generation.py --config config/fs.yaml
```

2. Evaluation:
```bash
python eval.py --config config/fs.yaml
```

Supports multiple feature extractors (CLIP, DINOv2, BioCLIP) and classifiers (nearest neighbor, prototype/centroid).

### 3. Segmentation (`segmentation/`)

Zero-shot segmentation using GroundingDINO + SAM (Segment Anything Model).

```bash
# Generate masks
cd FungiTastic/baselines/segmentation
python generate_masks.py --config_path config/seg.yaml

# Evaluate
python eval.py --config_path config/seg.yaml
```

**Note**: GroundingDINO requires manual installation following [official guide](https://github.com/IDEA-Research/GroundingDINO#installation). CUDA setup can be tricky.

### 4. Open-Set Classification (`open_set/`)

Classification with unknown species detection (species not in training set).

## Common Development Tasks

### Running Inference

See `test.py` for a complete inference example:

```python
import timm
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
model = timm.create_model("hf-hub:BVRA/beit_base_patch16_224.in1k_ft_fungitastic_224", pretrained=True)
model.eval()

# Prepare image
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

img = Image.open('image.jpg')
output = model(transforms(img).unsqueeze(0))
predicted_idx = torch.argmax(output, dim=1).item()
```

### Loading Dataset

```python
from FungiTastic.dataset.fungi import FungiTastic

dataset = FungiTastic(
    root='path/to/dataset',
    data_subset='Mini',  # 'Mini', 'FewShot', or 'all'
    split='val',         # 'train', 'val', 'test', 'dna'
    size='300',          # '300', '500', '720', 'fullsize'
    task='closed'        # 'closed' or 'open'
)

image, category_id, file_path = dataset[0]
dataset.show_sample(0)  # Visualize
```

### Working with Metadata

Training metadata CSVs are located in `FungiTastic/metadata/<subset>/` and contain:
- `category_id`: Class identifier
- `scientificName`: Species name
- `observationID`: Unique observation ID
- Environmental data: habitat, substrate, elevation, landcover
- Temporal data: eventDate, year, month, day
- Spatial data: latitude, longitude, region
- Taxonomy: kingdom, phylum, class, order, family, genus

## Key Dependencies

- **timm**: PyTorch Image Models for model loading
- **fgvc**: Custom library for fine-grained visual classification (installed from GitHub releases)
- **torch/torchvision**: PyTorch framework
- **pandas**: Metadata handling
- **transformers**: For vision-language models
- **wandb**: Experiment tracking (optional)
- **huggingface_hub**: Model download

## Important Dataset Characteristics

1. **Long-tailed distribution**: Many rare species with few samples
2. **Fine-grained classification**: Visually similar species
3. **Temporal distribution shift**: Data collected over 20 years
4. **DNA-verified test set**: High-quality ground truth for subset of test data
5. **Multimodal**: Images + metadata (location, climate, substrate, etc.)

## Metadata Extraction Tools

Located in `FungiTastic/metadata_extraction/`:
- `captions/`: Extract image captions using vision-language models
- `tabular/`: Extract elevation and land cover data
- `satellite/`: Extract Sentinel satellite data and ecodatacube features
